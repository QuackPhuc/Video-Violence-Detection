import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any, Tuple
import os
import gc
import json
import time
from tqdm import tqdm
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping helper to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        """Check if should stop training."""
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpoint."""
        return {
            "counter": self.counter,
            "best_value": self.best_value,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from checkpoint."""
        self.counter = state_dict["counter"]
        self.best_value = state_dict["best_value"]
        self.early_stop = state_dict.get("early_stop", False)


class MetricTracker:
    """Track and aggregate metrics during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def update(self, metric_dict: Dict[str, float], n: int = 1):
        """Update metrics with new values."""
        for key, value in metric_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            self.metrics[key] += value * n
            self.counts[key] += n

    def average(self) -> Dict[str, float]:
        """Get averaged metrics."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics
            if self.counts[key] > 0
        }


class VideoActionTrainer:
    """Trainer for video action recognition models with DDP support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        rank: int,
        world_size: int,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: str = "./checkpoints",
        resume: Optional[str] = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        log_interval: int = 10,
        save_interval_epochs: int = 1,
        max_runtime_hours: Optional[float] = None,
        time_based_save: bool = False,
        save_interval_minutes: float = 30.0,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            rank: Process rank for DDP
            world_size: Total number of processes
            device: Device to run on
            scheduler: Learning rate scheduler
            checkpoint_dir: Directory for checkpoints
            resume: Path to checkpoint to resume from
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Minimum improvement for early stopping
            log_interval: Interval for logging metrics
            save_interval_epochs: Interval for saving checkpoints (epochs)
            max_runtime_hours: Maximum runtime in hours before stopping
            time_based_save: Enable time-based checkpoint saving
            save_interval_minutes: Time interval for time-based saves (minutes)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.save_interval_epochs = save_interval_epochs

        # Runtime management
        self.max_runtime_hours = max_runtime_hours
        self.time_based_save = time_based_save
        self.save_interval_minutes = save_interval_minutes
        self.training_start_time = None
        self.last_time_save = None
        self.runtime_exceeded = False
        self._early_stopped = False

        # Training state
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.training_history = {"train": [], "val": []}

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="min",
        )

        # Metric trackers
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()

        # Create checkpoint directory
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            logger.info(
                f"Trainer initialized for DDP with world size: {self.world_size}"
            )
            if self.max_runtime_hours:
                logger.info(f"Maximum runtime set to {self.max_runtime_hours} hours")
            if self.time_based_save:
                logger.info(
                    f"Time-based saving enabled every {self.save_interval_minutes} minutes"
                )

        # Resume from checkpoint if provided
        if resume:
            self.load_checkpoint(resume)

    def _synchronize_metric(self, metric_tensor: torch.Tensor) -> torch.Tensor:
        """Average a metric across all processes."""
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        metric_tensor /= self.world_size
        return metric_tensor

    def _compute_metrics(
        self, outputs: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for a batch."""
        with torch.no_grad():
            # Loss is already computed
            preds = (torch.sigmoid(outputs) > 0.5).float()
            accuracy = (preds == labels).float().mean().item()

            # Additional metrics can be added here
            metrics = {"accuracy": accuracy}

        return metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        # Set epoch for sampler
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            disable=(self.rank != 0),
            leave=False,
        )

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Compute metrics
            metrics = self._compute_metrics(outputs, labels)
            metrics["loss"] = loss.item()

            # Update tracker
            self.train_metrics.update(metrics, n=inputs.size(0))

            # Update progress bar
            if self.rank == 0 and batch_idx % self.log_interval == 0:
                avg_metrics = self.train_metrics.average()
                progress_bar.set_postfix(**avg_metrics)

        # Synchronize metrics across processes
        avg_metrics = self.train_metrics.average()
        for key in avg_metrics:
            metric_tensor = torch.tensor([avg_metrics[key]], device=self.device)
            self._synchronize_metric(metric_tensor)
            avg_metrics[key] = metric_tensor.item()

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return avg_metrics

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch} [Val]",
            disable=(self.rank != 0),
            leave=False,
        )

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Compute metrics
                metrics = self._compute_metrics(outputs, labels)
                metrics["loss"] = loss.item()

                # Update tracker
                self.val_metrics.update(metrics, n=inputs.size(0))

        # Synchronize metrics
        avg_metrics = self.val_metrics.average()
        for key in avg_metrics:
            metric_tensor = torch.tensor([avg_metrics[key]], device=self.device)
            self._synchronize_metric(metric_tensor)
            avg_metrics[key] = metric_tensor.item()

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return avg_metrics

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_type: str = "regular",
    ):
        """Save a checkpoint."""
        if self.rank != 0:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "early_stopping": self.early_stopping.state_dict(),
            "training_history": self.training_history,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "runtime_hours": self._get_elapsed_hours(),
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        if checkpoint_type == "time_based":
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"time_checkpoint_epoch_{epoch}.pth"
            )
        else:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
            )
        torch.save(checkpoint, checkpoint_path)

        # Save as latest
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")

        # Save training history
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found at {checkpoint_path}")
            return

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model and optimizer states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]

        # Load early stopping state
        if "early_stopping" in checkpoint:
            self.early_stopping.load_state_dict(checkpoint["early_stopping"])

        # Load training history
        if "training_history" in checkpoint:
            self.training_history = checkpoint["training_history"]

        if self.rank == 0:
            logger.info(f"Resumed from epoch {self.start_epoch}")

    def _get_elapsed_hours(self) -> float:
        """Get elapsed training time in hours."""
        if self.training_start_time is None:
            return 0.0
        return (time.time() - self.training_start_time) / 3600.0

    def _check_runtime_limit(self) -> bool:
        """Check if runtime limit has been exceeded."""
        if self.max_runtime_hours is None:
            return False

        elapsed = self._get_elapsed_hours()
        if elapsed >= self.max_runtime_hours:
            if self.rank == 0:
                logger.warning(
                    f"Runtime limit exceeded: {elapsed:.2f}/{self.max_runtime_hours} hours"
                )
            return True
        return False

    def _should_save_time_based(self) -> bool:
        """Check if it's time for a time-based save."""
        if not self.time_based_save:
            return False

        current_time = time.time()
        if self.last_time_save is None:
            self.last_time_save = current_time
            return False

        elapsed_minutes = (current_time - self.last_time_save) / 60.0
        if elapsed_minutes >= self.save_interval_minutes:
            self.last_time_save = current_time
            return True
        return False

    def _save_final_state(self, epoch: int, reason: str = "training_completed"):
        """Save final training state with evaluation."""
        if self.rank == 0:
            logger.info(f"Saving final state - Reason: {reason}")

        # Perform final validation
        val_metrics = self._validate_epoch(epoch)

        # Check if this is the best model
        val_loss = val_metrics["loss"]
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        # Save checkpoint
        self.save_checkpoint(epoch, is_best=is_best, metrics=val_metrics)

        # Log final metrics
        if self.rank == 0:
            elapsed = self._get_elapsed_hours()
            logger.info(
                f"Final validation - Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}"
            )
            logger.info(f"Total training time: {elapsed:.2f} hours")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def train(self, num_epochs: int):
        """Train the model."""
        if self.rank == 0:
            logger.info(f"Starting training for {num_epochs} epochs")

        # Initialize training start time
        if self.training_start_time is None:
            self.training_start_time = time.time()
            self.last_time_save = self.training_start_time

        for epoch in range(self.start_epoch, num_epochs + 1):
            # Check runtime limit before starting epoch
            runtime_exceeded = self._check_runtime_limit()
            if runtime_exceeded:
                self.runtime_exceeded = True
                if self.rank == 0:
                    logger.warning("Runtime limit exceeded, stopping training")
                self._save_final_state(epoch - 1, "runtime_limit_exceeded")
                break

            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate_epoch(epoch)

            # Update history
            self.training_history["train"].append(train_metrics)
            self.training_history["val"].append(val_metrics)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Check for improvement
            val_loss = val_metrics["loss"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                # Save the best model checkpoint immediately
                if self.rank == 0:
                    self.save_checkpoint(epoch, is_best=True, metrics=val_metrics)

            # Early stopping check
            should_stop = self.early_stopping(val_loss)

            # Logging
            if self.rank == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = self._get_elapsed_hours()
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f} - "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f} - "
                    f"LR: {lr:.6f} - "
                    f"Runtime: {elapsed:.2f}h"
                )

                if should_stop:
                    logger.info("Early stopping triggered")

            # Check for time-based save
            should_save_time_based = self._should_save_time_based()

            # Save periodic/latest checkpoint
            should_save_epoch = epoch % self.save_interval_epochs == 0
            if should_save_epoch or should_stop or should_save_time_based:
                checkpoint_type = "time_based" if should_save_time_based else "regular"

                self.save_checkpoint(
                    epoch,
                    is_best=False,
                    metrics=val_metrics,
                    checkpoint_type=checkpoint_type,
                )

                if should_save_time_based and self.rank == 0:
                    elapsed = self._get_elapsed_hours()
                    logger.info(f"Time-based checkpoint saved at {elapsed:.2f} hours")

            # Synchronize early stopping decision
            stop_tensor = torch.tensor([float(should_stop)], device=self.device)
            dist.broadcast(stop_tensor, src=0)

            if stop_tensor.item() > 0:
                if self.rank == 0:
                    logger.info("Training stopped by early stopping")
                self._early_stopped = True
                self._save_final_state(epoch, "early_stopping")
                break

            # Check runtime limit after epoch (in case it was exceeded during training)
            runtime_exceeded = self._check_runtime_limit()
            if runtime_exceeded:
                self.runtime_exceeded = True
                if self.rank == 0:
                    logger.warning(
                        "Runtime limit exceeded after epoch, stopping training"
                    )
                self._save_final_state(epoch, "runtime_limit_exceeded")
                break

        # Final synchronization
        dist.barrier()

        if self.rank == 0:
            if not self.runtime_exceeded:
                # Check if we have stop_tensor (might not exist if no early stopping check occurred)
                early_stopped = hasattr(self, "_early_stopped") and self._early_stopped
                if not early_stopped:
                    logger.info("Training completed successfully")
                    elapsed = self._get_elapsed_hours()
                    logger.info(f"Total training time: {elapsed:.2f} hours")
                else:
                    logger.info("Training completed")
            elif self.runtime_exceeded:
                logger.info("Training terminated due to runtime limit")
            else:
                logger.info("Training completed")
