import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.two_stream_cnn import TwoStreamGated3DCNN
from src.data.dataset import NpzVideoDataset
from src.data.augmentation import VideoAugmentation
from src.training.trainer import VideoActionTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up distributed training."""
    dist.destroy_process_group()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video Violence Detection Training")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )

    # Override config options
    parser.add_argument(
        "--train-data", type=str, help="Override training data directory"
    )
    parser.add_argument(
        "--val-data", type=str, help="Override validation data directory"
    )
    parser.add_argument("--train-csv", type=str, help="Override training csv file")
    parser.add_argument("--val-csv", type=str, help="Override validation csv file")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--num-epochs", type=int, help="Override number of epochs")
    parser.add_argument("--weight-decay", type=float, help="Override weight decay")
    parser.add_argument("--gradient-clip", type=float, help="Override gradient clip")
    parser.add_argument("--patience", type=int, help="Override patience")
    parser.add_argument(
        "--dropout-prob", type=float, help="Override dropout probability"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, help="Override checkpoint directory"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--no-augmentation", action="store_true", help="Disable data augmentation"
    )

    # Runtime management
    parser.add_argument(
        "--max-runtime",
        type=float,
        help="Maximum runtime in hours (will save checkpoint and stop when reached)",
    )
    parser.add_argument(
        "--time-based-save",
        action="store_true",
        help="Enable time-based checkpoint saving",
    )
    parser.add_argument(
        "--save-interval-minutes",
        type=float,
        help="Interval in minutes for time-based checkpoint saving",
    )
    parser.add_argument(
        "--epoch-save-step",
        type=int,
        help="Override epoch save step (save every N epochs)",
    )

    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> dict:
    """Load configuration from file and override with command line args."""
    # Default configuration
    config = {
        "data": {
            "train_dir": "/path/to/train",
            "val_dir": "/path/to/val",
            "train_csv": "/path/to/train.csv",
            "val_csv": "/path/to/val.csv",
            "num_workers": 2,
            "cache_size": None,
        },
        "model": {
            "num_classes": 1,
            "dropout_prob": 0.5,
            "feature_channels": [32, 64, 128, 256],
        },
        "training": {
            "batch_size": 4,
            "num_epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "gradient_clip": 1.0,
        },
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "patience": 5,
            "factor": 0.5,
            "min_lr": 1e-6,
        },
        "early_stopping": {"patience": 10, "min_delta": 1e-4},
        "augmentation": {"enabled": True, "seed": 42},
        "checkpoint": {
            "dir": "./checkpoints",
            "save_interval_epochs": 1,
            "time_based_save": False,
            "save_interval_minutes": 60.0,
        },
        "runtime": {
            "max_runtime_hours": 11.5,  # No time limit by default
            "track_runtime": True,
        },
    }

    # Load from file if exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f)
            # Deep merge configurations
            deep_merge(config, file_config)

    # Override with command line arguments
    if args.train_data:
        config["data"]["train_dir"] = args.train_data
    if args.val_data:
        config["data"]["val_dir"] = args.val_data
    if args.train_csv:
        config["data"]["train_csv"] = args.train_csv
    if args.val_csv:
        config["data"]["val_csv"] = args.val_csv
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.weight_decay:
        config["training"]["weight_decay"] = args.weight_decay
    if args.gradient_clip:
        config["training"]["gradient_clip"] = args.gradient_clip
    if args.patience:
        config["early_stopping"]["patience"] = args.patience
    if args.dropout_prob:
        config["model"]["dropout_prob"] = args.dropout_prob
    if args.checkpoint_dir:
        config["checkpoint"]["dir"] = args.checkpoint_dir
    if args.no_augmentation:
        config["augmentation"]["enabled"] = False

    # Runtime management overrides
    if args.max_runtime:
        config["runtime"]["max_runtime_hours"] = args.max_runtime
    if args.time_based_save:
        config["checkpoint"]["time_based_save"] = True
    if args.save_interval_minutes:
        config["checkpoint"]["save_interval_minutes"] = args.save_interval_minutes
    if args.epoch_save_step:
        config["checkpoint"]["save_interval_epochs"] = args.epoch_save_step

    return config


def deep_merge(base: dict, update: dict):
    """Deep merge two dictionaries."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def create_data_loaders(config: dict, rank: int, world_size: int):
    """Create data loaders for training and validation."""
    # Create augmentation if enabled
    augmentor = None
    if config["augmentation"]["enabled"]:
        augmentor = VideoAugmentation(seed=config["augmentation"]["seed"])

    # Create datasets
    train_dataset = NpzVideoDataset(
        data_dir=config["data"]["train_dir"],
        csv_file=config["data"]["train_csv"],
        transform=augmentor,
        channel_first=True,
        cache_size=config["data"]["cache_size"],
    )

    val_dataset = NpzVideoDataset(
        data_dir=config["data"]["val_dir"],
        csv_file=config["data"]["val_csv"],
        transform=None,  # No augmentation for validation
        channel_first=True,
        cache_size=config["data"]["cache_size"],
    )

    # Create samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=train_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=val_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def create_model(config: dict, device: torch.device):
    """Create and initialize model."""
    model = TwoStreamGated3DCNN(
        num_classes=config["model"]["num_classes"],
        dropout_prob=config["model"]["dropout_prob"],
        feature_channels=tuple(config["model"]["feature_channels"]),
    )
    return model.to(device)


def create_optimizer(model: nn.Module, config: dict):
    """Create optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Create learning rate scheduler."""
    scheduler_config = config["scheduler"]
    scheduler_type = scheduler_config["type"]

    if scheduler_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=scheduler_config["patience"],
            factor=scheduler_config["factor"],
            min_lr=scheduler_config["min_lr"],
        )
    elif scheduler_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["num_epochs"],
            eta_min=scheduler_config.get("min_lr", 0),
        )
    else:
        return None


def main():
    # Parse arguments
    args = parse_args()

    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Load configuration
    config = load_config(args.config, args)

    # Log configuration
    if rank == 0:
        logger.info("Training Configuration:")
        logger.info(yaml.dump(config, default_flow_style=False))

    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, rank, world_size)

    # Create model
    model = create_model(config, device)
    model = DDP(model, device_ids=[local_rank])

    # Create criterion
    criterion = nn.BCEWithLogitsLoss()

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Create trainer
    trainer = VideoActionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        rank=rank,
        world_size=world_size,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=config["checkpoint"]["dir"],
        resume=args.resume,
        early_stopping_patience=config["early_stopping"]["patience"],
        early_stopping_min_delta=config["early_stopping"]["min_delta"],
        save_interval_epochs=config["checkpoint"]["save_interval_epochs"],
        max_runtime_hours=config["runtime"]["max_runtime_hours"],
        time_based_save=config["checkpoint"]["time_based_save"],
        save_interval_minutes=config["checkpoint"]["save_interval_minutes"],
    )

    # Train model
    trainer.train(num_epochs=config["training"]["num_epochs"])

    # Cleanup
    cleanup_ddp()


if __name__ == "__main__":
    main()
