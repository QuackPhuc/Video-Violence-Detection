import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various metrics for binary classification."""

    def __init__(self, threshold: float = 0.5):
        """Initialize metrics calculator.

        Args:
            threshold: Decision threshold for binary classification
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset accumulated predictions and labels."""
        self.predictions = []
        self.probabilities = []
        self.labels = []

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Update with new predictions and labels.

        Args:
            outputs: Model outputs (logits)
            labels: Ground truth labels
        """
        with torch.no_grad():
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > self.threshold).astype(int)
            labels = labels.cpu().numpy()

            self.probabilities.extend(probs.flatten())
            self.predictions.extend(preds.flatten())
            self.labels.extend(labels.flatten())

    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary containing all computed metrics
        """
        if not self.predictions:
            logger.warning("No predictions to compute metrics")
            return {}

        labels = np.array(self.labels)
        predictions = np.array(self.predictions)
        probabilities = np.array(self.probabilities)

        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1_score": f1_score(labels, predictions, zero_division=0),
        }

        # Add AUC if we have both classes
        if len(np.unique(labels)) > 1:
            metrics["auc_roc"] = roc_auc_score(labels, probabilities)
        else:
            metrics["auc_roc"] = 0.0

        # Add confusion matrix
        cm = confusion_matrix(labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)

            # Calculate specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics["specificity"] = specificity

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix.

        Returns:
            Confusion matrix as numpy array
        """
        if not self.predictions:
            return np.array([])

        return confusion_matrix(self.labels, self.predictions)

    def get_classification_report(self) -> str:
        """Get detailed classification report.

        Returns:
            Classification report as string
        """
        from sklearn.metrics import classification_report

        if not self.predictions:
            return "No predictions available"

        return classification_report(
            self.labels, self.predictions, target_names=["Non-Violent", "Violent"]
        )


def calculate_video_metrics(
    predictions: List[torch.Tensor],
    labels: List[torch.Tensor],
    aggregation: str = "mean",
) -> Dict[str, float]:
    """Calculate metrics for video-level predictions.

    Args:
        predictions: List of predictions for each video
        labels: List of labels for each video
        aggregation: How to aggregate frame predictions ('mean', 'max', 'majority')

    Returns:
        Dictionary of video-level metrics
    """
    calculator = MetricsCalculator()

    for pred_frames, label in zip(predictions, labels):
        if aggregation == "mean":
            video_pred = pred_frames.mean()
        elif aggregation == "max":
            video_pred = pred_frames.max()
        elif aggregation == "majority":
            video_pred = (pred_frames > 0.5).float().mean()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        calculator.update(video_pred.unsqueeze(0), label.unsqueeze(0))

    return calculator.compute_metrics()


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self, patience: int = 10, min_delta: float = 0.0001, mode: str = "min"
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        """Check if should stop.

        Args:
            value: Current metric value

        Returns:
            True if should stop training
        """
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
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs without improvement"
                )

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.early_stop = False
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
