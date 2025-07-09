import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.metrics import MetricsCalculator, calculate_video_metrics, EarlyStopping


class TestMetricsCalculator:
    """Test MetricsCalculator class."""

    def test_initialization(self):
        """Test MetricsCalculator initialization."""
        calc = MetricsCalculator()
        assert calc.threshold == 0.5
        assert calc.predictions == []
        assert calc.probabilities == []
        assert calc.labels == []

    def test_initialization_custom_threshold(self):
        """Test with custom threshold."""
        calc = MetricsCalculator(threshold=0.7)
        assert calc.threshold == 0.7

    def test_reset(self):
        """Test reset functionality."""
        calc = MetricsCalculator()

        # Add some dummy data
        outputs = torch.randn(5, 1)
        labels = torch.randint(0, 2, (5,))
        calc.update(outputs, labels)

        assert len(calc.predictions) > 0

        # Reset should clear everything
        calc.reset()
        assert calc.predictions == []
        assert calc.probabilities == []
        assert calc.labels == []

    def test_update_single_batch(self):
        """Test update with single batch."""
        calc = MetricsCalculator()

        outputs = torch.tensor([[2.0], [-1.0], [0.5], [-0.5]])  # logits
        labels = torch.tensor([1, 0, 1, 0])

        calc.update(outputs, labels)

        assert len(calc.predictions) == 4
        assert len(calc.probabilities) == 4
        assert len(calc.labels) == 4

    def test_update_multiple_batches(self):
        """Test update with multiple batches."""
        calc = MetricsCalculator()

        # First batch
        outputs1 = torch.tensor([[1.0], [-1.0]])
        labels1 = torch.tensor([1, 0])
        calc.update(outputs1, labels1)

        # Second batch
        outputs2 = torch.tensor([[0.5], [-0.5], [2.0]])
        labels2 = torch.tensor([1, 0, 1])
        calc.update(outputs2, labels2)

        assert len(calc.predictions) == 5
        assert len(calc.probabilities) == 5
        assert len(calc.labels) == 5

    def test_sigmoid_conversion(self):
        """Test that logits are correctly converted to probabilities."""
        calc = MetricsCalculator()

        # Large positive logit should give probability close to 1
        outputs = torch.tensor([[10.0], [-10.0]])
        labels = torch.tensor([1, 0])
        calc.update(outputs, labels)

        probs = np.array(calc.probabilities)
        assert probs[0] > 0.99  # Should be close to 1
        assert probs[1] < 0.01  # Should be close to 0

    def test_threshold_application(self):
        """Test threshold application for predictions."""
        calc = MetricsCalculator(threshold=0.7)

        # Test with probabilities that cross the threshold
        outputs = torch.tensor(
            [[1.0], [0.5], [-0.5]]
        )  # Will give probs ~0.73, 0.62, 0.38
        labels = torch.tensor([1, 1, 0])
        calc.update(outputs, labels)

        preds = np.array(calc.predictions)
        # Only first should be > 0.7 threshold
        assert preds[0] == 1
        assert preds[1] == 0  # 0.62 < 0.7
        assert preds[2] == 0

    def test_compute_metrics_perfect_predictions(self):
        """Test metrics computation with perfect predictions."""
        calc = MetricsCalculator()

        # Perfect predictions
        outputs = torch.tensor([[5.0], [5.0], [-5.0], [-5.0]])
        labels = torch.tensor([1, 1, 0, 0])
        calc.update(outputs, labels)

        metrics = calc.compute_metrics()

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["auc_roc"] == 1.0

    def test_compute_metrics_with_errors(self):
        """Test metrics computation with some errors."""
        calc = MetricsCalculator()

        # 3 correct, 1 incorrect
        outputs = torch.tensor(
            [[5.0], [-5.0], [-5.0], [5.0]]
        )  # Predictions: 1, 0, 0, 1
        labels = torch.tensor([1, 0, 1, 1])  # True labels
        calc.update(outputs, labels)

        metrics = calc.compute_metrics()

        # TP=2, FP=0, TN=1, FN=1
        assert metrics["accuracy"] == 0.75  # (2+1)/4
        assert metrics["precision"] == 1.0  # 2/(2+0)
        assert metrics["recall"] == 2 / 3  # 2/(2+1)
        assert "f1_score" in metrics
        assert "auc_roc" in metrics

    def test_compute_metrics_confusion_matrix(self):
        """Test confusion matrix components in metrics."""
        calc = MetricsCalculator()

        outputs = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]])
        labels = torch.tensor([1, 0, 0, 1])  # TP=1, FP=1, TN=1, FN=1
        calc.update(outputs, labels)

        metrics = calc.compute_metrics()

        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["true_negatives"] == 1
        assert metrics["false_negatives"] == 1
        assert "specificity" in metrics

    def test_compute_metrics_single_class(self):
        """Test metrics with only one class present."""
        calc = MetricsCalculator()

        # Only positive examples
        outputs = torch.tensor([[1.0], [2.0], [0.5]])
        labels = torch.tensor([1, 1, 1])
        calc.update(outputs, labels)

        metrics = calc.compute_metrics()

        # Should handle gracefully, AUC should be 0 since only one class
        assert metrics["accuracy"] == 1.0
        assert metrics["auc_roc"] == 0.0

    def test_compute_metrics_empty(self):
        """Test metrics computation with no data."""
        calc = MetricsCalculator()

        metrics = calc.compute_metrics()

        assert metrics == {}

    def test_get_confusion_matrix(self):
        """Test confusion matrix generation."""
        calc = MetricsCalculator()

        outputs = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]])
        labels = torch.tensor([1, 0, 0, 1])
        calc.update(outputs, labels)

        cm = calc.get_confusion_matrix()

        assert cm.shape == (2, 2)
        # cm[0,0] = TN, cm[0,1] = FP, cm[1,0] = FN, cm[1,1] = TP
        assert cm[1, 1] == 1  # TP
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[0, 0] == 1  # TN

    def test_get_confusion_matrix_empty(self):
        """Test confusion matrix with no data."""
        calc = MetricsCalculator()

        cm = calc.get_confusion_matrix()

        assert len(cm) == 0

    def test_get_classification_report(self):
        """Test classification report generation."""
        calc = MetricsCalculator()

        outputs = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]])
        labels = torch.tensor([1, 0, 1, 0])
        calc.update(outputs, labels)

        report = calc.get_classification_report()

        assert isinstance(report, str)
        assert "Non-Violent" in report
        assert "Violent" in report
        assert "precision" in report
        assert "recall" in report

    def test_get_classification_report_empty(self):
        """Test classification report with no data."""
        calc = MetricsCalculator()

        report = calc.get_classification_report()

        assert report == "No predictions available"


class TestCalculateVideoMetrics:
    """Test calculate_video_metrics function."""

    def test_mean_aggregation(self):
        """Test video metrics with mean aggregation."""
        # Create dummy video predictions
        video1_preds = torch.tensor([0.8, 0.9, 0.7])  # Should aggregate to ~0.8
        video2_preds = torch.tensor([0.2, 0.1, 0.3])  # Should aggregate to ~0.2

        predictions = [video1_preds, video2_preds]
        labels = [torch.tensor(1), torch.tensor(0)]

        metrics = calculate_video_metrics(predictions, labels, aggregation="mean")

        # Should be perfect predictions with mean aggregation
        assert metrics["accuracy"] == 1.0

    def test_max_aggregation(self):
        """Test video metrics with max aggregation."""
        video1_preds = torch.tensor([0.3, 0.9, 0.2])  # Max = 0.9
        video2_preds = torch.tensor([0.1, 0.2, 0.3])  # Max = 0.3

        predictions = [video1_preds, video2_preds]
        labels = [torch.tensor(1), torch.tensor(0)]

        metrics = calculate_video_metrics(predictions, labels, aggregation="max")

        assert metrics["accuracy"] == 1.0

    def test_majority_aggregation(self):
        """Test video metrics with majority aggregation."""
        video1_preds = torch.tensor([0.8, 0.8, 0.2])  # Majority > 0.5
        video2_preds = torch.tensor([0.2, 0.2, 0.8])  # Majority < 0.5

        predictions = [video1_preds, video2_preds]
        labels = [torch.tensor(1), torch.tensor(0)]

        metrics = calculate_video_metrics(predictions, labels, aggregation="majority")

        assert metrics["accuracy"] == 1.0

    def test_invalid_aggregation(self):
        """Test with invalid aggregation method."""
        predictions = [torch.tensor([0.8]), torch.tensor([0.2])]
        labels = [torch.tensor(1), torch.tensor(0)]

        with pytest.raises(ValueError, match="Unknown aggregation method"):
            calculate_video_metrics(predictions, labels, aggregation="invalid")

    def test_multiple_videos(self):
        """Test with multiple videos."""
        predictions = [
            torch.tensor([0.8, 0.9]),  # High confidence positive
            torch.tensor([0.1, 0.2]),  # High confidence negative
            torch.tensor([0.7, 0.8]),  # Medium confidence positive
            torch.tensor([0.3, 0.2]),  # Medium confidence negative
        ]
        labels = [torch.tensor(1), torch.tensor(0), torch.tensor(1), torch.tensor(0)]

        metrics = calculate_video_metrics(predictions, labels, aggregation="mean")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics


class TestEarlyStopping:
    """Test EarlyStopping class."""

    def test_initialization_min_mode(self):
        """Test initialization in min mode."""
        early_stop = EarlyStopping(patience=5, min_delta=0.01, mode="min")

        assert early_stop.patience == 5
        assert early_stop.min_delta == 0.01
        assert early_stop.mode == "min"
        assert early_stop.counter == 0
        assert early_stop.best_value == float("inf")
        assert not early_stop.early_stop

    def test_initialization_max_mode(self):
        """Test initialization in max mode."""
        early_stop = EarlyStopping(patience=3, min_delta=0.001, mode="max")

        assert early_stop.mode == "max"
        assert early_stop.best_value == float("-inf")

    def test_improvement_detection_min_mode(self):
        """Test improvement detection in min mode."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        # First value should be improvement
        should_stop = early_stop(1.0)
        assert not should_stop
        assert early_stop.best_value == 1.0
        assert early_stop.counter == 0

        # Better value should be improvement
        should_stop = early_stop(0.8)
        assert not should_stop
        assert early_stop.best_value == 0.8
        assert early_stop.counter == 0

    def test_improvement_detection_max_mode(self):
        """Test improvement detection in max mode."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, mode="max")

        # First value
        should_stop = early_stop(0.5)
        assert not should_stop
        assert early_stop.best_value == 0.5

        # Better value
        should_stop = early_stop(0.8)
        assert not should_stop
        assert early_stop.best_value == 0.8

    def test_no_improvement_counter(self):
        """Test counter increment when no improvement."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        # Set initial value
        early_stop(1.0)

        # No improvement (same value)
        should_stop = early_stop(1.0)
        assert not should_stop
        assert early_stop.counter == 1

        # Still no improvement
        should_stop = early_stop(1.005)  # Less than min_delta improvement
        assert not should_stop
        assert early_stop.counter == 2

    def test_early_stopping_trigger(self):
        """Test early stopping trigger."""
        early_stop = EarlyStopping(patience=2, min_delta=0.01, mode="min")

        # Set initial value
        early_stop(1.0)

        # No improvement for patience steps
        early_stop(1.0)  # counter = 1
        should_stop = early_stop(1.0)  # counter = 2, should trigger

        assert should_stop
        assert early_stop.early_stop

    def test_min_delta_threshold(self):
        """Test min_delta threshold behavior."""
        early_stop = EarlyStopping(patience=3, min_delta=0.1, mode="min")

        early_stop(1.0)

        # Small improvement (less than min_delta)
        should_stop = early_stop(0.95)  # Improvement of 0.05 < 0.1
        assert not should_stop
        assert early_stop.counter == 1  # Should count as no improvement

        # Large improvement (greater than min_delta)
        should_stop = early_stop(0.8)  # Improvement of 0.15 > 0.1
        assert not should_stop
        assert early_stop.counter == 0  # Should reset counter

    def test_reset_functionality(self):
        """Test reset functionality."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        # Trigger some state changes
        early_stop(1.0)
        early_stop(1.0)  # counter = 1

        # Reset
        early_stop.reset()

        assert early_stop.counter == 0
        assert not early_stop.early_stop
        assert early_stop.best_value == float("inf")

    def test_patience_zero(self):
        """Test with patience=0."""
        early_stop = EarlyStopping(patience=0, min_delta=0.01, mode="min")

        early_stop(1.0)
        should_stop = early_stop(1.0)  # No improvement, should stop immediately

        assert should_stop

    def test_edge_case_values(self):
        """Test with edge case values."""
        early_stop = EarlyStopping(patience=2, min_delta=0.0, mode="min")

        # Test with very small improvements
        early_stop(1.0)
        should_stop = early_stop(0.999999)  # Tiny improvement
        assert not should_stop
        assert early_stop.counter == 0

        # Test with exact same value
        should_stop = early_stop(0.999999)
        assert not should_stop
        assert early_stop.counter == 1


class TestMetricsIntegration:
    """Integration tests for metrics components."""

    def test_full_training_simulation(self):
        """Simulate a full training cycle with metrics and early stopping."""
        calc = MetricsCalculator()
        early_stop = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        # Simulate training epochs
        epoch_losses = [
            1.0,
            0.8,
            0.7,
            0.69,
            0.688,
            0.687,
            0.686,
        ]  # Diminishing improvements

        for epoch, loss in enumerate(epoch_losses):
            # Simulate some predictions
            outputs = torch.randn(10, 1)
            labels = torch.randint(0, 2, (10,))
            calc.update(outputs, labels)

            should_stop = early_stop(loss)

            if should_stop:
                break

        # Should have stopped before the end due to small improvements
        assert early_stop.early_stop

        # Should have accumulated metrics
        metrics = calc.compute_metrics()
        assert len(metrics) > 0


if __name__ == "__main__":
    pytest.main([__file__])
