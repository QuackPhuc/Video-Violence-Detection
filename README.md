# Video Violence Detection

Deep learning model for detecting violence in video sequences using Two-Stream Gated 3D CNN architecture.

## 📊 Dataset

**Processed dataset**: [Video Violence Detection Dataset](https://www.kaggle.com/datasets/quackphuc/video-violence-detection)

- Pre-processed video clips with RGB + Optical Flow features
- Split into train/validation/test sets
- 6-channel input format: `[RGB(3) + Flow_XY(2) + Flow Magnitude(1)]`

## 🏆 Best Model

**Pre-trained weights**: [Video Violence Detection Model](https://www.kaggle.com/models/quackphuc/vidviodetection/)

## 🎯 Training Strategy

The model was trained through **3 progressive phases**:

### Phase 1: Initial Training (20 epochs)

- Learning Rate: `3e-4`
- Dropout: `0.2`
- Focus: Feature learning and initial convergence

### Phase 2: Fine-tuning (20 epochs)

- Learning Rate: `1e-4`
- Dropout: `0.3`
- Focus: Stability and regularization

### Phase 3: Final Optimization (10 epochs)

- Learning Rate: `1e-4`
- Dropout: `0.5`
- Focus: Overfitting prevention and final tuning

## 📈 Performance Results

**Test Set Evaluation (200 samples):**

### 📊 Overall Performance Metrics

| Metric          | Value      |
| --------------- | ---------- |
| **Accuracy**    | **86.00%** |
| **Precision**   | **90.00%** |
| **Recall**      | **81.00%** |
| **F1-Score**    | **85.26%** |
| **AUC-ROC**     | **92.98%** |
| **Specificity** | **91.00%** |

### 🎯 Confusion Matrix

|                        | Predicted Non-Violent | Predicted Violent |
| ---------------------- | --------------------- | ----------------- |
| **Actual Non-Violent** | 91 (TN)               | 9 (FP)            |
| **Actual Violent**     | 19 (FN)               | 81 (TP)           |

### 📋 Detailed Classification Report

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Non-Violent**  | 0.83      | 0.91   | 0.87     | 100     |
| **Violent**      | 0.90      | 0.81   | 0.85     | 100     |
| **Macro Avg**    | 0.86      | 0.86   | 0.86     | 200     |
| **Weighted Avg** | 0.86      | 0.86   | 0.86     | 200     |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Multi-GPU Training

```bash
!torchrun --nproc_per_node=2 src/training/train.py \
    --train-data /kaggle/input/video-violence-detection/train \
    --val-data /kaggle/input/video-violence-detection/val \
    --train-csv /kaggle/input/viviod-labeling/train.csv \
    --val-csv /kaggle/input/viviod-labeling/val.csv \
    --batch-size 2
```

### 💡 Hardware Recommendations

For **16GB VRAM GPUs**, use `--batch-size 2` to avoid memory overflow:

```bash
!torchrun --nproc_per_node=1 src/training/train.py \
    --train-data /path/to/train \
    --val-data /path/to/val \
    --train-csv /path/to/train.csv \
    --val-csv /path/to/val.csv \
    --batch-size 2
```

## 🏗️ Architecture

- **Two-Stream Gated 3D CNN** for spatiotemporal feature extraction
- **RGB Stream**: Appearance features from video frames
- **Flow Stream**: Motion features from optical flow
- **Gated Fusion**: Multiplicative attention mechanism
- **Progressive Training**: Multi-phase learning strategy

## 📝 Requirements

- Python 3.8+
- PyTorch 2.4.0+
- CUDA-capable GPU(s) with more than 16GB VRAM recommended
- See `requirements.txt` for full dependencies

## 📄 License

MIT License
