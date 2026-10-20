# Attention U-Net for Carotid Artery Segmentation in Ultrasound Images

Binary segmentation of artery in ultrasound images, using an **Attention U-Net** implemented from scratch in PyTorch. 
The whole pipeline (data loading, augmentation, model, loss functions, metrics, hyperparameter search, training and inference) lives in a single Kaggle notebook.

| Test metric | Value |
|---|---|
| Dice score | **0.893** |
| IoU | **0.836** |
| Hausdorff distance (normalized by the image diagonal) | 0.046 (≈ 17 px on 256 × 256 images) |
| BCE loss | 0.014 |

---

## Dataset

- **Source:** *Common Carotid Artery Ultrasound Images*, A. Momot, Mendeley Data, v1, 2022 — [doi:10.17632/d4xt63mgjm.1](https://data.mendeley.com/datasets/d4xt63mgjm/1) (also mirrored on Kaggle). I first worked with this dataset during a deep learning course by Dr. Clément Beitone (TIMC Laboratory, GMCAO team).
- **Content:** 1,100 ultrasound images (PNG, 709 × 749) with expert segmentation masks, acquired on 11 subjects (Mindray UMT-500Plus, L13-3s linear probe).
- The data is **not included** in this repository. On Kaggle, add the dataset to the notebook as an input.

## Method

### Preprocessing
- Random split: **64 % train / 16 % validation / 20 % test** (704 / 176 / 220 images, `random_state=42`).
- Images and masks resized to **256 × 256**; images normalized to [-1, 1].
- Training augmentations: horizontal flip, color jitter (brightness, contrast, saturation, hue), small random affine transform (±3°, scale 0.95–1.05), random grayscale.
- Batch size: 4.

### Model
Attention U-Net ([Oktay et al., 2018](https://arxiv.org/abs/1804.03999)) written from scratch:
- **Encoder:** 4 down-sampling blocks (double 3 × 3 convolution + BatchNorm + ReLU, then 2 × 2 max pooling), from 32 to 256 channels.
- **Bottleneck:** double convolution with 512 channels.
- **Decoder:** 4 up-sampling blocks (bilinear interpolation). Each **attention gate** filters the encoder skip connection using the decoder features as a gating signal, before concatenation.
- **Output:** 1 × 1 convolution producing one logit per pixel (sigmoid applied in the loss and the metrics).

### Training
- Optimizer: **AdamW** (weight decay 1e-4).
- `ReduceLROnPlateau` (factor 0.5, patience 4) and **early stopping** (patience 7) on the validation loss.
- **Grid search** on the validation set (12 configurations):

| Hyperparameter | Values tested |
|---|---|
| Loss function | BCE (with logits), soft Dice loss, combined loss (0.5 BCE + 0.5 Dice) |
| Max epochs | 20, 40 |
| Learning rate | 1e-4, 1e-5 |

- **Best configuration:** BCE loss, learning rate 1e-4, 40 epochs (best validation loss: 0.012).
- Hardware: Kaggle GPU (NVIDIA T4), about 4.7 h for the full grid search.

### Metrics
All metrics are implemented from scratch on binarized predictions (threshold 0.5):
- **Dice score** and **IoU**, averaged over images.
- **Hausdorff distance** (bidirectional, `scipy.spatial.distance.directed_hausdorff`), normalized by the image diagonal (√(256² + 256²) ≈ 362 px).

## Results

On the 220 test images, the model reaches a **Dice score of 0.893** and an **IoU of 0.836**. The notebook shows the training curves and several test predictions, with the ground-truth contour (green), the predicted contour (red) and their overlap (yellow).

## How to run

1. Open the notebook [`us-seg-attention-unet.ipynb`](us-seg-attention-unet.ipynb) on Kaggle.
2. Add the *Common Carotid Artery Ultrasound Images* dataset as an input.
3. Enable a GPU accelerator and run all cells.

Main dependencies: `torch`, `torchvision`, `torchsummary`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `Pillow`, `tqdm`.

## Repository structure

```
.
├── us-seg-attention-unet.ipynb   # full pipeline: data, model, training, inference
└── README.md
```

## Author

**Julie Lasry** — Machine Learning Engineer, medical imaging.
Feel free to reach out on LinkedIn.

## Acknowledgments

- Dataset: A. Momot, *Common Carotid Artery Ultrasound Images*, Mendeley Data, 2022.
- Dr. Clément Beitone (TIMC Laboratory, GMCAO team) for the deep learning course where I first used this dataset.
- O. Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, MIDL 2018.
