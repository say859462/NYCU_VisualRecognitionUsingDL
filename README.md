# NYCU Computer Vision 2026 HW1

- **Student ID:** 314551113  
- **Name:** 劉哲良  

## Introduction

This homework is an image classification system for the NYCU Visual Recognition using Deep Learning Homework 1 dataset. The task contains **100 classes** of plants and animals, and the main challenge is **fine-grained visual classification (FGVC)**, where many classes differ only in subtle local details such as wing patterns, flower structures, leaf textures, or small shape differences. The dataset also contains small objects, background clutter, and noticeable intra-class variation.

The final implementation uses a **ResNet-152** backbone with lightweight **partial Res2Net-style adapters**, a **PMG-inspired multi-granularity design** with global / part2 / part4 branches, **sub-center classifiers**, and a **reliability-aware consensus fusion head** for the final prediction. The core idea is to keep strong global semantics while preserving finer regional evidence for visually similar classes.

The training pipeline follows a staged PMG schedule. Early training emphasizes the global branch, then gradually adds the coarse branch, and finally enables the full PMG fusion stage. The implementation uses **AdamW**, **warm-up cosine annealing**, and a curriculum of training geometries from `Resize(512) + RandomCrop(448)` to `Resize(576) + RandomCrop(512)`, while validation / test use deterministic `Resize(576)` preprocessing.

## Environment Setup

Recommended environment: **Python 3.11+** with CUDA-enabled PyTorch.

Install the required packages:

```bash
pip install torch torchvision pandas tqdm matplotlib numpy pillow scipy gdown
```


## Usage

### Training

Train the model with the default configuration:

```bash
python main.py --config ./config.json
```

This script loads the PMG model, builds the train / validation dataloaders, applies staged training geometry, and saves the best checkpoints according to validation accuracy and validation loss.

### Inference

Generate the final submission file:

```bash
python test.py --config ./config.json --output_csv prediction.csv
```

The inference script loads `best_model_path` from `config.json`, runs prediction on the test split, and exports a CSV with columns `image_name` and `pred_label`.

### Analysis

Run branch-level analysis on the validation set:

```bash
python analyze.py --config ./config.json --resize 576
```

This script reports:
- global / part2 / part4 / concat accuracy
- rescue / failure cases between branches
- confidence and top-2 gap statistics
- per-class analysis CSV files

It is useful for understanding whether the fusion head is actually benefiting from multi-granularity evidence.

### Grad-CAM Visualization

Visualize attention and branch activation maps:

```bash
python gradcam_vis.py --config ./config.json
```

This script saves qualitative visualizations for the original image, branch-specific CAMs, and the learned attention map, which helps inspect whether the model attends to the correct object region.

## Performance Snapshot

Main observations from the homework:
- The PMG-style multi-granularity design improved representation quality by separating global, coarse, and fine evidence.
- The final system uses a **reliability-aware consensus fusion** module that combines branch logits together with branch-confidence statistics, rather than relying on a naive concatenation rule.


<img width="1280" height="60" alt="image" src="https://github.com/user-attachments/assets/f11a06d2-9ea5-480c-9d1b-e75d940356e0" />

