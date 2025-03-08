# CLEAR GLASS: Contrastive Learning with Enhanced Abstract Representations using Grouped Loss of Abstract Semantic Supervision

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

**CLEAR GLASS** (Contrastive Learning with Enhanced Abstract Representations using Grouped Loss of Abstract Semantic Supervision) is a novel approach to enhance abstract concept inference in images. Humans can recognize an image as an instance of a general concept, beyond simply identifying its objects and their relationships. In this paper, we investigate 1. The extent to which VLMs have this concept abstraction capacity, and 2. Strategies for encoding the sort of higher-concept information in images that would enable the resulting VLM model (CLEAR GLASS model) to have this capability to a greater degree. To this end, we introduce a grouped image-caption dataset (MAGIC), which consists of several groups of image captions and for each group a set of associated images and higher-level conceptual labels. We use a novel contrastive loss technique to induce the model to encode in the representation of each image (caption) in a group the information that is common to all members of the image-caption group.  Our main contribution is a grouped contrastive loss function based on text-image contrastive groups (outer contrastive loss) as well as an inner loss which measures the distances between image-caption instances in the group. Our training methodology results in the CLEAR GLASS model having the concept abstraction capacity as an emergent capacity because the model is not exposed to the higher-level concepts associated with each group. Instead, the training forces the model to create for each image-caption group a semantic representation that brings it closer to the semantic representation of the higher-level concepts in the latent semantic space. Our experiments show that this training methodology results in a model which shows improvement in abstract concept recognition compared to SOTA models.

## Prerequisites

- **Python 3.x**
- **PyTorch**
- **CUDA-compatible GPU** (for training)
- **MS COCO 2017 Dataset**
- Additional libraries (install necessary libraries based on the import statements in the code)

## Installation
   ```bash
   git clone https://github.com/ai-lab-conf/CLEAR_GLASS.git
   cd DRIVE
   ```

### Training Configuration

Default parameters:

- **Groups per batch**: 2
- **Epochs**: 7
- **Learning Rate**: 1e-08
- **Pairs per group**: 10
- **Temprature**: 0.1
- **Precision**: Mixed Precision (AMP)
- **Gradient Clipping Norm**: 1.0
- **Alpha (α)**: 0.7

#### Customizing Training Parameters

To modify training parameters:

- **Edit** `scripts/train_script.sh` with your desired configurations.
- **Directly pass arguments** when running the training script:

  ```bash
  python train.py --epochs 10 --learning_rate 2e-5
  ```

## Evaluation

After training, evaluation begins automatically if the provided script is used. 
The evaluation script displays metrics for relation inference accuracy of all the relevant models.
