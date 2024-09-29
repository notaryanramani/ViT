# ViT
Implementation of "An Image is Worth 16x16 Words" paper
![Vit](https://viso.ai/wp-content/uploads/2021/09/vision-transformer-vit.png)

## Contents
- [1. What is ViT?](#1-what-is-vit)
- [2. Dataset Used](#2-dataset-used)
- [3. The Training Script](#3-the-training-script)
- [4. Contribution](#4-contribution)

---

## 1. What is ViT?
The **Vision Transformer (ViT)** is a deep learning model architecture that adapts the Transformer, originally designed for NLP tasks, to computer vision. The core idea of ViT is to treat images as sequences of patches (similar to tokens in NLP), allowing self-attention mechanisms to learn global dependencies between patches.

### Key Features of ViT:
- **Patch Embeddings**: Instead of using convolutional layers, the input image is divided into fixed-size patches, and each patch is linearly embedded.
- **Self-Attention**: ViT utilizes self-attention mechanisms to capture long-range dependencies between patches.
- **Class Token**: A learnable class token is appended to the patch sequence for classification purposes.
- **Position Embeddings**: Since transformers are permutation invariant, position embeddings are added to patches to retain spatial information.
  
### Paper and Reference:
You can read the original ViT paper [here](https://arxiv.org/abs/2010.11929). The title of the paper is **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** by Dosovitskiy et al.

---

## 2. Dataset Used
### V1
The **CIFAR-10** dataset is used to train the ViT model in this project. CIFAR-10 is a popular benchmark dataset for image classification tasks and contains 60,000 color images of size 32x32 pixels, divided into 10 different classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

### Dataset Features:
- **Number of images**: 60,000 images, with 50,000 for training and 10,000 for testing.
- **Resolution**: 32x32 pixels.
- **Classes**: 10 mutually exclusive classes.
  
You can download the dataset and read more about it from the official CIFAR-10 page [here](https://www.cs.toronto.edu/~kriz/cifar.html).

---

## 3. The Training Script
The `main.py` script contains the complete training loop for the ViT model on the CIFAR-10 dataset. Below are the key functionalities of the script:

### Key Components:
- **Model Initialization**: The ViT model is instantiated, and the number of layers, hidden units, and patch size are defined.
- **Training Loop**: A loop that trains the model across multiple epochs using the CIFAR-10 dataset. The model uses cross-entropy loss and is optimized via AdamW.
- **Validation**: After every epoch, the model is validated on a holdout set to ensure it generalizes well.

---

## 4. Contribution
To contribute to this project, you need to set up your development environment by following the steps below:

### Using Virtual Environment (`venv`)
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/notaryanramani/ViT.git
   cd ViT
   ```

2. Setup python environment:
    Using python
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    or using conda
    ```bash
    conda create --name vit-project python=3.8 
    conda activate vit-project
    ```

3. Install required dependencies
    ```bash
    pip install -r requirements.txt
    ```




