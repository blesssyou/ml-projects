# Machine Learning Portfolio

This repository contains academic machine learning and deep learning projects implemented using Python and PyTorch.

## Projects

### 1. Bird Audio Classification

**Technologies:** Python, PyTorch, Scikit-learn, Hugging Face Transformers, Librosa

**Objective:** Binary classification of bird audio recordings using handcrafted audio features and pretrained deep learning representations.

**Implemented approaches:**
- Extracted MFCC features from audio recordings and prepared feature vectors for classification.
- Used pretrained CLAP audio embeddings as an alternative feature representation.
- Applied dimensionality reduction techniques to visualize feature spaces.
- Compared multiple classical machine learning classifiers using classification metrics.
- Implemented neural classification heads on top of pretrained audio representations.
- Compared frozen feature extraction and end-to-end fine-tuning strategies.
- Analyzed audio samples in feature space and visualized their spectrograms.

[**Jupyter Notebook**](./bird-audio-classification/bird_audio_classification.ipynb)

---

### 2. CNN Transfer Learning

**Technologies:** Python, PyTorch, Torchvision

**Objective:** Multi-task image classification using a shared convolutional feature extractor and task-specific classification heads.

**Datasets:** MNIST and Fashion-MNIST.

**Implemented approaches:**
- Designed a CNN with shared convolutional layers and separate classification heads for two classification tasks.
- Trained the model on MNIST and evaluated its classification performance.
- Investigated transfer learning by freezing shared feature extraction layers and training a new classification head.
- Compared frozen-feature training, fine-tuning, and training from scratch.
- Experimented with freezing different subsets of convolutional layers.
- Analyzed training and test learning curves across different training strategies.
- Explored cross-task class similarities using model predictions.

[**Jupyter Notebook**](./cnn-transfer-learning/cnn_transfer_learning.ipynb)

---

### 3. CIFAR-10 Autoencoder & Conditional VAE

**Technologies:** Python, PyTorch, Torchvision

**Objective:** Image reconstruction and conditional image generation using deep generative models.

**Dataset:** CIFAR-10.

**Implemented approaches:**
- Implemented and trained a convolutional autoencoder for image reconstruction.
- Designed a 64-dimensional latent representation.
- Visualized original and reconstructed images and analyzed training curves.
- Explored Gaussian and SMOTE-based sampling in both image and latent spaces.
- Implemented a conditional variational autoencoder for class-conditioned image generation.
- Visualized generated images and explored conditional generation from class combinations.

[**Jupyter Notebook**](./cifar10-autoencoder-cvae/cifar10_autoencoder_cvae.ipynb)
