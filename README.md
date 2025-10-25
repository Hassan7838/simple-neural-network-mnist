# simple-neural-network-mnist 
"Simple Neural Network Development for Digit Classification on MNIST Dataset"

This project focuses on building and training a **foundational neural network** using **TensorFlow/Keras** to classify handwritten digits from the **MNIST dataset**.  
The goal is to understand how activation functions affect model performance by comparing two simple feed-forward networks — one using **ReLU** and the other using **Sigmoid** activation functions.

Developed as part of my Machine Learning internship, this project helped strengthen my understanding of **deep learning basics**, **activation functions**, and **model training visualization**.

# Project Goals

To build, train, and evaluate a simple neural network that:
- Classifies handwritten digits (0–9) from the MNIST dataset.
- Experiments with two activation functions: **ReLU** and **Sigmoid**.
- Visualizes accuracy and loss curves for both models.
- Compares convergence speed, performance, and overall training behavior.

# Project Workflow

### 1. Dataset Loading and Preprocessing
- Loaded the **MNIST dataset** directly from TensorFlow/Keras datasets.
- Performed preprocessing:
  - Normalized pixel values to a 0–1 range.
  - Flattened 28×28 images into 1D vectors (for dense layers).
  - One-hot encoded the labels for 10-class classification.
- Split the dataset into **training (60,000 images)** and **testing (10,000 images)** sets.

### 2. Neural Network Architecture and Training
- Built a **simple feed-forward neural network** using the Keras Sequential API.
- Designed two models:
  - **Model A:** Uses **ReLU** activation in hidden layers.
  - **Model B:** Uses **Sigmoid** activation in hidden layers.
- Compiled both models with:
  - Optimizer: `adam`
  - Loss: `categorical_crossentropy`
  - Metric: `accuracy`
- Trained both models for a fixed number of epochs while tracking accuracy and loss.

### 3. Performance Visualization
For each model:
- Plotted **training vs. validation accuracy** across epochs.
- Plotted **training vs. validation loss** across epochs.
- Used **Matplotlib** for clear, well-labeled graphs.

### 4. Comparative Analysis
- Compared the performance of **ReLU** vs. **Sigmoid** models in terms of:
  - Convergence speed
  - Final accuracy
  - Loss behavior
- Discussed the impact of activation functions on learning efficiency and gradient flow.

# Tools and Libraries
- **Python 3.x**
- **TensorFlow / Keras** – For neural network modeling and training  
- **NumPy** – For numerical operations  
- **Matplotlib** – For visualizing accuracy and loss curves
  
# Dataset Information

**Dataset:** [MNIST Handwritten Digit Classification (Kaggle)](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)  
- The dataset is **not uploaded** due to size constraints.  
- You can download it directly from Kaggle and place it in your working directory.

## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/simple-neural-network-mnist.git
