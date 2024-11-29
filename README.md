# **GPU Accelerated Neural Network with MNIST Dataset**

## **Overview**
This project implements a fully connected neural network trained on the MNIST dataset, leveraging CUDA and cuBLAS for GPU acceleration. The model is designed to demonstrate forward propagation, backpropagation, and gradient updates on GPU hardware. Additionally, OpenGL is used to visualize the training accuracy and error over time.

---

## **Features**
- **GPU Accelerated Neural Network:**
  - Forward propagation and backpropagation using CUDA and cuBLAS.
  - Sigmoid activation function with CUDA kernel for efficient computation.
  - Cross-entropy loss for multi-class classification.
  - Gradient updates using stochastic gradient descent (SGD).
- **MNIST Dataset Processing:**
  - Loading and preprocessing of MNIST images and labels.
  - Conversion of data into GPU-compatible formats.
- **Training Visualization:**
  - Real-time visualization of training accuracy and error using OpenGL.
  - Interactive graph plotted after each training batch.

---

## **Technologies Used**
- **CUDA**: For GPU-based parallel computation.
- **cuBLAS**: For optimized matrix operations on GPU.
- **OpenGL**: For real-time visualization of training metrics.
- **C++**: Core programming language for implementing the neural network and integrating libraries.

---

## **How It Works**
1. **Data Loading and Preprocessing**:
   - The MNIST dataset (images and labels) is loaded from binary files.
   - Images are normalized, and labels are converted into one-hot encoding.
2. **Neural Network Initialization**:
   - The network is constructed with customizable layers.
   - Weights and biases are initialized randomly and stored on the GPU.
3. **Forward Propagation**:
   - Input data is processed layer by layer using matrix multiplication and the sigmoid activation function.
4. **Backward Propagation**:
   - Gradients are computed using the chain rule.
   - Cross-entropy loss and sigmoid derivative are used for error propagation.
5. **Weight Updates**:
   - Stochastic Gradient Descent (SGD) is applied to update weights and biases based on gradients.
6. **Training Visualization**:
   - OpenGL plots the training accuracy and error after every batch.

---

## **Setup and Installation**
### **Requirements**
- **CUDA**: Ensure a CUDA-capable GPU and CUDA toolkit installed.
- **cuBLAS**: Comes with the CUDA toolkit.
- **OpenGL**: Required for visualization (GLFW is used as a wrapper).
- **C++ Compiler**: Support for C++11 or later.
