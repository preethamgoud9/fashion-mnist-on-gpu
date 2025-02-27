
# Fashion-MNIST on GPU

##  Overview
This project demonstrates the implementation of an **Artificial Neural Network (ANN)** for classifying images from the **Fashion-MNIST dataset**. The model is trained on a **GPU** to accelerate computations using **PyTorch**.

Fashion-MNIST is a dataset of grayscale images of clothing items, designed as a drop-in replacement for the classic MNIST dataset of handwritten digits. This project showcases how to train a deep learning model efficiently using GPU acceleration.

##  Features
- Implements an **ANN** to classify Fashion-MNIST images.
- Utilizes **PyTorch** for model building and training.
- Runs on **GPU** for faster training (using CUDA-enabled devices).
- Includes **data preprocessing, model training, and evaluation**.

## 📂 Repository Structure
```
📦 fashion-mnist-on-gpu
├── ann_fashion_mnist_gpu.ipynb   # Jupyter Notebook containing the full implementation
├── README.md                     # Project documentation
└── requirements.txt              # List of dependencies
```

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/preethamgoud9/fashion-mnist-on-gpu.git
cd fashion-mnist-on-gpu
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv env  # Create a virtual environment
source env/bin/activate  # On macOS/Linux
env\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Jupyter Notebook
```bash
jupyter notebook
```
- Open **`ann_fashion_mnist_gpu.ipynb`** and run all cells to train the model.

## 🛠 Model Architecture
The ANN model consists of the following layers:
- **Input Layer**: Accepts 28x28 grayscale images (flattened to 784 features).
- **Hidden Layers**: Fully connected layers with ReLU activations.
- **Output Layer**: 10 classes representing different fashion items.
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: Adam.

## 📊 Dataset Information
Fashion-MNIST contains **70,000 images** (60,000 for training and 10,000 for testing) across **10 categories**:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## ⚡ GPU Acceleration
This project is optimized for **CUDA-enabled GPUs**. If a GPU is available, PyTorch will automatically utilize it. To verify GPU availability, run:
```python
import torch
print(torch.cuda.is_available())  # Should return True if a GPU is available
```

## 🔍 Results & Performance
After training, the model achieves **high accuracy** on the test set. The final accuracy and loss values are displayed in the notebook.

##  Future Improvements
- Implement **CNNs** to improve accuracy.
- Experiment with different **optimizers and learning rates**.
- Introduce **data augmentation** techniques.

## 🤝 Contributing
Feel free to contribute by:
- Improving model performance.
- Adding explanations or documentation.
- Extending the notebook with more experiments.

Fork the repository, make changes, and submit a pull request! 🚀

## 📜 License
This project is licensed under the **MIT License**. You are free to use, modify, and distribute it as per the license terms.

---

💡 **Developed by [Preetham Goud](https://github.com/preethamgoud9/)**

