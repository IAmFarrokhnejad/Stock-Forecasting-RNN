# Time Series Forecasting with LSTM and GRU Models

This repository contains Python code for time series forecasting using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) neural networks implemented in PyTorch. The models are trained and evaluated on a dataset named `amazon.csv`. (provided in this repository)

## Features
- **Model Architectures:** Includes LSTM and GRU implementations for sequence modeling.
- **Data Preprocessing:** Time series data is prepared with scaling, sequence generation, and train-validation splitting.
- **Model Training:** Comprehensive training loop with loss computation and model validation.
- **Performance Metrics:** Training and validation loss are reported for each epoch to track model performance.

## Requirements
- Python 3.8 or later
- PyTorch 2.0 or later
- NumPy
- Matplotlib
- scikit-learn

Install the required libraries using:
```bash
pip install torch numpy matplotlib scikit-learn
```

### Hardware
- A CUDA-compatible GPU is recommended for efficient training.

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/IAmFarrokhnejad/Stock-Forecasting-RNN
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision matplotlib seaborn scikit-learn
   ```

3. **Prepare Your Dataset**:

- Ensure the amazon.csv file is placed in the root directory. The file should contain a single time series column for analysis.
- In case of changing the data directory, update the data variable path in the script to point to your dataset path.


## Running the Code

Execute the script to train both the LSTM and GRU models:
```bash
python stock-forecasting.py
```

The script will:
1. Train and validate models sequentially for EfficientNet, MobileNetV2, and ShuffleNet.
2. Save the classification report and confusion matrix for each model in the specified directory.

---

## Results

The models were trained for 40 epochs, and the following trends were observed:

1. LSTM:

- Training Loss decreased to 0.0001 by epoch 40.
- Validation Loss stabilized at 0.0016.

2. GRU:

- Training Loss decreased to 0.0001 by epoch 40.
- Validation Loss stabilized at 0.0019.

Both models demonstrate strong convergence, with GRU showing slightly slower validation loss improvement compared to LSTM.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

[Morteza Farrokhnejad](https://github.com/IAmFarrokhnejad)

---

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework.
- Matplotlib for visualization.