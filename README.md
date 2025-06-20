# Rainfall Forecasting with Neural Networks

## Project Summary

This project implements a deep learning model to classify whether it will rain, based on key meteorological variables. Built using **PyTorch**, the model is trained on real-world weather data to learn subtle interactions between atmospheric factors like temperature, pressure, and humidity. The challenge lies in accurately predicting relatively rare rainfall events, which is handled by using a **hybrid loss function** combining **Binary Cross-Entropy** with an **AUC-based objective**.

---

## Dataset Overview

- **Source**: Kaggle Playground Series (Season 5, Episode 3)
- **Format**: CSV (training and test datasets)
- **Features**:
  - `pressure`, `maxtemp`, `temperature`, `mintemp`, `dewpoint`
  - `humidity`, `cloud`, `sunshine`, `winddirection`, `windspeed`
- **Label**: `rainfall` (binary: 1 for rain, 0 for no rain)

---

## Neural Network Architecture

### Preprocessing
- Normalize all continuous features using z-score standardization.
- Handle missing data (if any) during preprocessing.
- Encode data as PyTorch tensors for GPU training.

### Model Design

The architecture follows a **fully connected feedforward neural network**:
- **Input Layer**: Receives weather features.
- **Hidden Layers**: Dense layers activated with `ReLU`.
- **Output Layer**: Single neuron with `Sigmoid` activation for binary output.

Mathematically:

$$\hat{y} = \sigma(W_n \cdot \text{ReLU}(W_{n-1} \cdots \text{ReLU}(W_1 x + b_1) + \cdots + b_{n-1}) + b_n)$$

---

## Loss Function & Optimization

To enhance performance on imbalanced rainfall labels, the model minimizes a **hybrid loss**:

$$\mathcal{L} = \text{Binary Cross Entropy} + \lambda \cdot (1 - \text{AUC})$$

- **Binary Cross Entropy (BCE)** handles overall classification.
- **AUC Loss** helps rank positive examples above negatives.
- Optimized using the **Adam** optimizer with gradient descent:

$$\theta = \theta - \eta \cdot \nabla_\theta \mathcal{L}$$

---

## Evaluation Metrics

Model performance is evaluated on:
- **AUC-ROC**: Measures ranking quality across thresholds
- **Precision**: Accuracy on predicted rain events
- **Recall**: Sensitivity to actual rain occurrences
- **Accuracy**: General correctness across both classes

---

## Key Outcomes

- The model achieved strong **recall** on rainy samples, ensuring low false negatives.
- AUC scores indicate consistent separation between rainy and non-rainy data.
- Demonstrated robustness on test data with minimal overfitting due to normalization and simple architecture.

---

## Reference

> Walter Reade & Elizabeth Park.  
> *"Binary Prediction with a Rainfall Dataset"*, 2025.  
