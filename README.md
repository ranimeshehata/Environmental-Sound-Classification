# Environmental-Sound-Classification

This project focuses on classifying environmental audio clips from the UrbanSound8K dataset using deep learning models: a Bi-directional LSTM (BiLSTM) and a Transformer (implemented from scratch).

---

## 📁 Dataset

- **Name:** UrbanSound8K  
- **Samples:** 8,732 audio clips (≤ 4 seconds)
- **Classes:**  
  1. Air Conditioner  
  2. Car Horn  
  3. Children Playing  
  4. Dog Bark  
  5. Drilling  
  6. Engine Idling  
  7. Gun Shot  
  8. Jackhammer  
  9. Siren  
  10. Street Music

- **Splits:**  
  - Folds 1–6: Training  
  - Folds 7–8: Validation  
  - Folds 9–10: Testing

---

## 🧪 Feature Extraction

Each audio file is transformed into a 45-dimensional feature vector including:

- 13 MFCCs  
- Delta & Delta-Delta MFCCs  
- Spectral centroid, bandwidth, flatness, rolloff  
- Zero-Crossing Rate  
- RMS Energy

---

## 🔁 BiLSTM Model

- Input shape: (45, 1)  
- Architecture: BiLSTM → Fully Connected  
- Hyperparameter tuning on:
  - `hidden_size`: [64, 128, 256, 512]
  - `num_layers`: [1, 2, 4, 6]

**Best Configuration:**
- `hidden_size = 128`, `num_layers = 1`
- **Validation Accuracy:** 48.24%  
- **Test Accuracy:** 47.37%  
- **F1 Score:** 46.68%

---

## 🔀 Transformer Model

A custom Transformer model was implemented with:

- Multi-head self-attention  
- Positional encoding  
- Feedforward layers  
- Layer normalization + residual connections

**Best Configuration (no reshaping):**
- `d_model = 512`, `num_heads = 4`, `d_ff = 128`, `num_layers = 6`
- **Validation Accuracy:** 58.21%  
- **Test Accuracy:** 55.66%  
- **F1 Score:** 54.59%

---

## ❗ Most Confusing Classes (Transformer)

- Air Conditioner ↔ Children Playing  
- Street Music ↔ Children Playing  
- Gun Shot ↔ Dog Bark  
- Children Playing ↔ Dog Bark  
- Air Conditioner ↔ Drilling

These confusions are due to overlapping acoustic patterns, background noise, and limits of feature representation.

---

## 📊 Evaluation Metrics

- Accuracy  
- Weighted F1 Score  
- Confusion Matrix Visualization

---
