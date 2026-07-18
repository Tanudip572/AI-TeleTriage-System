# 🩺 AI Tele-Triage System

A **Multimodal Deep Learning-based Skin Disease Detection System** that combines **dermoscopic images** and **clinical text** to predict skin diseases and provide preliminary recommendations.

🔗 **Live Demo:** https://ai-teletriage-system-tel3rokodvk5jrychg7yjy.streamlit.app/

---

# 📌 Project Overview

Skin diseases are among the most common medical conditions worldwide. Early detection can significantly improve treatment outcomes.

This project proposes an **AI-assisted Tele-Triage System** that integrates:

- 📷 Dermoscopic Image Analysis
- 📝 Clinical Text Analysis
- 🧠 Multimodal Fusion Learning

to classify skin lesions into one of **7 skin disease categories**.

---

# ✨ Features

- Upload dermoscopic skin lesion images
- Enter patient clinical information
- Automatic clinical note generation
- DenseNet121 image feature extraction
- BiLSTM clinical text encoding
- Fusion Neural Network prediction
- Top-3 disease predictions
- Confidence score
- Disease recommendations
- Interactive Streamlit interface

---

# 🏗 Project Architecture

```
                  Dermoscopic Image
                         │
                  DenseNet121 CNN
                         │
                  Image Embeddings
                         │
                         │
Clinical Information ─► Tokenizer
                         │
                      BiLSTM
                         │
                  Text Embeddings
                         │
              Multimodal Fusion Network
                         │
                  Disease Prediction
                         │
              Recommendation Generator
```

---

# 📂 Project Structure

```
AI-TeleTriage-System
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── models/
│   └── fusion_final.keras
│
├── saved_objects/
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── image_size.pkl
│   ├── max_len.pkl
│   ├── recommendations.pkl
│   └── fusion_class_names.pkl
│
├── utils/
│   ├── loader.py
│   ├── predictor.py
│   ├── preprocessing.py
│   ├── recommendation.py
│   └── report.py
│
├── reports/
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_Image_Preprocessing.ipynb
│   ├── 03_Custom_CNN_Model.ipynb
│   ├── 04_Transfer_Learning_DenseNet121.ipynb
│   ├── 05_BiLSTM_Clinical_Text_Encoder.ipynb
│   ├── 06_Fusion_Model.ipynb
│   └── 07_AI_TeleTriage_System.ipynb
```

---

# 🧠 Deep Learning Models

## Image Branch

- DenseNet121
- Image Size: 224×224
- Transfer Learning
- Global Average Pooling
- Dense Embedding Layer

---

## Clinical Text Branch

- Text Cleaning
- Tokenization
- Padding
- Embedding Layer
- BiLSTM

---

## Fusion Network

The learned image embeddings and text embeddings are concatenated and passed through fully connected layers for final classification.

```
Image Features
        │
        │
Text Features
        │
        ▼
Concatenate
        │
Dense
        │
Dropout
        │
Dense
        │
Softmax (7 Classes)
```

---

# 📊 Dataset

**HAM10000 (Human Against Machine with 10000 Training Images)**

Dataset contains **10,015 dermoscopic images** from seven diagnostic categories.

Classes:

| Code | Disease |
|------|----------|
| akiec | Actinic Keratoses / Bowen's Disease |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis-like Lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevus |
| vasc | Vascular Lesions |

---

# ⚙ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Pillow

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Tanudip572/AI-TeleTriage-System.git

cd AI-TeleTriage-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

# 🖥 Application Workflow

1. Upload a dermoscopic image
2. Enter patient information
3. Generate clinical note
4. Extract image features
5. Extract text features
6. Perform multimodal fusion
7. Predict disease
8. Display:
   - Prediction
   - Confidence
   - Top-3 diseases
   - Clinical recommendations

---

# 📈 Output

The application provides

- Predicted Disease
- Prediction Confidence
- Top-3 Predictions
- Probability Distribution
- Clinical Recommendation

---

# 🎯 Future Improvements

- PDF Report Generation
- Grad-CAM Heatmap Visualization
- Explainable AI (XAI)
- Multi-language Support
- Electronic Health Record (EHR) Integration
- Mobile Application
- Doctor Dashboard

---

# ⚠ Disclaimer

This AI system is intended **only for educational and preliminary screening purposes**.

It **must not** replace professional medical diagnosis or treatment.

Always consult a qualified dermatologist for medical advice.

---

# 👨‍💻 Author

**Tanudip Ghosh**

Master's Student

Artificial Intelligence | Machine Learning | Deep Learning

GitHub: https://github.com/Tanudip572

---

# ⭐ If you found this project useful...

Please consider giving it a ⭐ on GitHub!
