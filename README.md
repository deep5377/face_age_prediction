# Face Age Prediction with Machine Learning & Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Model: CNN](https://img.shields.io/badge/model-CNN-red.svg)]()
[![Status: Completed](https://img.shields.io/badge/status-completed-brightgreen.svg)]()


##  Project Objective

The aim is to develop a robust age classification system that can:
- Analyze and preprocess facial image datasets
- Train ML models using extracted features
- Build and evaluate CNNs for image classification
- Deploy live webcam age prediction using OpenCV



## Dataset

The dataset includes labeled face images grouped into the following age categories:
- 0–2
- 3–9
- 10–19
- 20–29
- 30–39
- 40–49
- 50–69
- 70+

📦 Dataset file: `code/ZIPPED_DATASETS/combined_faces.zip`

---

## 🧪 Project Structure

 face_age_prediction-main/

├── 1_EDA_dataset_prep.ipynb # Exploratory Data Analysis

├── 2_dataset_prep_ML_feature_extraction.ipynb # ML feature extraction

├── 3_ML_classification_modelling.ipynb # Traditional ML models

├── 4_5_training_data_augmentation.ipynb # Data augmentation

├── 4_deep_learning_CNN_modelling.ipynb # CNN model design

├── 5_deep_learning_final_CNN_model.ipynb # Final training & evaluation

├── final_age_detection_live.ipynb # Live age prediction via webcam

├── age_detect_cnn_model.h5 # Saved CNN model

└── datasets_unzipping.ipynb # Dataset unzip script


---


## Graphs
### 📊 Confusion Matrix
![Confusion Matrix]
![confusion_matrix_age_prediction](https://github.com/user-attachments/assets/27eaaca0-95b8-455a-afe5-e0593c62e783)


### 📈 Accuracy Over Epochs
![Accuracy Plot]
![accuracy_plot](https://github.com/user-attachments/assets/1b1cd600-fcc4-457a-aade-edb247e6530c)

### 📉 Loss Over Epochs
![Loss Plot]
![loss_plot](https://github.com/user-attachments/assets/b5b6b51b-abfa-4171-9a6d-6a8675b5bb0a)

## ⚙️ Installation & Setup


### 📦 Requirements
```bash
pip install -r requirements.txt
```


---

## ▶️ Running the Pipeline

1. **Extract the Dataset**
   - Run: `datasets_unzipping.ipynb`

2. **Data Exploration & Preparation**
   - Run: `1_EDA_dataset_prep.ipynb` and `2_dataset_prep_ML_feature_extraction.ipynb`

3. **Train Models**
   - Traditional ML: `3_ML_classification_modelling.ipynb`
   - Deep Learning (CNN): `4_deep_learning_CNN_modelling.ipynb`
   - Final Model: `5_deep_learning_final_CNN_model.ipynb`

4. **Live Webcam Age Detection**
   - Run: `final_age_detection_live.ipynb`

---

## 📊 Results & Evaluation

### ✅ Final Model Accuracy
- **Training Accuracy:** 92%
- **Validation Accuracy:** 85%
- **Training Loss:** 0.3
- **Validation Loss:** 0.5


---

## 🛠️ Tech Stack

- **Languages:** Python
- **Libraries:** TensorFlow, Keras, OpenCV, Scikit-learn, NumPy, Matplotlib
- **Tools:** Jupyter Notebook, Google Colab
- **Model:** CNN with image classification layers
- **Deployment:** Real-time webcam application via OpenCV

---


## 👤 Author

**Deep Patel**  


