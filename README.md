# Skin Disease Classification and Deployment

This project is a deep learning-based system for classifying **7 types of skin diseases** using **Convolutional Neural Networks (CNN)** and the **HAM10000** dataset. 
The model is trained in a Jupyter Notebook and deployed via a Flask web application.

---

## 🚀 Project Structure


---

## 📚 Dataset

- **Dataset Used:** HAM10000 (from Kaggle)
- **Classes:**
  - Actinic keratoses
  - Basal cell carcinoma
  - Benign keratosis-like lesions
  - Dermatofibroma
  - Melanocytic nevi
  - Vascular lesions
  - Melanoma

---

## 📒 Model Training

- Framework: TensorFlow/Keras
- Input Shape: 28x28x3 (resized)
- Metrics: Accuracy
- Output: Trained model saved as `skin_disease_model.h5`

### Run Jupyter Notebook:
```bash
cd sp
jupyter notebook
cd "sp deployment"
python app.py

python -m venv exam
source exam/bin/activate  # or exam\Scripts\activate on Windows
pip install -r requirements.txt

requirement
flask
tensorflow
numpy
opencv-python

