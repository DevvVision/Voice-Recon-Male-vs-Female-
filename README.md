🎤 Voice Classification Using Machine Learning
This project focuses on classifying voice samples using machine learning. It leverages audio feature extraction techniques and a trained classification model to predict the characteristics of a given voice sample.

🚀 Features
Preprocessing & Feature Engineering

Uses pandas to load and preprocess a dataset of voice samples.
Extracts specific frequency-based features from audio signals using librosa.
Applies Standard Scaling to normalize the dataset.
Machine Learning Model

Uses scikit-learn for model training and evaluation.
Implements a Logistic Regression classifier for prediction.
Saves and loads the trained model using joblib.
Audio Feature Extraction

Extracts key features like:
Mean Frequency
Spectral Entropy
Spectral Flatness
Fundamental Frequency
Frequency Centroid
Interquartile Range (IQR)
Prediction on New Audio Files

Uses librosa to analyze an input audio file (.mp3).
Extracts features and applies the trained model for classification.
📌 Dependencies
numpy
pandas
scikit-learn
librosa
joblib
matplotlib
scipy
📂 Usage
Install dependencies:
bash
Copy
Edit
pip install numpy pandas scikit-learn librosa joblib matplotlib scipy
Train the model (if needed) and save it using joblib.dump().
Run the prediction script with a new audio file:
python
Copy
Edit
loaded_model = joblib.load('classification_model.pkl')
audio_path = "Assets/f1.mp3"
features = compute_audio_features(audio_path)
predictions = loaded_model.predict(sc.transform([features]))
print(f"Predictions: {predictions}")
📈 Results
The model predicts whether the input voice sample belongs to a male or female speaker (or other categories based on dataset labels).
