# 🎤 Voice Classification Using Machine Learning  

This project focuses on **classifying voice samples** using machine learning. It extracts essential **audio features** using `librosa`, preprocesses the data, and applies **machine learning models** to classify voices. The trained model is saved using `joblib` and can analyze new audio files for predictions.  

## 🚀 Features  
- **Audio Feature Extraction** using `librosa`:  
  - **Mean Frequency**  
  - **Spectral Entropy**  
  - **Spectral Flatness**  
  - **Frequency Centroid**  
  - **Interquartile Range (IQR)**  
  - **Fundamental Frequency**  

- **Preprocessing & Data Handling**  
  - Loads and preprocesses a **voice dataset** using `pandas`.  
  - Applies **Standard Scaling** for normalization.  
  - Converts categorical labels to numerical values using `LabelEncoder`.  
  - Splits data into **training and testing sets**.  

- **Machine Learning Model**  
  - Implements a **Logistic Regression classifier** using `scikit-learn`.  
  - Trains and evaluates the model on extracted features.  
  - Saves and loads the trained model using `joblib`.  

- **Prediction on New Audio Files**  
  - Extracts features from an input audio file (`.mp3`).  
  - Transforms and scales the features.  
  - Uses the trained model to **classify the speaker’s voice**.  

## 📌 Dependencies  
To run this project, install the following dependencies:  
```bash
pip install numpy pandas scikit-learn librosa joblib matplotlib scipy

📂 Usage
1️⃣ Train the Model (Optional)
If you want to train the model yourself, run the training script:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
ds = pd.read_csv("voice.csv")
X = ds.iloc[:, [0,1,3,5,8,9,11,12]].values
y = ds.iloc[:, -1].values

# Encode labels
lb = LabelEncoder()
y = lb.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Standard Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train classifier
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Save model
joblib.dump(classifier, 'classification_model.pkl')
print("Model saved successfully!")


2️⃣ Predict on New Audio File
Use a pre-trained model to predict the classification of an audio file:

import joblib
import numpy as np
import librosa

# Load pre-trained model
loaded_model = joblib.load('classification_model.pkl')

# Function to extract features from an audio file
def compute_audio_features(audio_file, sr=22050):
    y, sr = librosa.load(audio_file, sr=sr)
    S = np.abs(librosa.stft(y)) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0] * 2 - 1)
    power = np.sum(S, axis=1)

    mean_freq = np.sum(frequencies * power) / np.sum(power)
    std_freq = np.sqrt(np.sum((frequencies - mean_freq) ** 2 * power) / np.sum(power))
    Q25 = frequencies[np.searchsorted(np.cumsum(power) / np.sum(power), 0.25)]
    Q75 = frequencies[np.searchsorted(np.cumsum(power) / np.sum(power), 0.75)]
    IQR = Q75 - Q25
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(S=S))
    freq_centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))

    features = [
        mean_freq / 1000 if not np.isnan(mean_freq) else 0,
        std_freq / 1000 if not np.isnan(std_freq) else 0,
        Q25 / 1000 if not np.isnan(Q25) else 0,
        IQR / 1000 if not np.isnan(IQR) else 0,
        spectral_flatness if not np.isnan(spectral_flatness) else 0,
        freq_centroid / 1000 if not np.isnan(freq_centroid) else 0
    ]
    return features

# Extract features from an audio file and predict
audio_path = "Assets/f1.mp3"
features = compute_audio_features(audio_path)
prediction = loaded_model.predict([features])

print(f"Predicted Class: {prediction}")

📈 Results
The trained model predicts whether a given voice sample belongs to a male or female speaker (or other categories based on dataset labels).
Extracted audio features contribute to improving classification accuracy.
The approach can be extended to classify different types of voices or sound patterns.
📜 Future Enhancements
🔹 Train the model using deep learning (e.g., CNNs, RNNs).
🔹 Use MFCC features for improved accuracy.
🔹 Implement a real-time voice classification system.

🔗 Author: Harshit Srivastava
🔗 GitHub: DevvVision


This README is well-structured and provides details about **project features, dependencies, training, usage, and future improvements**. Let me know if you need modifications! 🚀
