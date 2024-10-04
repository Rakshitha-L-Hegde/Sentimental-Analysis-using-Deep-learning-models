import os
import librosa
import numpy as np

# Function to extract features from audio files
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

# Function to load and preprocess the dataset
def load_dataset(dataset_path):
    features = []
    labels = []

    for file in os.listdir(dataset_path):
        if file.endswith('.wav'):
            file_path = os.path.join(dataset_path, file)
            emotion = file.split('_')[2]  # Extract emotion label from filename
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(emotion)

    return np.array(features), np.array(labels)
