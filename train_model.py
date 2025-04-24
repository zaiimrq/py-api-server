import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('hasil_klasifikasi_akg_200.csv')

# Handle missing values in jenis_kelamin
df['jenis_kelamin'] = df['jenis_kelamin'].fillna(2.0)  # Using 2.0 for missing values

# Prepare features (X) and target (y)
X = df[['kategori_umur', 'jenis_kelamin', 'total_protein', 'total_lemak', 'total_karbohidrat', 'total_kalori']]
y = df['klasifikasi']

# Initialize and fit LabelEncoder for the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate and print accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Save the model and label encoder
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

print("Model and Label Encoder have been saved successfully!")
