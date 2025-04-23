import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Contoh data dummy (harus diganti dengan dataset asli jika ada)
data = {
    'protein': [10, 20, 5, 15, 25],
    'karbohidrat': [30, 50, 20, 45, 60],
    'kalori': [200, 350, 120, 300, 400],
    'lemak': [5, 10, 2, 8, 15],
    'akg': [20, 40, 10, 35, 50],
    'label': ['rendah', 'tinggi', 'rendah', 'sedang', 'tinggi']
}

df = pd.DataFrame(data)

# Preprocessing
X = df[['protein', 'karbohidrat', 'kalori', 'lemak', 'akg']]
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Simpan model dan label encoder
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model trained and saved.")
