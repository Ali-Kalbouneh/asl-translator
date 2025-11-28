import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Read from csv file

csv_path = "../data/letters.csv"
x = [] # feature vectors (list of lists)
y = [] # labels (list of letters)

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row["label"]

        features = [float(row[f"f{i}"]) for i in range(63)] # read f0..f62 and convert to floats

        x.append(features)
        y.append(label)

x = np.array(x, dtype=np.float32)
y = np.array(y)

print(f"Loaded {len(y)} samples.")   


# Split into 80% training, 20% testing

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 42
)

print(f"Training Samples: {len(y_train)}, Test Samples: {len(y_test)}")

# Training the Classifier

clsf = RandomForestClassifier(
    n_estimators = 200,
    random_state = 42
)

clsf.fit(x_train, y_train)

# Accuracy evaluation

y_pred = clsf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f}\n")

print("Classification report:")
print(classification_report(y_test, y_pred))

# Saving the Model

os.makedirs("../models", exist_ok=True)

model_path = "../models/asl_letters_rf.pkl"
joblib.dump(clsf, model_path)
print(f"\nModel saved to: {model_path}")