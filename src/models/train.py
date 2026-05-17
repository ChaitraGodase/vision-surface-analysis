import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessing.preprocess import preprocess_image
from src.features.extract import extract_features
from src.features.select import select_features


def train_model(data_dir="data/raw/"):
    X, y = [], []

    print("📥 Loading dataset...")

    for label in os.listdir(data_dir):
        folder = os.path.join(data_dir, label)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            path = os.path.join(folder, file)

            img = preprocess_image(path)

            if img is None:
                continue

            feat = extract_features(img)

            X.append(feat)
            y.append(label)

    X = np.array(X)

    print(f"✅ Total samples: {len(X)}")

    X_new, selector = select_features(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n📊 Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(selector, "models/selector.pkl")

    print("✅ Model saved!")