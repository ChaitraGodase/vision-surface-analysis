import os

# =========================================
# MATPLOTLIB MEMORY FIX
# =========================================

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import joblib
import numpy as np

from src.preprocessing.preprocess import preprocess_image
from src.features.extract import extract_features
from src.models.depth_estimation import (
    load_model,
    generate_depth_map
)

# =========================================
# GLOBAL CACHE
# =========================================

processor = None
depth_model = None

rf_model = None
selector_model = None


def load_depth_once():

    global processor
    global depth_model

    if processor is None or depth_model is None:

        processor, depth_model = load_model()

    return processor, depth_model


def load_classifier_once():

    global rf_model
    global selector_model

    if rf_model is None:

        rf_model = joblib.load("models/model.pkl")

    if selector_model is None:

        selector_model = joblib.load("models/selector.pkl")

    return rf_model, selector_model


def analyze_image(image_path):

    try:

        # =========================================
        # LOAD CLASSIFIER
        # =========================================

        model, selector = load_classifier_once()

        # =========================================
        # LOAD DEPTH MODEL ONLY WHEN NEEDED
        # =========================================

        processor, depth_model = load_depth_once()

        # =========================================
        # PREPROCESS
        # =========================================

        img = preprocess_image(image_path)

        if img is None:

            return {
                "error": "Preprocessing failed"
            }

        # =========================================
        # FEATURE EXTRACTION
        # =========================================

        features = extract_features(img)

        # =========================================
        # OUTPUT FOLDER
        # =========================================

        os.makedirs(
            "app/static/outputs",
            exist_ok=True
        )

        # =========================================
        # DEPTH ESTIMATION
        # =========================================

        depth_save = "app/static/outputs/depth.png"

        depth_display = "static/outputs/depth.png"

        depth_map = generate_depth_map(
            image_path,
            processor,
            depth_model,
            depth_save
        )

        # =========================================
        # DEPTH ANALYTICS
        # =========================================

        avg_depth = float(np.mean(depth_map))

        max_depth = float(np.max(depth_map))

        min_depth = float(np.min(depth_map))

        # =========================================
        # DEPTH LEVEL
        # =========================================

        if avg_depth > 0.6:

            depth_level = "Deep"

        elif avg_depth > 0.3:

            depth_level = "Moderate"

        else:

            depth_level = "Shallow"

        # =========================================
        # CLASSIFICATION
        # =========================================

        selected = selector.transform([features])

        probs = model.predict_proba(selected)[0]

        prediction = model.classes_[np.argmax(probs)]

        confidence = float(np.max(probs))

        # =========================================
        # FEATURE NAMES
        # =========================================

        feature_names = [

            "Texture-1",
            "Texture-2",
            "Texture-3",
            "Edge Density",
            "Gradient X",
            "Gradient Y",
            "Contrast",
            "Homogeneity",
            "Energy",
            "Entropy"
        ]

        features_named = [

            {
                "name": name,
                "value": float(val)
            }

            for name, val in zip(
                feature_names,
                features[:10]
            )
        ]

        # =========================================
        # FEATURE IMPORTANCE GRAPH
        # =========================================

        importance_save = (
            "app/static/outputs/importance.png"
        )

        importance_display = (
            "static/outputs/importance.png"
        )

        importances = model.feature_importances_

        plt.figure(figsize=(8, 4))

        plt.bar(
            range(len(importances[:10])),
            importances[:10]
        )

        plt.title("Feature Importance")

        plt.xlabel("Feature Index")

        plt.ylabel("Importance Score")

        plt.tight_layout()

        plt.savefig(importance_save)

        plt.close()

        # =========================================
        # FINAL OUTPUT
        # =========================================

        return {

            "prediction": str(prediction),

            "confidence": round(
                confidence * 100,
                2
            ),

            "depth_level": depth_level,

            "avg_depth": round(avg_depth, 3),

            "max_depth": round(max_depth, 3),

            "min_depth": round(min_depth, 3),

            "depth_path": depth_display,

            "importance_path": importance_display,

            "features_sample": features_named
        }

    except Exception as e:

        return {
            "error": str(e)
        }