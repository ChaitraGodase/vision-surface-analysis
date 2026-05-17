import os
import sys
import numpy as np

# =====================================================
# ROOT PATH FIX
# =====================================================

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.append(ROOT_DIR)

# =====================================================
# IMPORTS
# =====================================================

from src.preprocessing.preprocess import preprocess_image

from src.features.extract import extract_features

from src.models.depth_estimation import (
    load_model,
    generate_depth_map
)

# =====================================================
# OUTPUT FOLDERS
# =====================================================

os.makedirs("outputs/depth_maps", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

# =====================================================
# LOAD DPT MODEL
# =====================================================

processor, depth_model = load_model()

# =====================================================
# MAIN ANALYSIS FUNCTION
# =====================================================

def analyze_image(image_path):

    try:

        # =============================================
        # PREPROCESS
        # =============================================

        img = preprocess_image(image_path)

        if img is None:

            return {
                "error": "Preprocessing failed"
            }

        # =============================================
        # FEATURE EXTRACTION
        # =============================================

        features = extract_features(img)

        # =============================================
        # DEPTH MAP GENERATION
        # =============================================

        depth_path = "outputs/depth_maps/depth.png"

        depth_map = generate_depth_map(
            image_path,
            processor,
            depth_model,
            depth_path
        )

        # =============================================
        # DEPTH ANALYTICS
        # =============================================

        avg_depth = float(np.mean(depth_map))

        max_depth = float(np.max(depth_map))

        min_depth = float(np.min(depth_map))

        # =============================================
        # SEVERITY CLASSIFICATION
        # =============================================

        if avg_depth > 0.6:

            prediction = "Deep Surface Damage"

        elif avg_depth > 0.3:

            prediction = "Moderate Surface Wear"

        else:

            prediction = "Shallow Surface Damage"

        confidence = round(avg_depth * 100, 2)

        # =============================================
        # RETURN RESULTS
        # =============================================

        return {

            "prediction": prediction,

            "confidence": confidence,

            "avg_depth": round(avg_depth, 3),

            "max_depth": round(max_depth, 3),

            "min_depth": round(min_depth, 3),

            "depth_path": depth_path,

            "features": features.tolist()

        }

    except Exception as e:

        return {
            "error": str(e)
        }