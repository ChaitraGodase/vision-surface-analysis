import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import joblib

from PIL import Image

from src.preprocessing.preprocess import preprocess_image
from src.features.extract import extract_features

st.set_page_config(
    page_title="AI Surface Intelligence",
    layout="wide"
)

# =========================================
# LOAD CSS
# =========================================

with open("assets/styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =========================================
# TITLE
# =========================================

st.markdown(
    '''
    <div class="hero">
        <h1>🚀 AI Surface Intelligence</h1>
        <p>Spatial Vision Analytics Dashboard</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# =========================================
# FILE UPLOAD
# =========================================

uploaded = st.file_uploader(
    "Upload Surface Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:

    image = Image.open(uploaded)

    img_np = np.array(image)

    st.image(
        image,
        use_container_width=True
    )

    # =====================================
    # PREPROCESS
    # =====================================

    processed = preprocess_image(img_np)

    # =====================================
    # FEATURES
    # =====================================

    features = extract_features(processed)

    # =====================================
    # LOAD MODEL
    # =====================================

    model = joblib.load("models/model.pkl")

    selector = joblib.load("models/selector.pkl")

    selected = selector.transform([features])

    probs = model.predict_proba(selected)[0]

    prediction = model.classes_[np.argmax(probs)]

    confidence = float(np.max(probs))

    # =====================================
    # DEPTH SIMULATION
    # =====================================

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    depth = cv2.Laplacian(gray, cv2.CV_64F)

    depth = np.absolute(depth)

    depth = depth / depth.max()

    # =====================================
    # DASHBOARD
    # =====================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Prediction",
            prediction
        )

    with c2:

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

    with c3:

        st.metric(
            "Depth Severity",
            "Moderate"
        )

    # =====================================
    # DEPTH MAP
    # =====================================

    st.subheader("Depth Estimation")

    st.image(
        depth,
        use_container_width=True
    )

    # =====================================
    # FEATURE IMPORTANCE
    # =====================================

    if hasattr(model, "feature_importances_"):

        fig = go.Figure()

        fig.add_trace(

            go.Bar(
                y=model.feature_importances_[:10]
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================
    # FEATURE ANALYTICS
    # =====================================

    st.subheader("Feature Analytics")

    for i, value in enumerate(features[:10]):

        st.progress(float(abs(value)))