import streamlit as st
from PIL import Image
import numpy as np

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI Surface Intelligence",
    page_icon="🚀",
    layout="wide"
)

# =========================================
# LOAD CSS
# =========================================

def load_css():

    with open("app/static/styles.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# =========================================
# HEADER
# =========================================

st.markdown(
    """
    <h1 class="main-title">
        🚀 AI Surface Intelligence
    </h1>

    <p class="sub-title">
        Interactive Depth + Vision Analysis
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================
# FILE UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "Choose Surface Image",
    type=["png", "jpg", "jpeg"]
)

# =========================================
# MAIN ANALYSIS
# =========================================

if uploaded_file:

    image = Image.open(uploaded_file)

    image_np = np.array(image)

    st.markdown(
        """
        <div class="section-title">
            📷 Input Image
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        image,
        use_container_width=True
    )

    # =========================================
    # MOCK AI RESULTS
    # =========================================

    prediction = "Pothole Detected"

    confidence = "96.4%"

    depth_level = "Moderate"

    avg_depth = "0.47"

    # =========================================
    # METRICS
    # =========================================

    st.markdown(
        """
        <div class="section-title">
            📊 Analysis Metrics
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Prediction",
            prediction
        )

    with c2:
        st.metric(
            "Confidence",
            confidence
        )

    with c3:
        st.metric(
            "Depth Level",
            depth_level
        )

    with c4:
        st.metric(
            "Avg Depth",
            avg_depth
        )

    # =========================================
    # PERFORMANCE
    # =========================================

    st.markdown(
        """
        <div class="section-title">
            ⚡ Performance Monitor
        </div>
        """,
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Inference", "0.42s")
    p2.metric("GPU Usage", "62%")
    p3.metric("Latency", "120ms")
    p4.metric("FPS", "24")

    # =========================================
    # HEATMAP
    # =========================================

    st.markdown(
        """
        <div class="section-title">
            🌊 Depth Heatmap
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        image,
        use_container_width=True
    )

    # =========================================
    # SPATIAL OVERLAY
    # =========================================

    st.markdown(
        """
        <div class="section-title">
            🚨 Spatial Severity Overlay
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("🟢 Safe Zones")

    st.warning("🟡 Moderate Regions")

    st.error("🔴 Critical Surface Damage")

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.title("⚙️ AI Controls")

    st.slider(
        "Depth Threshold",
        0.0,
        1.0,
        0.5
    )

    st.selectbox(
        "Model",
        [
            "DPT Small",
            "MiDaS",
            "Depth Anything"
        ]
    )

    st.checkbox(
        "Enable Heatmap",
        value=True
    )

    st.checkbox(
        "Enable GPU Monitoring",
        value=True
    )

    st.checkbox(
        "Enable Spatial Overlay",
        value=True
    )

    st.button("🚀 Analyze")