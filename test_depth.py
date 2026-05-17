import os

# ✅ Import functions
from src.models.depth_estimation import load_model, generate_depth_map

# ✅ Create output folder
os.makedirs("outputs", exist_ok=True)

# ✅ Load model once
processor, model = load_model()

# ✅ Input image
image_path = "data/raw/potholes/sample.jpg"   # change if needed

# ✅ Output path
save_path = "outputs/depth_map.png"

# ✅ Generate depth map
generate_depth_map(image_path, processor, model, save_path)

print("✅ Depth map generated successfully!")