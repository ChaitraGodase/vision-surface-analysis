import os

# =========================================
# MATPLOTLIB MEMORY FIX
# =========================================

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")

import torch
import matplotlib.pyplot as plt

from PIL import Image

from transformers import (
    DPTImageProcessor,
    DPTForDepthEstimation
)

# =========================================
# MODEL NAME
# =========================================

MODEL_NAME = "Intel/dpt-hybrid-midas"

# =========================================
# GLOBAL CACHE
# =========================================

processor = None
model = None


# =========================================
# LOAD MODEL ONLY ONCE
# =========================================

def load_model():

    global processor
    global model

    if processor is None or model is None:

        processor = DPTImageProcessor.from_pretrained(
            MODEL_NAME
        )

        model = DPTForDepthEstimation.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True
        )

        model.eval()

    return processor, model


# =========================================
# GENERATE DEPTH MAP
# =========================================

def generate_depth_map(
    image_path,
    processor,
    model,
    save_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(

        predicted_depth.unsqueeze(1),

        size=image.size[::-1],

        mode="bicubic",

        align_corners=False

    ).squeeze()

    depth_map = prediction.cpu().numpy()

    # =====================================
    # NORMALIZE
    # =====================================

    depth_map = (

        depth_map - depth_map.min()

    ) / (

        depth_map.max() - depth_map.min()
    )

    # =====================================
    # SAVE DEPTH VISUALIZATION
    # =====================================

    plt.figure(figsize=(8, 8))

    plt.imshow(
        depth_map,
        cmap="inferno"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    return depth_map