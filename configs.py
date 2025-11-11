# LinuxIT-TCP-IP/configs.py
import os
import yaml

BASE_DIR = os.path.dirname(__file__)
YAML_PATH = os.path.join(BASE_DIR, "config", "server_config.yaml")

with open(YAML_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

m = cfg["model"]

# Classifier weights folder/file
MODEL_PATH = os.path.dirname(m["model_path"])
MODEL_DEFAULT = os.path.basename(m["model_path"])
PREPROCESSOR_PATH = m.get("preprocessor_path")
NUM_CLASSES = m.get("num_classes", 8)

# Images
IMG_PATH = os.path.normpath(os.path.join(BASE_DIR, m["img_path"]))
if os.path.isdir(IMG_PATH):
    LIST_IMAGES_TEST = [f for f in os.listdir(IMG_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
else:
    LIST_IMAGES_TEST = []

# Threshold and classes
CONFIDENCE_THRESHOLD = m.get("confidence_threshold", 0.9)
CLASSES = tuple(m.get("classes", []))