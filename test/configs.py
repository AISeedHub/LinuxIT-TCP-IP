# test/configs.py
import os, yaml

YAML_PATH = os.path.join(r"server_config.yaml")
with open(YAML_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
m = cfg["model"]

BASE_DIR = os.path.dirname(YAML_PATH)

# classifier
MODEL_PATH = os.path.dirname(m["model_path"]) 
MODEL_DEFAULT = os.path.basename(m["model_path"])

# preprocessor
PREPROCESSOR_PATH = m.get("preprocessor_path")
if PREPROCESSOR_PATH and not os.path.isabs(PREPROCESSOR_PATH):
    PREPROCESSOR_PATH = os.path.normpath(os.path.join(BASE_DIR, PREPROCESSOR_PATH))

# images (fix here)
IMG_PATH = m["img_path"]
if not os.path.isabs(IMG_PATH):
    IMG_PATH = os.path.normpath(os.path.join(BASE_DIR, IMG_PATH))

if os.path.isdir(IMG_PATH):
    LIST_IMAGES_TEST = [f for f in os.listdir(IMG_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
else:
    LIST_IMAGES_TEST = []

# Threshold and classes
CONFIDENCE_THRESHOLD = m.get("confidence_threshold", 0.9)
CLASSES = tuple(m.get("classes", []))