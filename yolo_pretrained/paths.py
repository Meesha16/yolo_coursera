from pathlib import Path

base_dir = Path(__file__).resolve().parent
image_path = base_dir / "images" / "testing" / "scene2.jpg"
config_path = base_dir / "model" / "yolov3.config"
weights_path = base_dir / "model" / "yolov3.weights"