from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
import os
import hashlib
from werkzeug.utils import secure_filename

# Config
PICKLE_FILE = "heatmap_images.pkl"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load saved data or initialize
if os.path.exists(PICKLE_FILE):
    with open(PICKLE_FILE, "rb") as f:
        saved_data = pickle.load(f)
else:
    saved_data = {"hashes": [], "images": []}

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_heatmap(img):
    """Detect heatmap based on presence of heatmap colors."""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "red1": ((0, 100, 100), (10, 255, 255)),
        "red2": ((160, 100, 100), (179, 255, 255)),
        "yellow": ((20, 100, 100), (30, 255, 255)),
        "green": ((40, 50, 50), (80, 255, 255)),
        "blue": ((90, 50, 50), (130, 255, 255)),
    }
    found_colors = 0
    for _, (low, high) in color_ranges.items():
        mask = cv2.inRange(img_hsv, np.array(low), np.array(high))
        if cv2.countNonZero(mask) > 500:
            found_colors += 1
    return found_colors >= 2


def estimate_co2_emissions(img):
    """Estimate CO₂ emissions based on red/yellow hot zones."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    hot_mask = red_mask | yellow_mask

    hot_pixels = cv2.countNonZero(hot_mask)
    total_pixels = img.shape[0] * img.shape[1]
    hot_ratio = hot_pixels / total_pixels

    co2_estimate = hot_ratio * 100  # scaling factor for demo
    return round(co2_estimate, 2)


def get_image_hash(img):
    img_bytes = cv2.imencode(".png", img)[1].tobytes()
    return hashlib.sha256(img_bytes).hexdigest()


@app.route("/check-heatmap", methods=["POST"])
def check_heatmap():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        img = cv2.imread(file_path)
        if img is None:
            return jsonify({"result": "error", "message": "Could not read image"}), 400

        # Main detection
        if not is_heatmap(img):
            return jsonify({"result": "no"}), 200

        img_hash = get_image_hash(img)
        if img_hash in saved_data["hashes"]:
            return jsonify({"result": "duplicate"}), 200

        saved_data["hashes"].append(img_hash)
        saved_data["images"].append(img)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(saved_data, f)

        # If YES, calculate emissions
        co2_value = estimate_co2_emissions(img)
        return jsonify({"result": "yes", "estimated_CO2_Mt": co2_value}), 200

    return jsonify({"error": "Invalid file type"}), 400


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Heatmap detection & CO₂ estimation API running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
