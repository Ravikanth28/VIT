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
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define common heatmap colors in HSV
    color_ranges = {
        "red1": ((0, 50, 50), (10, 255, 255)),
        "red2": ((170, 50, 50), (180, 255, 255)),
        "yellow": ((20, 50, 50), (35, 255, 255)),
        "green": ((40, 50, 50), (85, 255, 255)),
        "blue": ((90, 50, 50), (130, 255, 255))
    }

    found_colors = 0
    for _, (low, high) in color_ranges.items():
        mask = cv2.inRange(img_hsv, np.array(low), np.array(high))
        ratio = cv2.countNonZero(mask) / img.size
        if ratio > 0.01:
            found_colors += 1

    if found_colors < 4:
        return False

    avg_saturation = np.mean(img_hsv[:, :, 1])
    if avg_saturation < 100:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var > 150:
        return False

    return True


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

        if not is_heatmap(img):
            return jsonify({"result": "no"}), 200

        img_hash = get_image_hash(img)
        if img_hash in saved_data["hashes"]:
            return jsonify({"result": "duplicate"}), 200

        saved_data["hashes"].append(img_hash)
        saved_data["images"].append(img)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(saved_data, f)

        return jsonify({"result": "yes"}), 200

    return jsonify({"error": "Invalid file type"}), 400


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Heatmap detection API running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
