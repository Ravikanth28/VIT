from flask import Flask, request, jsonify, logging
import cv2
import numpy as np
import os
import hashlib
from werkzeug.utils import secure_filename
import tempfile 

# --- Configuration ---
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# --- Flask App Initialization ---
app = Flask(_name) # CORRECTED: Changed _name to _name_

# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def is_heatmap(img):
    """
    Detects if an image qualifies as a heatmap by checking for the presence
    of at least two distinct color ranges typical of heatmaps.
    """
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
    """
    Estimates a CO₂ value based on the proportion of "hot" colors (red and yellow)
    in the image.
    """
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

    co2_estimate = hot_ratio * 100 
    return round(co2_estimate, 2)

# --- Main API Endpoint ---

@app.route("/check-heatmap", methods=["POST"])
def check_heatmap():
    """
    Main endpoint to receive an image, validate it, and return a CO₂ estimate.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    temp_path = None
    try:
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"result": "error", "message": "Could not read image file"}), 400

        if not is_heatmap(img):
            return jsonify({"result": "no"}), 200

        co2_value = estimate_co2_emissions(img)
        return jsonify({"result": "yes", "estimated_CO2_Mt": co2_value}), 200

    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred on the server."}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/", methods=["GET"])
def home():
    """A simple welcome endpoint to confirm the API is running."""
    return jsonify({"message": "Heatmap detection & CO₂ estimation API is running."})


if _name_ == "_main_":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
