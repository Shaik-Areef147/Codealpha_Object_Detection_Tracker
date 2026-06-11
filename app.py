from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    ext = os.path.splitext(file.filename)[1].lower()

    # IMAGE DETECTION
    if ext in IMAGE_EXTENSIONS:

        results = model(filepath)

        output_path = os.path.join(
            RESULT_FOLDER,
            "result_image.jpg"
        )

        annotated = results[0].plot()

        cv2.imwrite(output_path, annotated)

        return f"""
        <html>
        <body style="text-align:center;font-family:Arial;">
        <h2>✅ Object Detection Completed</h2>
        <img src="/static/results/result_image.jpg" width="700">
        <br><br>
        <a href="/">Go Back</a>
        </body>
        </html>
        """

    # VIDEO DETECTION
    elif ext in VIDEO_EXTENSIONS:

        cap = cv2.VideoCapture(filepath)

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Process every 10th frame only
            if frame_count % 10 != 0:
                continue

            frame = cv2.resize(frame, (640, 480))

            results = model(frame)

        cap.release()

        return """
        <html>
        <body style="text-align:center;font-family:Arial;">
        <h2>✅ Video Processed Successfully</h2>
        <p>Use short videos (5-10 sec) for Render deployment.</p>
        <a href="/">Go Back</a>
        </body>
        </html>
        """

    else:
        return "Unsupported File Format"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
