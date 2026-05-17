from flask import Flask, render_template, request
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.predict import analyze_image

app = Flask(__name__)

UPLOAD_FOLDER = "app/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            result = analyze_image(filepath)
            print("RESULT:", result)  # 🔥 DEBUG

            image_path = "static/uploads/" + file.filename

    return render_template("index.html", result=result, image_path=image_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)