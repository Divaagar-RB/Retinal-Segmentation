
import os
from PIL import Image
import subprocess
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "upload/images"
RESULT_FOLDER = "Results/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Path to your model checkpoint
CHECKPOINT = "saved_models/Epo27_Dice0.2873.pth"
SAVE_FOLDER = RESULT_FOLDER

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")  # upload form

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    # Save uploaded file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Convert uploaded image to PNG for display
    base, ext = os.path.splitext(file.filename)
    png_path = os.path.join(app.config["UPLOAD_FOLDER"], base + ".png")
    with Image.open(file_path) as img:
        img.save(png_path)

    # Run test.py and capture the last saved image
    result_path = subprocess.check_output([
        "python",
        "test.py",
        "--checkpoint", CHECKPOINT,
        "--save_path", SAVE_FOLDER,
        "--input_image", file_path
    ], universal_newlines=True)

    # test.py should print the last result path
    result_file = os.path.basename(result_path.strip())
    
    # Pass uploaded image and result image to template
    return render_template("result.html", 
                           uploaded_image=base + ".png",
                           result_image=result_file)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(os.path.join(app.config["RESULT_FOLDER"], "Eval"), filename)


