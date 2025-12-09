from flask import Flask, render_template, send_from_directory
import json
import os

app = Flask(__name__)

# Path to the folder containing violation images
VIOLATION_IMAGES_FOLDER = "src/violations_images"  # Relative path

# Path to the JSON file that stores violation information
VIOLATIONS_JSON_FILE = "src/violations.json"  # Relative path

@app.route('/')
def index():
    """Main route to display violation data."""
    # Load violation data from the JSON file
    if os.path.exists(VIOLATIONS_JSON_FILE):
        with open(VIOLATIONS_JSON_FILE, "r") as f:
            violations = json.load(f)
    else:
        violations = []

    # Render the data in an HTML template (we'll create this template next)
    return render_template('index.html', violations=violations)

@app.route('/violations_images/<filename>')
def send_image(filename):
    """Route to serve the captured violation images."""
    # Use the send_from_directory function to serve images from the violations_images folder
    return send_from_directory(VIOLATION_IMAGES_FOLDER, filename)

if __name__ == "__main__":
    # Start the Flask web server
    app.run(debug=True)
