from flask import Flask, render_template, request, send_from_directory, redirect, url_for, session
import json
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "admin@present.p"   # CHANGE THIS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIOLATION_DIR = os.path.join(BASE_DIR, "violations_images")
JSON_PATH = os.path.join(BASE_DIR, "violations.json")


@app.route("/violations/<filename>")
def send_image(filename):
    return send_from_directory(VIOLATION_DIR, filename)


def repair_json_image_paths(data):
    image_files = set(os.listdir(VIOLATION_DIR))

    for v in data:
        img = v.get("image_path", "").strip()

        if img and img in image_files:
            continue

        timestamp = v.get("timestamp", "").replace(":", "-").replace(" ", "_")

        for file in image_files:
            if timestamp in file:
                v["image_path"] = file
                break

        if not v.get("image_path"):
            v["image_path"] = "no_image.png"

    return data


# -----------------------------------
# LOGIN PAGE
# -----------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if session.get("logged_in"):
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email == "admin@present.p" and password == "admin":
            session["logged_in"] = True
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")


# -----------------------------------
# LOGOUT
# -----------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -----------------------------------
# DASHBOARD (PROTECTED)
# -----------------------------------
@app.route("/")
def index():

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    search = request.args.get("search", "").strip().lower()
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    sort = request.args.get("sort", "timestamp_desc")
    page = int(request.args.get("page", 1))
    per_page = 8

    # Load JSON
    if os.path.exists(JSON_PATH) and os.path.getsize(JSON_PATH) > 0:
        with open(JSON_PATH, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []
    else:
        data = []

    data = repair_json_image_paths(data)

    # Search filter
    if search:
        data = [
            v for v in data
            if search in v.get("license_plate", "").lower()
            or search in v.get("violation", "").lower()
        ]

    # ---- DATE PARSER (DD/MM/YYYY + YYYY-MM-DD) ----
    def parse_date(d):
        if not d:
            return None

        if "/" in d:  # dd/mm/yyyy
            try:
                return datetime.strptime(d, "%d/%m/%Y")
            except:
                pass

        if "-" in d:  # yyyy-mm-dd
            try:
                return datetime.strptime(d, "%Y-%m-%d")
            except:
                pass

        return None

    start_dt = parse_date(start)
    end_dt = parse_date(end)

    # Filter
    if start_dt:
        data = [
            v for v in data
            if datetime.strptime(v["timestamp"], "%Y-%m-%d %H:%M:%S") >= start_dt
        ]

    if end_dt:
        data = [
            v for v in data
            if datetime.strptime(v["timestamp"], "%Y-%m-%d %H:%M:%S") <= end_dt
        ]

    # Sorting
    if sort == "timestamp_asc":
        data.sort(key=lambda x: x["timestamp"])
    elif sort == "plate_asc":
        data.sort(key=lambda x: x["license_plate"])
    elif sort == "plate_desc":
        data.sort(key=lambda x: x["license_plate"], reverse=True)
    else:
        data.sort(key=lambda x: x["timestamp"], reverse=True)

    # Pagination
    total = len(data)
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated = data[start_index:end_index]

    return render_template(
        "index.html",
        violations=paginated,
        page=page,
        total=total,
        has_prev=page > 1,
        has_next=end_index < total,
        search=search,
        start_date=start,
        end_date=end,
        sort=sort
    )


if __name__ == "__main__":
    app.run(debug=True)
