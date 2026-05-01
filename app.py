from flask import Flask, render_template, request, send_from_directory, redirect, session, Response
import os
import cv2
from ultralytics import YOLO
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.secret_key = "mega_secret_key"

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# 🔥 LAST RESULT ONLY
last_persons = 0
last_alert = "✅ Safe"

# LIVE TEMP
live_persons_temp = 0

camera_running = False

# USERS
USERS = {
    "police": "police123",
    "forensic": "forensic123",
    "cyber": "cyber123",
    "detective": "detect123",
    "security": "security123"
}

model = YOLO("yolov8n.pt")

def is_logged_in():
    return "user" in session


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = request.form.get("role")
        password = request.form.get("password")

        if USERS.get(role) == password:
            session["user"] = role
            return redirect("/dashboard")
        else:
            return render_template("login.html", error="Wrong credentials ❌")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect("/login")

    return render_template(
        "dashboard.html",
        files=len(os.listdir(UPLOAD_FOLDER)),
        persons=last_persons,
        alert=last_alert,
        user=session["user"]
    )


@app.route("/analyze")
def analyze():
    if not is_logged_in():
        return redirect("/login")
    return render_template("index.html")


@app.route("/outputs/<filename>")
def get_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


# ---------------- PDF ----------------
def generate_pdf(count, alert, images=[], mode="upload"):
    name = f"{mode}_report_{datetime.now().strftime('%H%M%S')}.pdf"
    pdf_path = os.path.join(REPORT_FOLDER, name)

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("<b>AI Surveillance Report</b>", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", styles["Normal"]))
    content.append(Paragraph(f"Persons Detected: <b>{count}</b>", styles["Normal"]))
    content.append(Paragraph(f"Alert: <b>{alert}</b>", styles["Normal"]))
    content.append(Spacer(1, 15))

    for img in images:
        path = os.path.join(OUTPUT_FOLDER, img)
        if os.path.exists(path):
            content.append(Image(path, width=300, height=200))
            content.append(Spacer(1, 10))

    doc.build(content)


@app.route("/download_pdf/<type>")
def download_pdf(type):
    files = os.listdir(REPORT_FOLDER)
    filtered = [f for f in files if type in f]

    if not filtered:
        return "No report ❌"

    latest = sorted(filtered)[-1]
    return send_from_directory(REPORT_FOLDER, latest, as_attachment=True)


# ---------------- LIVE CAMERA ----------------
def gen_frames():
    global camera_running, live_persons_temp

    camera_running = True
    live_persons_temp = 0

    cap = cv2.VideoCapture(0)

    while camera_running:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        results = model(frame)

        frame_persons = 0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    frame_persons += 1

                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # 🔥 ONLY LAST FRAME VALUE
        live_persons_temp = frame_persons

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.route("/live")
def live():
    if not is_logged_in():
        return redirect("/login")
    return render_template("live.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


# 🔥 STOP LIVE → FINAL UPDATE
@app.route("/stop_camera")
def stop_camera():
    global camera_running, last_persons, last_alert, live_persons_temp

    camera_running = False

    # 🔥 FINAL SAVE
    last_persons = live_persons_temp
    last_alert = "👤 Live Detection" if live_persons_temp > 0 else "✅ Safe"

    # 🔥 PDF CREATE FOR LIVE
    generate_pdf(last_persons, last_alert, [], "live")

    return redirect("/dashboard")


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload():
    global last_persons, last_alert

    if not is_logged_in():
        return redirect("/login")

    file = request.files.get("video")
    if not file:
        return "No file ❌"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    filename = file.filename.lower()
    detections = 0
    image_paths = []

    if filename.endswith((".jpg", ".png", ".jpeg")):
        frame = cv2.imread(filepath)
        results = model(frame)

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    detections += 1
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        out = f"result_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, out), frame)
        image_paths.append(out)

    else:
        cap = cv2.VideoCapture(filepath)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        detections += 1

        cap.release()

    alert_message = "👤 Person detected" if detections > 0 else "✅ Safe"

    # 🔥 ONLY LAST VALUE
    last_persons = detections
    last_alert = alert_message

    generate_pdf(detections, alert_message, image_paths, "upload")

    return render_template(
        "result.html",
        count=detections,
        alert=alert_message,
        images=image_paths
    )


if __name__ == "__main__":
    app.run(debug=True)