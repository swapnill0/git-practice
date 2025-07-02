from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

from collections import deque

from flask_mail import Mail, Message
app = Flask(__name__)
# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'kunalkhalkar433@gmail.com'  # Replace
app.config['MAIL_PASSWORD'] = 'lvcs njzl rwtr dqlb'     # Replace (App Password if using Gmail)
app.config['MAIL_DEFAULT_SENDER'] = 'kunalkhalkar433@gmail.com'

mail = Mail(app)


UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CCTV_STREAM'] = None  # Will be set dynamically

# Load the sequence-based model
model = load_model('suspicious_activity_detection.1.h5')
labels = ['Suspicious', 'Non-Suspicious']

def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    return frame

def send_screenshot_email(image_path):
    msg = Message('🚨 Suspicious Activity Detected', recipients=['kunal36khalkar@gmail.com'])  # Change recipient
    msg.body = "Suspicious activity was detected. Screenshot attached."
    
    with open(image_path, 'rb') as f:
        msg.attach("screenshot.jpg", "image/jpeg", f.read())
    
    mail.send(msg)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' in request.files and request.files['video'].filename != '':
            file = request.files['video']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('play_video', filename=filename))

        cctv_url = request.form.get('cctv_url')
        if cctv_url:
            app.config['CCTV_STREAM'] = cctv_url
            return redirect(url_for('play_cctv'))

        return "Please upload a video or enter a CCTV stream URL."

    return render_template('index.html')

@app.route('/play/<filename>')
def play_video(filename):
    return render_template('play.html', filename=filename)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(generate_video(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

from collections import deque
import time

def generate_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(video_path)

    frame_buffer = deque(maxlen=30)
    last_prediction = "Analyzing..."
    frame_count = 0
    skip_rate = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        resized = cv2.resize(frame, (64, 64))
        normalized = resized / 255.0
        frame_buffer.append(normalized)

        if len(frame_buffer) == 30 and frame_count % skip_rate == 0:
            sequence = np.array([frame_buffer])
            prediction = model.predict(sequence, verbose=0)[0][0]
            last_prediction = labels[int(prediction > 0.5)]
            frame_buffer.popleft()

            if last_prediction == "Suspicious":
                screenshot_path = f"static/screenshot_video_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, frame)
                with app.app_context():  # ✅ this is the fix
                    send_screenshot_email(screenshot_path)  

        cv2.putText(frame, f"Prediction: {last_prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if last_prediction == 'Suspicious' else (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route('/play-cctv')
def play_cctv():
    return render_template('play_cctv.html')

@app.route('/cctv_feed')
def cctv_feed():
    return Response(generate_cctv(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_cctv():
    cctv_url = app.config.get('CCTV_STREAM')
    if not cctv_url:
        return

    cap = cv2.VideoCapture(cctv_url)
    frame_buffer = deque(maxlen=30)
    last_prediction = "Analyzing..."
    frame_count = 0
    skip_rate = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        resized = cv2.resize(frame, (64, 64))
        normalized = resized / 255.0
        frame_buffer.append(normalized)

        if len(frame_buffer) == 30 and frame_count % skip_rate == 0:
            sequence = np.array([frame_buffer])
            prediction = model.predict(sequence, verbose=0)[0][0]
            last_prediction = labels[int(prediction > 0.5)]
            frame_buffer.popleft()

            if last_prediction == "Suspicious":
                screenshot_path = f"static/screenshot_cctv_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, frame)
                with app.app_context():
                    send_screenshot_email(screenshot_path)

        cv2.putText(frame, f"CCTV Prediction: {last_prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if last_prediction == 'Suspicious' else (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


if __name__ == '__main__':
    app.run(debug=True)
