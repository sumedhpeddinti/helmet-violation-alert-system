import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import os
import time
import pandas as pd
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# --- CONFIGURATION ---
YOLO_MODEL = "best.pt"
KNOWN_FACES_DIR = r"C:\Users\known_faces"
EMAIL_CSV = "student_data.csv"
SENDGRID_API_KEY = "     "  # Replace with your actual SendGrid key
FROM_EMAIL = "     @email.com"  # Replace with your verified sender email

# --- Load Emails ---
email_df = pd.read_csv(EMAIL_CSV)
email_map = dict(zip(email_df["Roll Number"].astype(str), email_df["Email"]))

# --- Load YOLO Model ---
model = YOLO(YOLO_MODEL)

# --- Load Known Faces ---
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".png"):
        roll_number = os.path.splitext(filename)[0]
        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(roll_number)

print(f"[INFO] Loaded {len(known_face_names)} known faces.")

# --- Tracking Variables ---
violation_times = {}  # {roll_number: first_detected_time}
sent_mails = set()    # to avoid multiple emails per session

# --- Email Function ---
def send_violation_email(roll_number):
    to_email = email_map.get(roll_number)
    if not to_email:
        print(f"[WARNING] Email not found for {roll_number}")
        return

    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=to_email,
        subject="Helmet Violation Detected",
        html_content=f"""
        Hello,<br><br>
        Roll number <b>{roll_number}</b> was detected without a helmet at the college gate.<br>
        <b>Time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}<br>
        <b>Fine:</b> ₹50<br>
        <b>Link:</b> [Put your link here]<br><br>
        Please ensure safety by wearing a helmet.<br><br>
        Regards,<br>
        College Surveillance System
        """
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
        print(f"[EMAIL SENT] To {to_email}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

# --- Webcam Feed ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    head_detected = any(model.names[int(b.cls[0])].lower() == "head" for b in results.boxes)

    if head_detected:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Track violation time
            if name != "Unknown":
                now = time.time()
                if name not in violation_times:
                    violation_times[name] = now
                elif now - violation_times[name] >= 5 and name not in sent_mails:
                    send_violation_email(name)
                    sent_mails.add(name)
            else:
                continue

            # Draw face box
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Draw helmet/head boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        color = (0, 255, 0) if class_name.lower() == "helmet" else (0, 165, 255)
        label = f"{class_name} {box.conf[0]:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Helmet Detection & Violation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
