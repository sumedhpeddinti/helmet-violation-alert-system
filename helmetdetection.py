from ultralytics import YOLO
import cv2

# Load your YOLO model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO inference
    results = model(frame)

    # Copy frame to draw on
    filtered_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Only keep 'helmet' class (class ID = 3)
            if cls_id == 3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(filtered_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    filtered_frame, f"Helmet {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

    # Show the filtered frame
    cv2.imshow("Helmet Only Detection", filtered_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
