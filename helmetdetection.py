from ultralytics import YOLO
import cv2

# Load your YOLO model
model = YOLO("best.pt") #add your trained model to detect helmet
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

    # Show the filtered frame
    cv2.imshow("Helmet Only Detection", filtered_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
