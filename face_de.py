import face_recognition
import cv2
import numpy as np
import os

# Path to the folder containing roll-number-labeled images
images_path = r"C:\Users\known_faces"

# Initialize lists for known face encodings and labels
known_face_encodings = []
known_face_names = []

# Load and encode all PNG images in the folder
for filename in os.listdir(images_path):
    if filename.endswith(".png"):
        # Extract roll number from filename
        roll_number = os.path.splitext(filename)[0]

        # Load image and encode
        image_path = os.path.join(images_path, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(roll_number)
        else:
            print(f"Warning: No faces found in {filename}")

# Access webcam
video_capture = cv2.VideoCapture(0)

# Variables for performance optimization
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame to 1/4 for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Process every other frame
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Get best match
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Get size of the text for better label box sizing
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size, _ = cv2.getTextSize(name, font, 1.0, 1)
        text_width = text_size[0]
        text_height = text_size[1]

        # Adjust label box to fit text
        cv2.rectangle(frame, (left, bottom), (left + text_width + 12, bottom + text_height + 20), (0, 0, 255), cv2.FILLED)

        # Put the roll number inside
        cv2.putText(frame, name, (left + 6, bottom + text_height + 5), font, 1.0, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow('Video', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
