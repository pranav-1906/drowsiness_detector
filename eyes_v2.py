import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# File to store dataset
csv_filename = "drowsiness_data.csv"
file_exists = os.path.isfile(csv_filename)

# Create CSV and write header if it doesn't exist
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["EAR", "Label"])  # Label: 1 = eyes open, 0 = eyes closed

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_points, img_w, img_h):
    eye = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_points])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)
THRESHOLD = 0.23  # Adjust this if needed

print("Press 'o' to label as OPEN, 'c' to label as CLOSED, 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks.landmark, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            for pt in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks.landmark[pt].x * w)
                y = int(landmarks.landmark[pt].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Wait for key press to label data
            key = cv2.waitKey(1) & 0xFF
            if key == ord('o'):
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([ear, 1])  # Eyes open
            elif key == ord('c'):
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([ear, 0])  # Eyes closed
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("Collecting EAR Data", frame)