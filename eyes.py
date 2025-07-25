import cv2
import mediapipe as mp
import numpy as np
import pygame  

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Initialize Pygame for alarm
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")  # Ensure alarm.mp3 is in the same folder

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(landmarks, eye_points, img_w, img_h):
    """Calculates EAR (Eye Aspect Ratio) to detect closed eyes."""
    eye = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_points])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)  # EAR formula
    return ear

# Open webcam
cap = cv2.VideoCapture(0)

frame_counter = 0  
THRESHOLD = 0.23   
CLOSED_FRAMES = 15  
alarm_playing = False  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            for point in LEFT_EYE + RIGHT_EYE:
                x, y = int(face_landmarks.landmark[point].x * w), int(face_landmarks.landmark[point].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if avg_ear < THRESHOLD:
                frame_counter += 1
                if frame_counter >= CLOSED_FRAMES and not alarm_playing:
                    pygame.mixer.music.play(-1)  
                    alarm_playing = True  
            else:
                frame_counter = 0  
                if alarm_playing:
                    pygame.mixer.music.stop()  
                    alarm_playing = False  

            # Display warning text at the center of the screen
            if alarm_playing:
                text = "DROWSY! WAKE UP!"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()