import cv2
import mediapipe as mp
import numpy as np
import pygame

# 🔊 Alarm
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)

def stop_alarm():
    pygame.mixer.music.stop()

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# EAR calculation
def calculate_ear(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

# Eye points
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)

alarm_on = False
counter = 0

EAR_THRESHOLD = 0.20
FRAME_LIMIT = 20   # 🔥 increase for stability

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]

            right_eye = [(int(face_landmarks.landmark[i].x * w),
                          int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            leftEAR = calculate_ear(left_eye)
            rightEAR = calculate_ear(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye points
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # 🔥 CORE LOGIC
            if ear < EAR_THRESHOLD:
                counter += 1

                # Ignore quick blinks
                if counter >= FRAME_LIMIT:
                    cv2.putText(frame, "DROWSY ALERT!", (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 0, 255), 3)

                    if not alarm_on:
                        alarm_on = True
                        play_alarm()
            else:
                counter = 0
                if alarm_on:
                    stop_alarm()
                alarm_on = False

            # Display EAR + status
            status = "Closed" if ear < EAR_THRESHOLD else "Open"

            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detection (Pro)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()