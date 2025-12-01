import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.linear_model import LinearRegression

mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = 468
RIGHT_IRIS = 473

LEFT_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EAR_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(lms, indices):
    p = [(lms[i].x, lms[i].y) for i in indices]
    v1 = np.linalg.norm(np.array(p[1])-np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2])-np.array(p[4]))
    h  = np.linalg.norm(np.array(p[0])-np.array(p[3]))
    return (v1+v2)/(2*h)

def get_iris_position(lms, w, h):
    iris_x = lms[LEFT_IRIS].x
    iris_y = lms[LEFT_IRIS].y
    return iris_x, iris_y

def run_calibrated_gaze_ui():

    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)

    UI_W, UI_H = 800, 500

    # calibration points in UI
    calib_targets = [
        (100, 100),
        (700, 100),
        (400, 250),
        (100, 400),
        (700, 400)
    ]

    collected_iris = []
    collected_targets = []

    blink_counter = 0
    EAR_THRESH = 0.23
    BLINK_FRAMES = 2
    cooldown = 0

    model_x = LinearRegression()
    model_y = LinearRegression()

    calibrated = False
    current_target = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True
    ) as fm:

        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)

            ui = np.zeros((UI_H, UI_W, 3), dtype=np.uint8)

            # draw calibration targets or UI state
            if not calibrated:
                tx, ty = calib_targets[current_target]
                cv2.circle(ui, (tx, ty), 20, (0, 255, 255), -1)
                cv2.putText(ui, "Look at the dot and BLINK",
                            (200, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255,255,255), 2)
            else:
                cv2.putText(ui, "Tracking...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255,255,255), 2)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark

                # EAR for blink detection
                leftEAR = eye_aspect_ratio(lms, LEFT_EAR_IDX)
                rightEAR = eye_aspect_ratio(lms, RIGHT_EAR_IDX)
                ear = (leftEAR + rightEAR) / 2

                iris_x, iris_y = get_iris_position(lms, w, h)

                if not calibrated:
                    if ear < EAR_THRESH:
                        blink_counter += 1
                    else:
                        if blink_counter >= BLINK_FRAMES and cooldown == 0:
                            collected_iris.append([iris_x, iris_y])
                            collected_targets.append(list(calib_targets[current_target]))
                            cooldown = 10
                            current_target += 1
                            if current_target == len(calib_targets):
                                # train models
                                X = np.array(collected_iris)
                                Y = np.array(collected_targets)
                                model_x.fit(X, Y[:,0])
                                model_y.fit(X, Y[:,1])
                                calibrated = True
                        blink_counter = 0

                    if cooldown > 0:
                        cooldown -= 1

                else:
                    predicted_x = int(model_x.predict([[iris_x, iris_y]])[0])
                    predicted_y = int(model_y.predict([[iris_x, iris_y]])[0])
                    # draw cursor
                    cv2.circle(ui,(predicted_x, predicted_y), 10,(0,255,0), -1)
            
            cv2.imshow("Webcam", frame)
            cv2.imshow("Calibrated Eye UI", ui)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_calibrated_gaze_ui()
