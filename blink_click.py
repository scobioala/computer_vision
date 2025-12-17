import cv2
import time
import numpy as np
import pyautogui
import mediapipe as mp

# 6 key points per eye:
# [outer_corner, upper1, upper2, inner_corner, lower2, lower1]
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_idx):
    """
    Compute EAR using normalized coordinates (x,y in [0,1]).
    landmarks: list of 468 landmarks
    eye_idx: list of 6 indices
    """
    pts = [(landmarks[i].x, landmarks[i].y) for i in eye_idx]

    p0, p1, p2, p3, p4, p5 = pts

    # Vertical distances
    dist_v1 = euclidean_dist(p1, p5)
    dist_v2 = euclidean_dist(p2, p4)

    # Horizontal distance
    dist_h = euclidean_dist(p0, p3)

    if dist_h == 0:
        return 0.0

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return ear


def run_blink_mouse_control():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    EAR_THRESH = 0.23 # below this threshold the eye is considered "closed"
    BLINK_FRAMES = 3 # min consecutive frames eyes closed to count as blink
    DROWSY_FRAMES = 15 # if eyes closed this long, then drowsiness alert

    blink_frame_counter = 0
    total_blinks = 0

    click_cooldown_frames = 0
    CLICK_COOLDOWN = 8 # frames after a click during which we ignore more clicks
    clicked_flag = False

    prev_time = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            ear = None
            drowsy = False
            blink_event = False
            click_event = False

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # draw face mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

                landmarks = face_landmarks.landmark

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
                ear = (left_ear + right_ear) / 2.0

                # blink / drowsiness logic
                if ear < EAR_THRESH:
                    blink_frame_counter += 1
                else:
                    # eye reopened
                    # was this a blink or long closure?
                    if blink_frame_counter >= BLINK_FRAMES:
                        blink_event = True
                        total_blinks += 1

                        # Short-ish closure, then treat as click
                        if blink_frame_counter < DROWSY_FRAMES:
                            if click_cooldown_frames == 0:
                                click_event = True
                                clicked_flag = True
                                click_cooldown_frames = CLICK_COOLDOWN
                                # mouse click
                                pyautogui.click()

                    blink_frame_counter = 0

                if blink_frame_counter >= DROWSY_FRAMES:
                    drowsy = True

            # cooldown for click spam
            if click_cooldown_frames > 0:
                click_cooldown_frames -= 1
            else:
                clicked_flag = False

            # FPS calcs
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # overlay UI
            cv2.putText(
                frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            if ear is not None:
                cv2.putText(
                    frame, f"EAR: {ear:.3f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )

            cv2.putText(
                frame, f"Blinks: {total_blinks}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

            if clicked_flag:
                cv2.putText(
                    frame, "CLICK!", (w // 2 - 70, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3
                )

            if drowsy:
                cv2.putText(
                    frame, "DROWSY / LONG EYES CLOSED", (80, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
                )

            cv2.imshow("Blink-to-Click + Drowsiness Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_blink_mouse_control()
