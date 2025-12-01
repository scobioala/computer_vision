import cv2
import time
import numpy as np

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_handedness(frame, handedness_list, hand_landmarks):
    h, w, _ = frame.shape

    # Use wrist landmark to position label
    wrist = hand_landmarks.landmark[0]
    x, y = int(w * wrist.x), int(h * wrist.y)

    # handedness_list is a ClassificationList; pick its first classification
    handedness = handedness_list.classification[0]

    label = handedness.label        # "Left" or "Right"
    score = handedness.score        # confidence score

    text = f"{label} ({score:.2f})"

    cv2.putText(
        frame,
        text,
        (x - 30, y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

def draw_bounding_box(frame, hand_landmarks):
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 0, 0), 2)


def run_hand_tracking_on_webcam():

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                print("Empty frame! Skipping.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

                    # Fancy drawing
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Add-ons
                    draw_bounding_box(frame, hand_landmarks)
                    draw_handedness(frame, results.multi_handedness[i], hand_landmarks)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            draw_fps(frame, fps)

            cv2.imshow("Fancy Hand Tracking", cv2.flip(frame, 1))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hand_tracking_on_webcam()
