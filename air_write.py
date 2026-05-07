import cv2
import mediapipe as mp
import numpy as np
from django.http import StreamingHttpResponse

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_draw = mp.solutions.drawing_utils


def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    canvas = None
    prev_x, prev_y = None, None
    smoothening = 5

    color = (0, 0, 255)
    thickness = 5
    eraser = False

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)

        if canvas is None:
            canvas = np.zeros_like(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = img.shape
                lm = hand_landmarks.landmark

                curr_x = int(lm[8].x * w)
                curr_y = int(lm[8].y * h)

                if prev_x is None:
                    prev_x, prev_y = curr_x, curr_y

                x = prev_x + (curr_x - prev_x) // smoothening
                y = prev_y + (curr_y - prev_y) // smoothening

                if lm[8].y < lm[6].y:  # index finger up

                    draw_color = (0, 0, 0) if eraser else color

                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        draw_color,
                        thickness,
                        cv2.LINE_AA
                    )

                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        # Merge canvas
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, canvas)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )