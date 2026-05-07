import cv2
import mediapipe as mp
import numpy as np
from django.http import StreamingHttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse


current_color = (0, 0, 255)
current_thickness = 5
current_eraser = False


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_draw = mp.solutions.drawing_utils

canvas = None
prev_x, prev_y = None, None



def landing(request):
    return render(request, "air/landing.html")

def index(request):
    return render(request, "air/index.html")



def generate_frames():
    global canvas, prev_x, prev_y
    global current_color, current_thickness, current_eraser

    smoothening = 5

    while True:
        success, img = cap.read()
        if not success:
            continue

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

                
                if lm[8].y < lm[6].y:

                    draw_color = (0, 0, 0) if current_eraser else current_color

                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        draw_color,
                        current_thickness,
                        cv2.LINE_AA
                    )

                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None


        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, canvas)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def set_mode(request, mode):
    global current_color, current_eraser

    if request.method == "POST":

        if mode == "red":
            current_color = (0, 0, 255)
            current_eraser = False

        elif mode == "green":
            current_color = (0, 255, 0)
            current_eraser = False

        elif mode == "blue":
            current_color = (255, 0, 0)
            current_eraser = False

        elif mode == "eraser":
            current_eraser = True

    return HttpResponseRedirect(reverse('app'))


def change_size(request, action):
    global current_thickness

    if request.method == "POST":

        if action == "increase":
            current_thickness += 2

        elif action == "decrease":
            current_thickness = max(1, current_thickness - 2)
    return HttpResponseRedirect(reverse('app'))