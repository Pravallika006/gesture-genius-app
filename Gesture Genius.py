import cv2
import mediapipe as mp
import autopy
import pydirectinput as p1
import tkinter as tk
from tkinter import *
import numpy as np
import threading

# Initialize the Tkinter GUI
gui = tk.Tk() 
gui.title("GESTURE GENIUS")
gui.geometry("900x500")
gui.resizable(False, False)
gui.configure(bg="white")

# Set the application icon
img = PhotoImage(file="img.png")
gui.iconphoto(False, img)

# Create a label to display an image
img_label = Label(image=img, height=500, width=925)
img_label.place(x=-15, y=-2)

# Shared variable to indicate whether to exit the recognition loop
exit_recognition = False

# Function to start gesture recognition
def start_gesture_recognition():
    def detect_finger_states(landmarks):
        finger_tips = []
        tip_ids = [4, 8, 12, 16, 20]

        if landmarks[tip_ids[0]][1] > landmarks[tip_ids[0] - 1][1]:
            finger_tips.append(1)
        else:
            finger_tips.append(0)

        for id in range(1, 5):
            if landmarks[tip_ids[id]][2] < landmarks[tip_ids[id] - 3][2]:
                finger_tips.append(1)
            else:
                finger_tips.append(0)

        return finger_tips

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    w_scr, h_scr = autopy.screen.size()
    p_x, p_y = 0, 0

    while not exit_recognition:
        check, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmark_list = []

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                for id, landmark in enumerate(landmarks.landmark):
                    mp_drawing.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)
                    h, w, c = img.shape
                    center_x, center_y = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append([id, center_x, center_y])

            finger_states = detect_finger_states(landmark_list)

            if finger_states[1] == 1 and finger_states[2] == 0 and finger_states[4] == 0:
                x3 = np.interp(landmark_list[8][1], (75, 640 - 75), (0, w_scr))
                y3 = np.interp(landmark_list[8][2], (75, 480 - 75), (0, h_scr))

                c_x = p_x + (x3 - p_x) / 7
                c_y = p_y + (y3 - p_y) / 7

                autopy.mouse.move(w_scr - c_x, c_y)
                p_x, p_y = c_x, c_y

            if finger_states[1] == 0 and finger_states[0] == 1:
                p1.click(button='left')

            if sum(finger_states) == 5:
                p1.keyDown("right")
                p1.keyUp("left")

            elif sum(finger_states) == 0:
                p1.keyDown("left")
                p1.keyUp("right")
            elif finger_states[1] == 1 and finger_states[2] == 1 and finger_states[3] == 1:
                p1.press("space")
            elif finger_states[1] == 1:
                p1.keyUp("right")
                p1.keyUp("left")

        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Function to close the GUI and exit the recognition
def quit_app():
    global exit_recognition
    exit_recognition = True
    gui.destroy()

# Configuration elements for the title bar
Label(text="GENIUS GESTURE", font="Algerian 25 ", fg="white", bg="black").place(x=315, y=30)

# Configuration elements for the Start button
start_button = Button(gui, text="START", width=10, bg="white", font='garamond', command=lambda: threading.Thread(target=start_gesture_recognition).start())
start_button.place(x=200, y=100)

# Configuration elements for the Exit button
exit_button = Button(gui, text="EXIT", width=10, bg="white", font='garamond', fg="black", command=quit_app)
exit_button.place(x=600, y=100)

# Run the tkinter GUI
gui.mainloop()