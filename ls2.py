import cv2
import mediapipe as mp
import pydirectinput as p1
import tkinter as tk
from tkinter import *
import numpy as np
import threading
from PIL import Image, ImageTk

class GestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        root.title("GESTURE GENIUS")
        root.geometry("900x500")
        root.resizable(False, False)
        root.configure(bg="white")

        # Set the application icon
        img = PhotoImage(file="product2 (1).png")
        root.iconphoto(False, img)

        # Create a label to display an image
        self.img_label = Label(image=img, height=500, width=925)
        self.img_label.place(x=-15, y=-2)

        # Shared variable to indicate whether to exit the recognition loop
        self.exit_recognition = False

        # Variables for gesture detection
        self.gesture_history = []

        # Configuration elements for the title bar
        Label(text="GENIUS GESTURE", font="Algerian 25 ", fg="white", bg="black").place(x=315, y=30)

        # Configuration elements for the Start button
        start_button = Button(root, text="START", width=10, bg="white", font='garamond', command=self.start_gesture_recognition)
        start_button.place(x=200, y=100)

    def detect_finger_states(self, landmarks):
        finger_tips = [0, 0, 0, 0, 0]  # Initialize with all fingers down

        if len(landmarks) >= 21:
            # Check if thumb is up
            if landmarks[4][1] < landmarks[3][1]:
                finger_tips[0] = 1

            # Check if other fingers are up
            for id in range(1, 5):
                if landmarks[id * 4][2] < landmarks[id * 4 - 2][2]:
                    finger_tips[id] = 1

        return finger_tips

    def detect_exit_gesture(self, finger_states):
        self.gesture_history.append(finger_states)
        if len(self.gesture_history) > 10:
            self.gesture_history.pop(0)

        # Check for the "three-finger pinch" gesture in the last few frames
        if len(self.gesture_history) == 10:
            pinch_gesture = [0, 0, 0, 1, 1]
            for i in range(10):
                if self.gesture_history[i] != pinch_gesture:
                    return False
            return True
        return False

    def start_gesture_recognition(self):
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
        w_scr, h_scr = p1.size()
        p_x, p_y = 0, 0

        while not self.exit_recognition:
            ret, img = cap.read()

            if not ret:
                print("Error: Could not capture frame.")
                break

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

                finger_states = self.detect_finger_states(landmark_list)

                if self.detect_exit_gesture(finger_states):
                    self.exit_recognition = True
                    break

                if finger_states[1] == 1 and finger_states[2] == 0 and finger_states[4] == 0:
                    x3 = np.interp(landmark_list[8][1], (75, w - 75), (0, w_scr))
                    y3 = np.interp(landmark_list[8][2], (75, h - 75), (0, h_scr))

                    c_x = p_x + (x3 - p_x) / 7
                    c_y = p_y + (y3 - p_y) / 7

                    p1.moveTo(c_x, c_y, duration=0.2)
                    p_x, p_y = c_x, c_y

            img_rgb_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self.img_label.img = ImageTk.PhotoImage(image=img_rgb_pil)
            self.img_label.config(image=self.img_label.img)

            self.root.update()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gui = tk.Tk()
    app = GestureRecognitionApp(gui)
    gui.mainloop()
