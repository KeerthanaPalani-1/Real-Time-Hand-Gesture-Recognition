import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

data = []
label = None

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Fail to open camera！")
    exit()

print("Press number keys 1~5, or 'o' and 'g' to set the label. Press 's' to save data, and 'q' to quit.")

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to access the camera")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.extend([lm.x, lm.y, lm.z])

                if label is not None:
                    cv2.putText(img, f"Label: {label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Put the key event handling outside (to ensure it works every frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s') and label is not None:
            if len(hand_data) == 63:
                hand_data.append(label)
                data.append(hand_data)
                print(f"Save gesture {label}")
        elif key in [ord(str(i)) for i in range(1, 6)]:
            label = int(chr(key))
            print(f"Current label is：{label}")
        elif key == ord('g'):
            label = 'good'
            print("Current label is：good")
        elif key == ord('o'):
            label = 'ok'
            print("Current label is：ok")
        elif key == ord('q'):
            break

        cv2.imshow("Hand Gesture Capture", img)

except Exception as e:
    print("Error：", e)

finally:
    cap.release()
    cv2.destroyAllWindows()
    if data:
        df = pd.DataFrame(data)

        csv_file = "hand_gesture_data.csv"
        # If the file exists, open it in append mode without writing the header.
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)

        print("Data has been appended and saved to hand_gesture_data.csv")
