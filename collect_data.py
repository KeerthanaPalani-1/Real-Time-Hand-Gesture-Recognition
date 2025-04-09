import cv2                    # OpenCV for image processing
import mediapipe as mp        # MediaPipe for hand landmark detection
import pandas as pd           # Save data to CSV
import os                     # Check file existence

# Initialize MediaPipe Modules
mp_hands = mp.solutions.hands          # Import the hands module
hands = mp_hands.Hands()               # Create a Hands object for hand detection
mp_draw = mp.solutions.drawing_utils   # Drawing utility for landmarks
# Initialize Data
data = []       # All gesture data collected
label = None    # Current label for the gesture
# Open the Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fail to open camera！")
    exit()
# User Instruction
print("Press number keys 0~5, or 'o' and 'g' to set the label. Press 's' to save data, and 'q' to quit.")

# Main Loop: Capture, Detect, and Extract Hand Gestures
try:
    while True:
        success, img = cap.read()                            # Capture a frame
        if not success:
            print("Failed to access the camera")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Convert BGR to RGB (required by MediaPipe)
        result = hands.process(img_rgb)                     # Analyze the image with MediaPipe to detect hand landmarks

        if result.multi_hand_landmarks:                     # If any hands are detected
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draw key points and connections

                hand_data = []                              # 21 landmarks per hand frame
                for lm in hand_landmarks.landmark:
                    hand_data.extend([lm.x, lm.y, lm.z])    # Add x, y, z coordinates to the list
                # Display label on screen
                if label is not None:
                    cv2.putText(img, f"Label: {label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Keyboard Controls: Save / Set Label / Quit
        key = cv2.waitKey(10) & 0xFF                      # Get keyboard input
        if key == ord('s') and label is not None:         # Press the 'S' key to save the current gesture data.
            if len(hand_data) == 63:                      # Each hand has 21 landmarks 3(x,y,z) = 63
                hand_data.append(label)
                data.append(hand_data)
                print(f"Save gesture {label}")
        elif key in [ord(str(i)) for i in range(0, 6)]:   # Press number keys 0 to 5 to set the label.
            label = int(chr(key))
            print(f"Current label is：{label}")
        elif key == ord('g'):                             # Press the 'g' key to set the label as 'good'
            label = 'good'
            print("Current label is：good")
        elif key == ord('o'):                             # Press the 'o' key to set the label as 'ok'
            label = 'ok'
            print("Current label is：ok")
        elif key == ord('q'):                             # Press the 'q' key to exit the program
            break

# Show the current frame
        cv2.imshow("Hand Gesture Capture", img)

#  Error Handling
except Exception as e:
    print("Error：", e)

# Release Resources and Save Data to CSV
finally:
    cap.release()                    # Release the camera resource
    cv2.destroyAllWindows()          # Close all windows
    if data:
        df = pd.DataFrame(data)

        csv_file = "hand_gesture_data.csv"
        # If the file exists, open it in append mode without writing the header.
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)

        print("Data has been appended and saved to hand_gesture_data.csv")
