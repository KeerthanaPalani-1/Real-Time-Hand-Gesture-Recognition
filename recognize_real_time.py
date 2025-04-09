import cv2
import mediapipe as mp
import pickle

# Load the trained gesture recognition model from file
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils    # Drawing utility to show hand landmarks

# Start video capture from default webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()                         # Capture a frame
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # Convert to RGB for MediaPipe
    result = hands.process(img_rgb)                   # Process the image to detect hands

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)   # Draw landmarks

            hand_data = []                               # 21 landmarks per hand frame
            for lm in hand_landmarks.landmark:
                hand_data.extend([lm.x, lm.y, lm.z])     # Add x, y, z coordinates to the list

            prediction = model.predict([hand_data])[0]   # Predict the gesture class
            # Display prediction
            cv2.putText(img, f"Gesture is : {prediction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    cv2.imshow("Real-time Hand Gesture Recognition", img)   # Show video with annotations
    if cv2.waitKey(1) & 0xFF == ord('q'):                # Press 'q' to quit
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
