# 🖐️ Real-Time Hand Gesture Recognition with MediaPipe

This project allows real-time hand gesture recognition using a webcam, powered by [MediaPipe](https://mediapipe.dev/) for hand landmark detection and a simple machine learning classifier (KNN) for gesture classification.

It supports:
- Numeric gestures: `1`, `2`, `3`, `4`, `5`
- Custom gestures: `👍` (good), `👌` (ok)

## 📸 Demo
![Demo](gesture.gif)

---

## 🚀 Features
- Live webcam input using OpenCV
- Hand landmark extraction using MediaPipe (21 keypoints per hand)
- Gesture data collection and CSV saving
- Simple classifier training (KNN)
- Real-time gesture prediction and overlay display
- Easily extendable with more gestures

---

## 📁 Project Structure

```
gesture_project/
├── collect_data.py         # collect gesture
├── train_model.py          # tain model
├── recognize_real_time.py  # Real time recognize
└── README.md               # read me 
```



---

## 🧪 How to Use

### 1. Install Dependencies

```bash
pip install mediapipe opencv-python scikit-learn numpy pandas
```

### 2. Collect Hand Gesture Data

```
python collect_data.py
```

Instructions:

- Press keys `1` to `5` to label number gestures
- Press `g` for 👍 (good)
- Press `o` for 👌 (ok)
- Press `s` to save a frame with the current label
- Press `q` to quit and save the CSV

Each saved frame will store 63 values (21 keypoints × 3D) + 1 label.

![Demo](gesture_collect.gif)

### 3. Train the Classifier

```
python train_model.py
```

This will load the CSV, train a KNN classifier, and save the model to `gesture_model.pkl`.

### 4. Real-Time Prediction

```
python recognize_real_time.py
```

Your webcam will open and show predicted gestures live on the screen.



## 📥Other Resources

[Model I trained](https://drive.google.com/file/d/1zXLfWREJxeB_WoMbGbVZHWxcQc4tC56s/view?usp=drive_link)

[Data I collected](https://drive.google.com/file/d/1rwz_Nib5BfTYvAwFfAleCFdFbeHAjJRq/view?usp=sharing)