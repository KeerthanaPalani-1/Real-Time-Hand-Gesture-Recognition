import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("hand_gesture_data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(str)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print("Training accuracy：", model.score(X_test, y_test))

# Save Model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model has been saved to gesture_model.pkl")
