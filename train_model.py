import pandas as pd
from sklearn.model_selection import train_test_split   # Split the data into training and testing sets
from sklearn.metrics import classification_report      # Classification Report
from sklearn.neighbors import KNeighborsClassifier     # KNN
from sklearn.ensemble import RandomForestClassifier    # Random Forest
from sklearn.svm import SVC                            # SVC
from sklearn.neural_network import MLPClassifier       # MLP
import pickle              # Used to save the trained model locally

# Load hand gesture data from CSV file
df = pd.read_csv("hand_gesture_data.csv")

# Separate features (X) and labels (y)
X = df.iloc[:, :-1]                   # All columns except the last are features
y = df.iloc[:, -1].astype(str)        # Last column is the label; convert to string type

# Check label distribution
print("Label Distribution:\n", y.value_counts())

# Split the dataset into training and testing sets (80% train, 20% test)
# random_state=42 ensures the split is the same every time for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 1. KNN Model ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test)
print(f"\n🔵 KNN Accuracy: {knn_acc:.4f}")
print("KNN Classification Report:\n", classification_report(y_test, knn.predict(X_test)))

# --- 2. Random Forest Model ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)
print(f"\n🌲 Random Forest Accuracy: {rf_acc:.4f}")
print("Random Forest Classification Report:\n", classification_report(y_test, rf.predict(X_test)))

# --- 3. SVM ---
svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # gamma='scale' is usually good enough
svm.fit(X_train, y_train)
accuracy = svm.score(X_test, y_test)
print(f"\n⚪ SVM Accuracy: {accuracy:.4f}")
print("SVM Classification Report:\n", classification_report(y_test, svm.predict(X_test)))


# --- 4. MLP classifier ---  Multi-Layer Perceptron, a type of feedforward neural network.
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers: one with 100 neurons, one with 50
    activation='relu',             # Non-linear Activation function
    solver='adam',                 # Adaptive gradient descent optimization algorithm
    max_iter=1000,                  # Maximum number of training iterations
    random_state=42,
)
mlp.fit(X_train, y_train)
accuracy = mlp.score(X_test, y_test)
print(f"\n🧠 MLP Accuracy: {accuracy:.4f}")
print("MLP Classification Report:\n", classification_report(y_test, mlp.predict(X_test)))


# save model
with open("gesture_model_mlp.pkl", "wb") as f:
    pickle.dump(mlp, f)
print("Random Forest model saved to gesture_model_mlp.pkl")

