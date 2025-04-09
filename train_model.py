import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split   # Split the data into training and testing sets
import pickle              # Used to save the trained model locally

# Load hand gesture data from CSV file
df = pd.read_csv("hand_gesture_data.csv")

# Separate features (X) and labels (y)
X = df.iloc[:, :-1]                   # All columns except the last are features
y = df.iloc[:, -1].astype(str)        # Last column is the label; convert to string type

# Print the count of each label to check class distribution
print(df['63'].value_counts())

# Split the dataset into training and testing sets (80% train, 20% test)
# random_state=42 ensures the split is the same every time for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the K-Nearest Neighbors classifier with k=3
model = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
model.fit(X_train, y_train)

# Evaluate the model's accuracy on the test set
print("Training accuracy：", model.score(X_test, y_test))

# Save the trained model to a file using pickle
# This allows loading the model later without retraining
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model has been saved to gesture_model.pkl")

