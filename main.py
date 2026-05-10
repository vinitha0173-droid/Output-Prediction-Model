import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Load Dataset
data = pd.read_csv("data.csv")

# Features and Target
X = data[['Hours', 'Attendance', 'Assignments', 'InternalMarks', 'PreviousScore']]
y = data['Result']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

print("----- MODEL ACCURACY -----")

# Train and Test Models
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {accuracy:.2f}")

# Final Model for Prediction
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -------------------------------
# CONFUSION MATRIX
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")

plt.close()

# -------------------------------
# ROC CURVE
# -------------------------------

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("roc_curve.png")

plt.close()

print("\nConfusion Matrix saved as confusion_matrix.png")
print("ROC Curve saved as roc_curve.png")

# -------------------------------
# USER INPUT PREDICTION
# -------------------------------

print("\n----- STUDENT RESULT PREDICTION -----")

hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance Percentage: "))
assignments = float(input("Enter Assignment Score: "))
internal = float(input("Enter Internal Marks: "))
previous = float(input("Enter Previous Score: "))

user_data = pd.DataFrame([[
    hours,
    attendance,
    assignments,
    internal,
    previous
]], columns=[
    'Hours',
    'Attendance',
    'Assignments',
    'InternalMarks',
    'PreviousScore'
])

prediction = model.predict(user_data)

if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")
