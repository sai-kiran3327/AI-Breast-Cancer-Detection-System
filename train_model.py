import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("breast_cancer_bd_updated.csv")

print(df.head())
print("\nColumns:")
print(df.columns)

print("\nUnique Class Values:")
print(df["class"].unique())

print("\nClass Count:")
print(df["class"].value_counts())

print("Dataset Shape :", df.shape)

# Remove ID column if present
if "Sample code number" in df.columns:
    df = df.drop(columns=["Sample code number"])

# ==========================
# Features & Target
# ==========================

X = df.drop("class", axis=1)
y = df["class"]

print("\nUnique Classes:", y.unique())
print("\nClass Distribution:\n")
print(y.value_counts())

# Convert labels from 2/4 to 0/1 if necessary
if set(y.unique()) == {2, 4}:
    print("\nConverting labels...")
    y = y.replace({2: 0, 4: 1})

print("\nLabels after conversion:")
print(y.value_counts())

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Train Model
# ==========================

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Evaluation
# ==========================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("\nTraining Accuracy :", accuracy_score(y_train, train_pred))
print("Testing Accuracy  :", accuracy_score(y_test, test_pred))

print("\nClassification Report\n")
print(classification_report(y_test, test_pred))

# ==========================
# Save Model
# ==========================

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel Saved Successfully.")