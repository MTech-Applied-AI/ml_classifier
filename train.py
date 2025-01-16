import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, names=columns)

# Encode labels (string -> numeric)
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

# Features and Target
X = df.drop(columns=['class'])
y = df['class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the model
with open("model/classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

# Save the Label Encoder (useful for decoding predictions)
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save dataset split counts
split_info = {"train_count": len(X_train), "test_count": len(X_test)}
with open("model/split_info.pkl", "wb") as f:
    pickle.dump(split_info, f)

print("Model trained and saved successfully!")
