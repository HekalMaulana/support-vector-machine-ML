import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training SVM Model
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)    

# Predict Training set Result
print(classifier.predict(sc.transform([[30, 87000]])))

# Predicting Test Set Result
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Accuracy Score
print(accuracy_score(y_true=y_test, y_pred=y_pred))
print(confusion_matrix(y_true=y_test, y_pred=y_pred))