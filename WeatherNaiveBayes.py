# Load imports.
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
import time


# Time how long it takes to run KNN.
start_time = time.time()

# Read in the weather data.
data_raw = pd.read_csv("us_accidents_weather_data.csv")
data_raw = data_raw.dropna(axis = 0)

# Remove an empty column. There was an error in creating this weather dataset.
data_raw.pop(data_raw.columns[0])


# Create the attribute and class data for the Random Forest classifier.
X = data_raw.drop(['Severity', 'ID', 'Weather_Timestamp', 'Weather_Condition', 'Sunrise_Sunset'], axis=1)
y = data_raw.Severity

# Split data into train and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


#****UNDER SAMPLING CODE SNIPPET
# Summarize class distribution.
# print("Before undersampling: ", Counter(y_train))
#
# undersample = RandomUnderSampler(sampling_strategy='majority')
# X_train, y_train = undersample.fit_resample(X_train, y_train)
#
# Summarize new class distribution.
# print("After undersampling: ", Counter(y_train))


#****OVER SAMPLING CODE SNIPPET
# Summarize class distribution.
# print("Before oversampling: ",Counter(y_train))
#
# SMOTE = SMOTE()
# X_train, y_train = SMOTE.fit_resample(X_train, y_train)
#
# Summarize new class distribution.
# print("After oversampling: ", Counter(y_train))


# Create a Gaussian Naive Bayes classifier.
NBClassifier = GaussianNB()
NBClassifier.fit(X_train, y_train)
y_pred = NBClassifier.predict(X_test)


# Calculate model metrics.
model_accuracy = accuracy_score(y_test, y_pred)
model_precision = precision_score(y_test, y_pred, average = None)
model_recall = recall_score(y_test, y_pred, average = None)
model_f1 = f1_score(y_test, y_pred, average = None)
run_time = time.time() - start_time

print("\nAccuracy:", model_accuracy)
print("Precision:", model_precision)
print("Recall:", model_recall)
print("f1 score:", model_f1)
print("Run time:", run_time)


# Create the confusion matrix.
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix = cm).plot()
plt.show()