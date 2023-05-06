# Load imports.
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')

# Time how long it takes to run KNN.
start_time = time.time()

pd.set_option('display.max_columns', None)

# Read in the weather data.
data_raw = pd.read_csv("us_accidents_weather_data.csv")
data_raw = data_raw.sample(n = 400000) # Take on the first couple thousand rows.
data_raw = data_raw.dropna(axis = 0)


# Remove an empty column. There was an error in creating this weather dataset.
data_raw.pop(data_raw.columns[0])

# One-hot encode the Precipitation(in),Weather_Condition,Sunrise_Sunset
weather_condition_dummies = pd.get_dummies(data_raw.Weather_Condition)
sunset_dummies = pd.get_dummies(data_raw.Sunrise_Sunset)

data = pd.concat([data_raw, weather_condition_dummies, sunset_dummies], axis = 1)


# Create the attribute and class data for the KNN classifier.
X = data.drop(['Severity', 'ID', 'Weather_Timestamp', 'Weather_Condition', 'Sunrise_Sunset'], axis=1)
y = data.Severity

# Split data into train and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify = y)


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

accuracies = []

# Get results for various k values.
for i in range(1, 11):
    # Build the KNN classifier.
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)


    # Calculate model metrics.
    model_accuracy = accuracy_score(y_test, y_pred)
    model_precision = precision_score(y_test, y_pred, average = None)
    model_recall = recall_score(y_test, y_pred, average = None)
    model_f1 = f1_score(y_test, y_pred, average = None)
    run_time = time.time() - start_time

    accuracies.append(model_accuracy)

    print("\nNearest Neigbors:", i)
    print("Accuracy:", model_accuracy)
    print("Precision:", model_precision)
    print("Recall:", model_recall)
    print("f1 score:", model_f1)
    print("Run time:", run_time)


# Create accuracy over k plot.
plt.scatter(range(1, 11), accuracies)
plt.show()