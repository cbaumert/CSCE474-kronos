# Load imports.
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
import pandas as pd
import time


# Time how long it takes to run KNN.
start_time = time.time()

# Read in the weather data.
data_raw = pd.read_csv("us_accidents_weather_data_CLEAN.csv")
data_raw = data_raw.dropna(axis = 0)
#data_raw = data_raw.sample(n = 5000) # Randomly choose a couple thousand rows.

# Remove an empty column. There was an error in creating this weather dataset.
data_raw.pop(data_raw.columns[0])

# One-hot encode the Precipitation(in),Weather_Condition,Sunrise_Sunset.
weather_condition_dummies = pd.get_dummies(data_raw.Weather_Condition)
sunset_dummies = pd.get_dummies(data_raw.Sunrise_Sunset)

data = pd.concat([data_raw, weather_condition_dummies, sunset_dummies], axis = 1)


# Create the attribute and class data for the Random Forest classifier.
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


# Build the Random Forest Classifier.
randForestClassifier = RandomForestClassifier()

param_dist = {'n_estimators': randint(50, 5000), 'max_depth': randint(1, 20)}
rand_search = RandomizedSearchCV(randForestClassifier, param_distributions = param_dist, n_iter = 5, cv = 5)
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_
print('Best hyperparameters:',  rand_search.best_params_)


# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

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


# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix = cm).plot()
plt.show()


# Create a series containing feature importances from the model and feature names from the training data.
feature_importances = pd.Series(best_rf.feature_importances_, index = X_train.columns).sort_values(ascending = False)
top_importances = feature_importances[:10]

# Plot a simple bar chart
top_importances.plot.bar()
plt.show()

print("\nTop 10 important features:\n", top_importances)

