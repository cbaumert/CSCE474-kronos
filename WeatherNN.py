# Import python libraries required in this example:
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
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
y = data_raw.Severity - 1


# Split data into train, test, and validation set.
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
print("Before oversampling: ",Counter(y_train))

SMOTE = SMOTE()
X_train, y_train = SMOTE.fit_resample(X_train, y_train)

# Summarize new class distribution.
print("After oversampling: ", Counter(y_train))


inputs = tf.keras.Input(shape=(X.shape[1], ))
x = Dense(64, activation = 'relu')(inputs)
x = Dense(64, activation = 'relu')(x)
x = Dense(64, activation = 'relu')(x)
outputs = Dense(4, activation = 'softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

batch_size = 40
epochs = 10

history = model.fit(
    X_train,
    y_train,
    validation_split = 0.1,
    batch_size = batch_size,
    epochs = epochs,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 3,
            restore_best_weights = True
        )
    ]
)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)

print("Test Accuracy:", model.evaluate(X_test, y_test, verbose = 0)[1])


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
