# Series Predction with RNN
# Mean Absolute Error =

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


# %%
data = pd.read_csv("TCS.NS.csv").dropna()
data = data.reset_index()
series = data['Close']

max_close_value = max(series)

series = series / max_close_value


# %%
plt.figure(figsize=(25, 6))
plt.plot([i for i in range(len(series))], series)
plt.show()


# %%
window_size = 101


# %%
def windowed_dataset(series, window_size):
    X = []
    y = []

    for i in range(len(series) + 1 - window_size):
        X.append(series[i: i + window_size - 1])
        y.append(series[i + window_size - 1])

    return X, y


# %%
X, y = windowed_dataset(series, window_size)

X = np.array(X)
y = np.array(y)


# %%
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)


# %%

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(1)
])

model.compile(loss="mae", optimizer="adam", metrics=['MAE'])
history = model.fit(train_X, train_y, epochs=100,
                    validation_data=(test_X, test_y))


# %%
prediction = model.predict(test_X)
prediction = prediction.flatten()


# %%
mae = 0
for i in range(len(test_X)):
    mae += abs(test_y[i] - prediction[i])
mae = mae / len(test_X)
print(f'Mean Absolute Error = {mae}')


# %%
plt.figure(figsize=(25, 6))
plt.title("Prediction on test data")
plt.scatter([i for i in range(len(test_X))], test_y, color='g', label='Actual')
plt.plot([i for i in range(len(test_X))],
         prediction, color='b', label='Predicted')
plt.legend()
plt.show()


# %%
future_time = 1
future_prediction = []
temp = X[-1]


for i in range(future_time):
    p = model.predict(np.array([temp]))[0][0]
    future_prediction.append(p)
    temp = np.concatenate(
        (np.array(temp[1:]).flatten(), np.array([p])), axis=0)


# %%
for i in range(future_time):
    if future_prediction[i] > max_close_value:
        max_close_value = future_prediction[i]
    print(future_prediction[i] * max_close_value)
