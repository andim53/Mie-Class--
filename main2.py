# --- Required Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# --- Initialization for reproducibility (Windows/MacOS example combined)
def init_random_seed(iseed):
    import os
    os.environ["PYTHONHASHSEED"] = str(iseed)
    os.environ["TF_DETERMINISTIC_OPS"] = "true"
    np.random.seed(iseed)
    tf.random.set_seed(iseed)

# --- Plot Training History
def PlotHistory(history, metrics, target):
    '''
    Plot the training and validation metrics over epochs.
    metrics: str - Metric to plot (e.g., 'MSE').
    target: str - Target variable name.
    '''
    hist = pd.DataFrame(history.history)
    hist["Epoch"] = history.epoch
    plt.figure()
    plt.xlabel("Number of Epochs")
    plt.ylabel(f"{metrics} of {target}")
    plt.plot(hist["Epoch"], hist[metrics], label="Training")
    plt.plot(hist["Epoch"], hist["val_" + metrics], label="Validation")
    plt.yscale('log')  # Optional for scaling
    plt.legend()
    plt.title(f"Training History: {metrics}")
    plt.show()

# --- Plot Actual vs Predicted
def PlotCorrelation(y_actual, y_predicted):
    '''
    Plot Actual vs Predicted values.
    y_actual: np.array - Ground truth values.
    y_predicted: np.array - Predicted values from the model.
    '''
    y_actual = np.array(y_actual).flatten()
    y_predicted = np.array(y_predicted).flatten()
    plt.figure()
    plt.axis('equal')
    plt.axis('square')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.scatter(y_actual, y_predicted, color='blue', alpha=0.3)
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='gray', linestyle='--')
    plt.title("Actual vs Predicted")
    plt.show()

# --- Initialization
init_random_seed(1)

# --- Read Input Data
filename_inp = "magn_CoFe9.csv"
target = "dEform_eV"  # Target variable
df = pd.read_csv(filename_inp)

# --- Separate X/Y Data
data_x = df[["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]]
data_y = df[target]
train_size = 0.7
train_x, test_x, train_y, test_y = train_test_split(
    data_x, data_y, train_size=train_size, shuffle=True, random_state=1
)

# --- Build Model
activation = "relu"
optimizer = "SGD"

model = Sequential([
    Dense(32, input_shape=[train_x.shape[1]]),
    Dense(32, activation=activation),
    Dense(32, activation=activation),
    Dense(32, activation=activation),
    Dense(1)
])

# --- Compile Model
model.compile(loss="MSE", optimizer=optimizer, metrics=["MSE"])
model.summary()

# --- Train Model
history = model.fit(
    train_x, train_y,
    epochs=100,
    validation_split=0.3,
    verbose=0
)

# --- Plot Training History
#PlotHistory(history, "MSE", target)

# --- Evaluate Model
result_train = model.evaluate(train_x, train_y, verbose=0)
result_test = model.evaluate(test_x, test_y, verbose=0)

#print("Validation Results:")
print(f"Train Loss : {result_train[0]:.4E}")
print(f"Test Loss  : {result_test[0]:.4E}")

# --- Predictions
predict_train = model.predict(train_x).flatten()
predict_test = model.predict(test_x).flatten()

# --- Plot Actual vs Predicted
#PlotCorrelation(train_y, predict_train)
#PlotCorrelation(test_y, predict_test)

# --- R2 Score
r2score_train = r2_score(train_y, predict_train)
r2score_test = r2_score(test_y, predict_test)

print(f"R2 Score (Train): {r2score_train:.4E}")
print(f"R2 Score (Test) : {r2score_test:.4E}")
