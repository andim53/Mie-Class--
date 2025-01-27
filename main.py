# --- initialization for generating random numbers
def init_WinOS(iseed):
    # --- clear session
    import keras.backend as K
    K.clear_session()
    # --- set OS environment
    import os
    os.environ["PYTHONHASHSEED"] = str(iseed)
    os.environ["TF_DETERMINISTIC_OPS"] = "true"
    os.environ["TF_CODNN_DETERMINISTIC"] = "true"
    # --- initialization
    np.random.seed(iseed)
    tr.random.set_seed(iseed)

def init_MacOS(iseed):
    # --- initialization
    tf.random.set_seed(iseed)

# --- Function for plotting training history
def PlotHistory(history, metrics):
    '''
    metrics = {MSE|MAE|...}
    '''
    # --- get epoch
    hist = pd.DataFrame(history.history)
    hist["Epoch"] = history.epoch
    # --- plot figures
    plt.figure()
    plt.xlabel("Number of epochs")
    plt.ylabel(f"{metrics} of {target}")
    plt.plot(hist["Epoch"], hist[metrics], label="Training")
    plt.plot(hist["Epoch"], hist["val_"+metrics], label="Validation")
    # plt.plot(hist["Epoch"].values, hist[metrics].values, ␣ →label="Training") # for pandas>3.4 maybe
    # plt.plot(hist["Epoch"].values, hist["val_"+metrics].values,␣ →label="Validation")
    plt.yscale('log')
    plt.legend()
    plt.show()

# --- Function for plotting actual-predicted plot
def PlotCorrelation(y_train, y_predict):
    # --- plot figures
    plt.axis('equal')
    plt.axis('square')
    plt.xlabel(f"Actual")
    plt.ylabel(f"Predicted")
    plt.scatter(y_train, y_predict, color='blue', alpha=0.3)
    #plt.xlim([-1.2, 0])
    #plt.ylim([-1.2, 0])
    #plt.plot([-100, 100], [-100, 100], color='gray')
    plt.show()

# --- import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

# --- Version information
import platform
import matplotlib
import tensorflow
import sklearn
import keras

ver = "0.0.1" # version of this program
#print(f"Vesion information:")
#print(f" This program : {ver}")
#print(f" Python : {platform.python_version()}")
#print(f" Pandas : {pd.__version__}")
#print(f" Numpy : {np.__version__}")
#print(f" Matplotlib : {matplotlib.__version__}")
#print(f" TensorFlow : {tensorflow.__version__}")
#print(f" Scikit-learn : {sklearn.__version__}")
#print(f" Keras : {keras.__version__}")

# --- initialization (only for Windows-OS)
iseed = 1
# OS = "Mac" # Win/Mac
# if OS == 'Win':
#     init_WinOS(iseed)
# elif OS == 'Mac':
#     init_MacOS(iseed)

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Read input data
filename_inp = "magn_CoFe9.csv"
target = "dEform_eV"
df = pd.read_csv(filename_inp)
# --- separate X/Y data
data_x = df[["a1", "a2","a3","a4","a5","a6","a7","a8","a9"]]
data_y = df[[target]]
train_size = 0.7
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,train_size=train_size, shuffle=True, random_state=1)
#print(f"Number of training data: {len(train_x):>5}")
#print(f"Number of testing data: {len(test_x):>5}")

from tensorflow import keras
from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import SGD

# --- Model parameters
activation = "relu"
# Set a specific learning rate
learning_rate = 0.01
optimizer = SGD(learning_rate=learning_rate)

#print("Building a model ...")

# --- Set NN architecture
model = keras.Sequential([
    Dense(8, activation=activation, input_shape=[train_x.shape[1]]),
    Dense(8, activation="relu"),
    Dense(1)
])

# --- Compile the model
model.compile(loss="MSE", optimizer=optimizer, metrics=["MSE"])
model.summary()

# --- Fit the model
#print("Fitting the model started ...")
model_history = model.fit(
    train_x, train_y,
    epochs=100,
    validation_split=0.3,
    verbose=0  # Change to 1 or 2 to see training progress
)

# --- plot training history
PlotHistory(model_history, "MSE")

# --- evaluate the NN model (for training)
result_train = model.evaluate(train_x, train_y, verbose=0)
print("Validation for training data")
print(f" Loss function : {result_train[0]: .4E}")
# --- evaluate the NN model (for testing)
result_test = model.evaluate(test_x, test_y, verbose=0)
print("Validation for testing data")
print(f" Loss function : {result_test[0]: .4E}")

# --- predict the training/testing model
predict_train = model.predict(train_x).flatten() # reproducibility
predict_test = model.predict(test_x).flatten()
# --- Plot actual-predicted
PlotCorrelation(train_y, predict_train)
PlotCorrelation(test_y, predict_test)
# --- evaluate the R2 score
r2score_train = r2_score(train_y, predict_train)
r2score_test = r2_score(test_y, predict_test)
print(f"R2 score for training data : {r2score_train: .4E}")
print(f"R2 score for testing data : {r2score_test: .4E}")