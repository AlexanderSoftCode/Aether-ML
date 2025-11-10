import numpy as np
import time
from Model.model import *
#typically we'd do 
#X, y, X_test, y_test = create_mnist_dataset("FILENAME")
data = np.load("fashion_mnist_train.npz")
X, y = data["X"], data["y"]

data = np.load("fashion_mnist_test.npz")
X_test, y_test = data["X"], data["y"]

#Now lets shuffle
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

#Lets scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5 ) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5 ) / 127.5

model = Model()
model.add(Layer_Dense(X.shape[1], 256, weight_regularizer_l2= 5e-4,
                      bias_regularizer_l2= 5e-4))
model.add(ReLU())
model.add(Layer_Dense(256, 256, weight_regularizer_l2= 5e-4,
                      bias_regularizer_l2= 5e-4))
model.add(ReLU())
model.add(Layer_Dense(256, 10, weight_regularizer_l2= 5e-4,
                      bias_regularizer_l2=5e-4))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer = Optimizer_Adam(), 
    accuracy= Accuracy_Categorical()
)

model.finalize()

start = time.time()
model.train(X = X, y = y, validation_data = (X_test, y_test),
            epochs = 10, batch_size = 128, print_every = 100)

print("Model training output removed")
model.evaluate(X_test, y_test)
end = time.time()

print("training time: ", end - start)
