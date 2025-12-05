import cupy as cp
import time
from CNN.Model.CNN_model_cupy import *

print("Cupy version")

with cp.load("data/cifar10_train.npz") as data:
    X, y = data["X"], data["y"]
    
with cp.load("data/cifar10_test.npz") as data:
    X_test, y_test = data["X"], data["y"]

X = X.reshape(X.shape[0], -1)   # (num_samples, 32*32*3 = 3072)
X_test = X_test.reshape(X_test.shape[0], -1)

model = Model()

model.add(Layer_Dense(X.shape[1], 256, weight_regularizer_l2= 5e-5, bias_regularizer_l2= 5e-5))
model.add(ReLU())
model.add(Layer_Dense(256, 256, weight_regularizer_l2= 5e-5, bias_regularizer_l2= 5e-5))
model.add(ReLU())
model.add(Layer_Dense(256, 256, weight_regularizer_l2= 5e-5, bias_regularizer_l2= 5e-5))
model.add(ReLU())
model.add(Layer_Dense(256, 32))
model.add(ReLU())
model.add(Layer_Dense(32, 10))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer= Optimizer_Adam(decay = 1e-4),
    accuracy= Accuracy_Categorical()
)

model.finalize()

start = time.time()
model.train(X, y, validation_data = (X_test, y_test),
            epochs = 10, batch_size = 128, print_every = 100)
end = time.time()
model.save_parameters("CNN/Model/saved_models/cifar2")

print("training time, ", end - start)
