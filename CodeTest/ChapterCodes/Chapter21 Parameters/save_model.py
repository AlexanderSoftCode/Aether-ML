import numpy as np
from CodeTest.Model.modelfinal import *
#Load dataset

data = np.load("CodeTest/fashion_mnist_train.npz")
X, y = data["X"], data["y"]

data = np.load("CodeTest/fashion_mnist_test.npz")
X_test, y_test = data["X"], data["y"]

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

#Scale and reshape
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model()

model.add(Layer_Dense(X.shape[1], 256))
model.add(ReLU())
model.add(Layer_Dense(256, 256))
model.add(ReLU())
model.add(Layer_Dense(256, 10))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer= Optimizer_Adam(decay = 1e-4),
    accuracy= Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data= (X_test, y_test),
            epochs = 10, batch_size = 128, print_every = 500)


model.save('fashion_mnist.model')
#Once we save the model we will do a new file that will load the model. 