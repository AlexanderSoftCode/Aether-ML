import numpy as np
import time
from CNN.models.CNN_model import *

print("Numpy version")
data = np.load("data/cifar10_clean.npz")
X, y = data["X"], data["y"]

N = X.shape[0]

rng = np.random.default_rng(seed = 42)
indices = rng.permutation(N)

X = X[indices] 
y = y[indices] 

split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Model()

#Add the first convolutional block
model.add(Conv_Layer(input_shape = (32, 32, 3), num_filters= 32, filter_size= (3, 3), strides= (1, 1),
                     padding= "same"))
model.add(Batch_Norm(epsilon = 1e-5, momentum= 0.9))
model.add(ReLU())
model.add(Pooling(filter_size= (2,2), strides = (2,2), padding = "valid", pooling_type= "max"))

#Add the second convolutional block
model.add(Conv_Layer(input_shape = (16, 16, 32), num_filters= 64, filter_size= (3, 3), strides= (1, 1),
                     padding= "same"))
model.add(Batch_Norm(epsilon = 1e-5, momentum= 0.9))
model.add(ReLU())
model.add(Layer_Dropout_Spatial(rate = 0.075))
model.add(Pooling(filter_size= (2,2), strides = (2,2), padding = "valid", pooling_type= "max"))

#Add a final convolutional block
model.add(Conv_Layer(input_shape= (8, 8, 64), num_filters = 128, filter_size= (3, 3), strides = (1, 1),
                     padding = "same"))
model.add(Batch_Norm(epsilon = 1e-5, momentum= 0.9))
model.add(ReLU())
model.add(Layer_Dropout_Spatial(rate = 0.1))
model.add(Pooling(filter_size= (2, 2), strides = (2, 2), padding = "valid", pooling_type = "max"))

#Flatten and use dense layers
model.add(Flatten())
model.add(Layer_Dense(4 * 4 * 128, n_neurons= 256, weight_regularizer_l2= 2e-5, bias_regularizer_l2= 2e-5))
model.add(Leaky_ReLU(alpha = 0.01))
model.add(Layer_Dropout(rate=0.075))
model.add(Layer_Dense(256, 10))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(label_smoothing= 0.01),
    optimizer = Optimizer_Adam(learning_rate = 0.001, decay = 5e-5),
    accuracy = Accuracy_Categorical()
)

model.finalize()

start = time.time()
model.backward_debug(X[:1], y[:1])
model.train(X, y, validation_data = (X_test, y_test),
            epochs = 1, batch_size = 128, print_every = 10)
end = time.time()

print("training time, ", end - start)