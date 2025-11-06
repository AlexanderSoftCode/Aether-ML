import numpy as np
import time 

from model_CNN import *
start = time.time() 
print("Numpy version")
#X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
data = np.load("CodeTest/fashion_mnist_train.npz")
X, y = data["X"], data["y"]

data = np.load("CodeTest/fashion_mnist_test.npz")
X_test, y_test = data["X"], data["y"]

X = X[..., np.newaxis]
X_test = X_test[..., np.newaxis]
#Now we need to shuffle the batches

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
model = Model()

#Add the first convolutional block
model.add(Conv_Layer(input_shape = (128, 28, 28, 1), num_filters= 8, filter_size= (3, 3), strides= (1, 1),
                     padding= "same"))
model.add(ReLU())
model.add(Pooling(filter_size= (2,2), strides = (2,2), padding = "valid", pooling_type= "max"))

#Add the second convolutional block
model.add(Conv_Layer(input_shape = (128, 14, 14, 8), num_filters= 16, filter_size= (3, 3), strides= (1, 1),
                     padding= "same"))
model.add(ReLU())
model.add(Pooling(filter_size= (2,2), strides = (2,2), padding = "valid", pooling_type= "max"))

#Flatten and use dense layers
model.add((Flatten()))
model.add(Layer_Dense(7 * 7 * 16, n_neurons= 256, weight_regularizer_l2= 5e-4, bias_regularizer_l2= 5e-4))
model.add(Layer_Dense(256, 10))
model.add(SoftMax())
'''
class Conv_Layer:
    def __init__(self, input_shape, num_filters = 1, filter_size = (3, 3), strides = (1, 1), padding = "same"):
    
class Pooling: 
    def __init__(self, filter_size = (2, 2), strides = (2, 2),
                  padding = "valid", pooling_type = "max"):
'''
model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer = Optimizer_Adam(decay = 5e-5),
    accuracy = Accuracy_Categorical()
)

model.finalize()
model.train(X, y, validation_data = (X_test, y_test),
            epochs = 5, batch_size = 128, print_every = 1000)