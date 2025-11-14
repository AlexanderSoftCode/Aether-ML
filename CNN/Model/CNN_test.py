import numpy as np
import time 

from CNN.Model.CNN_model import *

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
model.add(Conv_Layer(input_shape = (28, 28, 1), num_filters= 8, filter_size= (3, 3), strides= (1, 1),
                     padding= "same"))
model.add(Leaky_ReLU(alpha = 0.05))
model.add(Pooling(filter_size= (2,2), strides = (2,2), padding = "valid", pooling_type= "max"))

#Add the second convolutional block
model.add(Conv_Layer(input_shape = (14, 14, 8), num_filters= 16, filter_size= (3, 3), strides= (1, 1),
                     padding= "same"))
model.add(Leaky_ReLU(alpha = 0.05))
model.add(Layer_Dropout_Spatial(rate = 0.1))
model.add(Pooling(filter_size= (2,2), strides = (2,2), padding = "valid", pooling_type= "max"))

#Add a final convolutional block
model.add(Conv_Layer(input_shape= (7, 7, 16), num_filters = 32, filter_size= (3, 3), strides = (1, 1),
                     padding = "same"))
model.add(ReLU())
model.add(Layer_Dropout_Spatial(rate = 0.15))
model.add(Pooling(filter_size= (2, 2), strides = (2, 2), padding = "valid", pooling_type = "average"))

#Flatten and use dense layers
model.add((Flatten()))
model.add(Layer_Dense(3 * 3 * 32, n_neurons= 256, weight_regularizer_l2= 5e-5, bias_regularizer_l2= 5e-5))
model.add(ReLU())
model.add(Layer_Dense(256, 10))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer = Optimizer_Adam(learning_rate = 0.001, decay = 1e-5),
    accuracy = Accuracy_Categorical()
)

model.finalize()

start = time.time() 
model.train(X, y, validation_data = (X_test, y_test),
            epochs = 10, batch_size = 128, print_every = 100)
end = time.time()

model.save(path = "CNN/model3")

print("training time, ", end - start)