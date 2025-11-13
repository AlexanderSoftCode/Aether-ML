import cupy as cp
import time 

from CNN.Model.CNN_model_cupy import *
print("Cupy version")
#X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
data = cp.load("CodeTest/fashion_mnist_train_cupy.npz")
X, y = data["X"], data["y"]

data = cp.load("CodeTest/fashion_mnist_test_cupy.npz")
X_test, y_test = data["X"], data["y"]

y = y.astype(cp.int32)
y_test = y_test.astype(cp.int32)


X = X[..., cp.newaxis]
X_test = X_test[..., cp.newaxis]
#Now we need to shuffle the batches

keys = cp.array(range(X.shape[0]))
cp.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = (X.astype(cp.float32) - 127.5) / 127.5
X_test = (X_test.astype(cp.float32) - 127.5) / 127.5
model = Model()

#Add the first convolutional block
model.add(Conv_Layer(input_shape = (28, 28, 1), num_filters= 16, filter_size= (3, 3), strides= (1, 1),
                     padding= "same"))
model.add(Batch_Norm(epsilon = 1e-5, momentum= 0.9))
model.add(Leaky_ReLU(alpha = 0.01))
model.add(Pooling(filter_size= (2,2), strides = (2,2), padding = "valid", pooling_type= "max"))

#Add the second convolutional block
model.add(Conv_Layer(input_shape = (14, 14, 16), num_filters= 32, filter_size= (3, 3), strides= (1, 1),
                     padding= "same"))
model.add(Batch_Norm(epsilon = 1e-5, momentum= 0.9))
model.add(Leaky_ReLU(alpha = 0.01))
model.add(Layer_Dropout_Spatial(rate = 0.1))
model.add(Pooling(filter_size= (2,2), strides = (2,2), padding = "valid", pooling_type= "max"))

#Add a final convolutional block
model.add(Conv_Layer(input_shape= (7, 7, 32), num_filters = 64, filter_size= (3, 3), strides = (1, 1),
                     padding = "same"))
model.add(Batch_Norm(epsilon = 1e-5, momentum= 0.9))
model.add(ReLU())
model.add(Layer_Dropout_Spatial(rate = 0.15))
model.add(Pooling(filter_size= (2, 2), strides = (2, 2), padding = "valid", pooling_type = "average"))

#Flatten and use dense layers
model.add((Flatten()))
model.add(Layer_Dense(3 * 3 * 64, n_neurons= 256, weight_regularizer_l2= 5e-5, bias_regularizer_l2= 5e-5))
model.add(ReLU())
model.add(Layer_Dropout(rate=0.025))
model.add(Layer_Dense(256, 10))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer = Optimizer_Adam(learning_rate = 0.001, decay = 2e-5),
    accuracy = Accuracy_Categorical()
)

model.finalize()


start = time.time() 
model.train(X, y, validation_data = (X_test, y_test),
            epochs = 30, batch_size = 256, print_every = 100)
end = time.time()

_ = model.forward_debug(X[:1])  # single image
model.backward_debug(X[:1], y[:1])

model.save(path = "CNN/Model/model2")

print("training time, ", end - start)
#556 seconds if we used 20 epochs and 256 batch size, I believe accuracy was around 87%. 
#720.3932731151581 seconds if we used 30 epochs and 256 batch size, trainin gaccuracy was .887
#etsting accuracy was .882 