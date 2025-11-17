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
model.add(Conv_Layer(input_shape = (28, 28, 1), num_filters= 32, filter_size= (3, 3), strides= (1, 1),
                     padding= "same"))
model.add(Batch_Norm(epsilon = 1e-5, momentum= 0.9))
model.add(ReLU())
model.add(Pooling(filter_size= (2,2), strides = (2,2), padding = "valid", pooling_type= "max"))

#Add the second convolutional block
model.add(Conv_Layer(input_shape = (14, 14, 32), num_filters= 64, filter_size= (3, 3), strides= (1, 1),
                     padding= "same"))
model.add(Batch_Norm(epsilon = 1e-5, momentum= 0.9))
model.add(ReLU())
model.add(Layer_Dropout_Spatial(rate = 0.075))
model.add(Pooling(filter_size= (2,2), strides = (2,2), padding = "valid", pooling_type= "max"))

#Add a final convolutional block
model.add(Conv_Layer(input_shape= (7, 7, 64), num_filters = 128, filter_size= (3, 3), strides = (1, 1),
                     padding = "same"))
model.add(Batch_Norm(epsilon = 1e-5, momentum= 0.9))
model.add(ReLU())
model.add(Layer_Dropout_Spatial(rate = 0.1))
model.add(Pooling(filter_size= (2, 2), strides = (2, 2), padding = "valid", pooling_type = "max"))

#Flatten and use dense layers
model.add(Flatten())
model.add(Layer_Dense(3 * 3 * 128, n_neurons= 256, weight_regularizer_l2= 2e-5, bias_regularizer_l2= 2e-5))
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
#model.backward_debug(X[:1], y[:1])
model.train(X, y, validation_data = (X_test, y_test),
            epochs = 8, batch_size = 128, print_every = 100)
end = time.time()


model.save_parameters(path = "CNN/Model/model5")

print("training time, ", end - start)

#model 3 holds .947 training accuracy and .161 loss
#where validation was at .916 training accuracy and .247 loss. only change was .05 dropout on last layer

#model 2 record is .946 accuracy, 1.53 loss, with validation acc .919 loss .239 
#model 4 will hold the same settings as model 2 except
#include new loss calculation as well as label smoothing = 0.01
#model 4 had .951 training accuracy .221 loss with validation acc .919 loss .305
#model 5 had .952 training accuracy .281 loss with validation acc .911 loss .258
#essentially, I fixed the issue where the model was still using label smoothing 