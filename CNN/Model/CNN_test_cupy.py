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


#X = X[..., cp.newaxis]
#X_test = X_test[..., cp.newaxis]
#Now we need to shuffle the batches
start = time.time() 

keys = cp.array(range(X.shape[0]))
cp.random.shuffle(keys)
X = X[keys]
y = y[keys]

#Scale and reshape
X = (X.reshape(X.shape[0], -1).astype(cp.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(cp.float32) - 127.5) / 127.5

model = Model()

model.add(Layer_Dense(X.shape[1], 256, weight_regularizer_l2= 5e-5, bias_regularizer_l2= 5e-5))
model.add(ReLU())
model.add(Layer_Dense(256, 256, weight_regularizer_l2= 5e-5, bias_regularizer_l2= 5e-5))
model.add(ReLU())
model.add(Layer_Dense(256, 10, weight_regularizer_l2= 5e-5, bias_regularizer_l2= 5e-5))
model.add(SoftMax())

model.set(
    loss = Loss_CategoricalCrossEntropy(),
    optimizer= Optimizer_Adam(decay = 1e-4),
    accuracy= Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data = (X_test, y_test),
            epochs = 10, batch_size = 128, print_every = 100)

model.save(path = "CNN/Model/model1")


#_ = model.forward_debug(X[:1])  # single image
#model.backward_debug(X[:1], y[:1])


#model.save(path = "CNN/Model/model1")
end = time.time()
print("training time, ", end - start)