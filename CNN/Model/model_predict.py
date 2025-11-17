import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
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

model.load_paramters(path = "CNN/Model/model5")
#.912 validation, .253 loss

# Get predictions for test set
predictions = model.predict(X_test)
predicted_classes = cp.argmax(predictions, axis=1)

# Find correctly predicted samples
correct_mask = predicted_classes == y_test
correct_indices = cp.where(correct_mask)[0]

# Fashion MNIST class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Find two samples from different classes
selected_indices = []
selected_classes = set()

for idx in correct_indices:
    class_label = int(y_test[idx])
    if class_label not in selected_classes:
        selected_indices.append(int(idx))
        selected_classes.add(class_label)
    if len(selected_indices) == 2:
        break

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for i, idx in enumerate(selected_indices):
    # Get image and convert back to original scale
    img = cp.asnumpy(X_test[idx, :, :, 0])
    img = (img * 127.5 + 127.5).astype(np.uint8)
    
    true_label = int(y_test[idx])
    pred_label = int(predicted_classes[idx])
    confidence = float(cp.max(predictions[idx]) * 100)
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'True: {class_names[true_label]}\n'
                      f'Predicted: {class_names[pred_label]}\n'
                      f'Confidence: {confidence:.1f}%',
                      fontsize=25)
    axes[i].axis('off')

plt.tight_layout()
plt.show()