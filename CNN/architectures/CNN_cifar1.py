import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
from CNN.models.CNN_model_cupy import *

print("Cupy version")

with cp.load("data/cifar10_train.npz") as data:
    X, y = data["X"], data["y"]
    
with cp.load("data/cifar10_test.npz") as data:
    X_test, y_test = data["X"], data["y"]


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

'''
start = time.time() 
#model.backward_debug(X[:1], y[:1])
model.train(X, y, validation_data = (X_test, y_test),
            epochs = 8, batch_size = 128, print_every = 100)
end = time.time()


model.save_parameters(path = "CNN/Model/saved_models/cifar1")

print("training time, ", end - start)
'''

model.load_parameters(path = "CNN/saved_models/cifar1")

model.evaluate(X_val = X_test, y_val = y_test, batch_size= 128)
#Valdiation, acc: 0.761, loss: 0.698
#This shows that the load feature works 100%

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Process predictions in smaller batches to avoid memory issues
batch_size = 100
predictions_list = []

print("Getting predictions in batches...")
for i in range(0, len(X_test), batch_size):
    batch_end = min(i + batch_size, len(X_test))
    batch_predictions = model.predict(X_test[i:batch_end])
    predictions_list.append(batch_predictions)
    
    # Free GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    
    if i % 1000 == 0:
        print(f"Processed {i}/{len(X_test)} samples")

# Concatenate all predictions
predictions = cp.concatenate(predictions_list, axis=0)
predicted_classes = cp.argmax(predictions, axis=1)

# Find correctly and incorrectly predicted samples for cat class (class 3)
cat_class = 3
correct_mask = (predicted_classes == y_test) & (y_test == cat_class)
incorrect_mask = (predicted_classes != y_test) & (y_test == cat_class)

correct_indices = cp.where(correct_mask)[0]
incorrect_indices = cp.where(incorrect_mask)[0]

print(f"Total correct cat predictions: {len(correct_indices)}")
print(f"Total incorrect cat predictions: {len(incorrect_indices)}")

# Select one correct and one incorrect
selected_indices = []
if len(correct_indices) > 4:
    selected_indices.append(int(correct_indices[4]))
if len(incorrect_indices) > 0:
    selected_indices.append(int(incorrect_indices[0]))

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

titles = ['Correct Prediction', 'Incorrect Prediction']
for i, idx in enumerate(selected_indices):
    # Get image and move to CPU
    img = cp.asnumpy(X_test[idx])
    
    # Images are already in [0, 255] range, just clip and convert to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    true_label = int(y_test[idx])
    pred_label = int(predicted_classes[idx])
    confidence = float(cp.max(predictions[idx]) * 100)
    
    axes[i].imshow(img)
    axes[i].set_title(f'{titles[i]}\n'
                      f'True: {class_names[true_label]}\n'
                      f'Predicted: {class_names[pred_label]}\n'
                      f'Confidence: {confidence:.1f}%',
                      fontsize=11)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Clean up
cp.get_default_memory_pool().free_all_blocks()