"""
The setup script found in the README.md pop up
"""
import aether as ae
import numpy as np
import cupy as cp

from pathlib import Path

TRAIN_DIR = Path("data") / "cifar-10" / "cifar-10_train.npz"
TEST_DIR  = Path("data") / "cifar-10" / "cifar-10_test.npz"

# Using cp.load saves it as a cupy array
with cp.load(TRAIN_DIR, allow_pickle=False) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]

with cp.load(TEST_DIR, allow_pickle=False) as data:
    X_test = data["X_test"]
    y_test = data["y_test"]

TARGET_DEVICE="cupy"
feature_pipeline = ae.Compose([
    ae.ToTensor(dtype='float32', target_device=TARGET_DEVICE),
    ae.Rescale(factor=1.0 / 255.0),
    ae.StandardScaler()
])
# Setup
model = ae.Model()
model.manual_seed(seed=42) #applies for all compute and PRNG based layers

model.add(ae.Conv2d(3, 32, (3, 3), (1, 1), padding="same"))
model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
model.add(ae.ReLU())
model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))

model.add(ae.Conv2d(32, 64, (3, 3), (1, 1), padding="same"))
model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
model.add(ae.LeakyReLU())
model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))
model.add(ae.SpatialDropout(rate=0.1))

model.add(ae.Conv2d(64, 128, (3, 3), (1, 1), padding="same"))
model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
model.add(ae.ReLU())
model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))
model.add(ae.SpatialDropout(rate=0.1))

model.add(ae.GlobalAvgPool())
model.add(ae.Dense(128, 128, l2=2e-5))
model.add(ae.Dropout(rate=0.075))

model.add(ae.Dense(128, 10))

model.configure(
    loss=ae.SoftmaxCategoricalCrossEntropy(label_smoothing=0.05),
    optimizer=ae.AdamW(lr=0.001, decay=1e-4, weight_decay=0.01),
    accuracy=ae.CategoricalAccuracy(),
    preprocessor=feature_pipeline
)
model.to(TARGET_DEVICE)
model.finalize(input_shape=X_train.shape[1:])

model.train(X=X_train, 
    y=y_train, 
    epochs=25, 
    batch_size=128, 
    shuffle=True,
    print_every=100,
    validation_data=(X_test, y_test)
)