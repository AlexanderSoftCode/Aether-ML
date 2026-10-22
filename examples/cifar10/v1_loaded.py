from pathlib import Path

import cupy as cp

import aether as ae

TRAIN_DIR = Path("data") / "cifar-10" / "cifar-10_train.npz"
TEST_DIR  = Path("data") / "cifar-10" / "cifar-10_test.npz"

# Using cp.load saves it as a cupy array
with cp.load(TRAIN_DIR, allow_pickle=False) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]

with cp.load(TEST_DIR, allow_pickle=False) as data:
    X_test = data["X_test"]
    y_test = data["y_test"]

model = ae.Model.load(filepath="saved_models/cifar10_3block_cnn.aether")

model.evaluate(X=X_test, y=y_test, batch_size=128)