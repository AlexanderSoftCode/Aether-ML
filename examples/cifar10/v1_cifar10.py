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

TARGET_DEVICE = "cupy"
feature_pipeline = ae.Compose([
    ae.ToTensor(dtype='float32', target_device=TARGET_DEVICE),
    ae.Rescale(factor=1.0 / 255.0),
    ae.StandardScaler()
])

"""# 3. Convert target labels
y_train_tensor, y_test_tensor = ae.to_tensor(
    y_train, y_test, target_device=TARGET_DEVICE, preserve_integers=True
)
"""
train_x_demo = X_train[:8192*4]
train_y_demo = y_train[:8192*4]
val_x_demo   = X_test[:1000]
val_y_demo   = y_test[:1000]
SEED = 42
model = ae.Model()
model.manual_seed(seed=42)

# --- Block 1: Low-level edges & colors (32x32 -> 16x16) ---
model.add(ae.Conv2d(3, 32, (3, 3), (1, 1), padding="same"))
model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
model.add(ae.ReLU())
model.add(ae.Conv2d(32, 32, (3, 3), (1, 1), padding="same"))
model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
model.add(ae.ReLU())
model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))
model.add(ae.SpatialDropout(rate=0.1, seed=42))

# --- Block 2: Intermediate textures & patterns (16x16 -> 8x8) ---
model.add(ae.Conv2d(32, 64, (3, 3), (1, 1), padding="same", l2=1e-5))
model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
model.add(ae.ReLU())
model.add(ae.Conv2d(64, 64, (3, 3), (1, 1), padding="same"))
model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
model.add(ae.ReLU())
model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))
model.add(ae.SpatialDropout(rate=0.15, seed=42))

# --- Block 3: High-level class semantics (8x8 -> 4x4) ---
model.add(ae.Conv2d(64, 128, (3, 3), (1, 1), padding="same"))
model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
model.add(ae.ReLU())
model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))
model.add(ae.SpatialDropout(rate=0.2, seed=42))

# --- Head: Parameter-efficient classification ---
model.add(ae.GlobalAvgPool())
model.add(ae.Dense(128, 10))

# --- Compilation & Training ---
model.configure(
    loss=ae.SoftmaxCategoricalCrossEntropy(label_smoothing=0.05),
    optimizer=ae.AdamW(lr=0.001, decay=1e-4, weight_decay=0.01),
    accuracy=ae.CategoricalAccuracy(),
    preprocessor=feature_pipeline
)

model.to('cupy')
model.set_precision(compute_dtype="float16")
model.finalize(input_shape=X_train.shape[1:])

model.train(
    X=train_x_demo,
    y=train_y_demo,
    epochs=4,
    batch_size=256,
    shuffle=True,
    print_every=100,
    verbose=1,
    validation_data=(val_x_demo, val_y_demo)
)

model.save(filepath="saved_models/cifar10_3block_cnn.aether")
