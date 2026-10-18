import json
import os
import shutil
import subprocess
import joblib
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ML_MODEL_PATH = "modelo_har_movil.pkl"
CNN_MODEL_PATH = "modelo_cnn_har.keras"
CNN_MEAN_PATH = "mean_cnn.npy"
CNN_STD_PATH = "std_cnn.npy"
CNN_CLASSES_PATH = "cnn_classes.json"
DATA_REPO = "_uci341"
DATA_URL = "https://github.com/mohalim/UCI_00341_Dataset.git"

ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

WINDOW = 128
STRIDE = 64

def stats_1d(x):
    x = np.asarray(x, dtype=float)
    return [
        np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x),
        np.percentile(x, 25), np.percentile(x, 75),
        np.sqrt(np.mean(x ** 2)), np.mean(np.abs(np.diff(x))),
        np.percentile(x, 75) - np.percentile(x, 25)
    ]

def extract_features(acc, gyro):
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    channels = [
        acc[:, 0], acc[:, 1], acc[:, 2],
        gyro[:, 0], gyro[:, 1], gyro[:, 2],
        acc_mag, gyro_mag
    ]
    feats = []
    for ch in channels:
        feats.extend(stats_1d(ch))
    return np.asarray(feats, dtype=np.float32)

def find_raw_dir(root):
    for current, dirs, files in os.walk(root):
        if "labels.txt" in files:
            has_acc = any(name.startswith("acc_exp") for name in files)
            has_gyro = any(name.startswith("gyro_exp") for name in files)
            if has_acc and has_gyro:
                return current
    raise FileNotFoundError("No se encontró la carpeta RawData.")

def load_signal(raw_dir, exp, user, prefix):
    path = os.path.join(
        raw_dir,
        f"{prefix}_exp{int(exp):02d}_user{int(user):02d}.txt"
    )
    return np.loadtxt(path)

if os.path.exists(DATA_REPO):
    shutil.rmtree(DATA_REPO)

print("Descargando dataset UCI...")
subprocess.run(
    ["git", "clone", "--depth", "1", DATA_URL, DATA_REPO],
    check=True
)

raw_dir = find_raw_dir(DATA_REPO)
labels = np.loadtxt(os.path.join(raw_dir, "labels.txt"), dtype=int)
if labels.ndim == 1:
    labels = labels.reshape(1, -1)

X_feat, X_seq, y, groups = [], [], [], []
cache = {}

print("Creando ventanas de sensores...")
for exp, user, activity, start, end in labels:
    activity = int(activity)
    if activity not in ACTIVITY_MAP:
        continue

    key = (int(exp), int(user))
    if key not in cache:
        cache[key] = (
            load_signal(raw_dir, key[0], key[1], "acc"),
            load_signal(raw_dir, key[0], key[1], "gyro")
        )

    acc, gyro = cache[key]
    start = max(0, int(start))
    end = min(len(acc), len(gyro), int(end))

    for s in range(start, end - WINDOW + 1, STRIDE):
        e = s + WINDOW
        acc_w = acc[s:e].astype(np.float32)
        gyro_w = gyro[s:e].astype(np.float32)

        X_feat.append(extract_features(acc_w, gyro_w))
        X_seq.append(np.column_stack([acc_w, gyro_w]))
        y.append(ACTIVITY_MAP[activity])
        groups.append(int(user))

X_feat = np.asarray(X_feat, dtype=np.float32)
X_seq = np.asarray(X_seq, dtype=np.float32)
y = np.asarray(y)
groups = np.asarray(groups)

print("Ventanas:", len(y))
print("ML:", X_feat.shape, "| CNN:", X_seq.shape)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(splitter.split(X_feat, y, groups=groups))

Xf_train, Xf_test = X_feat[train_idx], X_feat[test_idx]
Xs_train, Xs_test = X_seq[train_idx], X_seq[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ---------------- MACHINE LEARNING ----------------
scaler = StandardScaler()
Xf_train_scaled = scaler.fit_transform(Xf_train)
Xf_test_scaled = scaler.transform(Xf_test)

knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(Xf_train_scaled, y_train)

tree = DecisionTreeClassifier(
    max_depth=14,
    min_samples_leaf=4,
    random_state=42
)
tree.fit(Xf_train, y_train)

acc_knn = accuracy_score(y_test, knn.predict(Xf_test_scaled))
acc_tree = accuracy_score(y_test, tree.predict(Xf_test))

bundle = {
    "knn": knn,
    "tree": tree,
    "scaler": scaler,
    "window": WINDOW,
    "sampling_hz": 50,
    "activity_map": ACTIVITY_MAP,
    "accuracy_knn": float(acc_knn),
    "accuracy_tree": float(acc_tree),
}
joblib.dump(bundle, ML_MODEL_PATH, compress=3)

print(f"Accuracy KNN: {acc_knn:.4f}")
print(f"Accuracy Árbol: {acc_tree:.4f}")

# ---------------- DEEP LEARNING CNN 1D ----------------
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_test_int = le.transform(y_test)
classes = [str(c) for c in le.classes_]

mean_ch = Xs_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
std_ch = (Xs_train.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)

Xs_train_norm = (Xs_train - mean_ch) / std_ch
Xs_test_norm = (Xs_test - mean_ch) / std_ch

tf.keras.utils.set_random_seed(42)

cnn = keras.Sequential([
    layers.Input(shape=(WINDOW, 6)),
    layers.Conv1D(64, 5, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation="relu", padding="same"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.35),
    layers.Dense(len(classes), activation="softmax")
])

cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )
]

print("Entrenando CNN 1D...")
cnn.fit(
    Xs_train_norm,
    y_train_int,
    validation_split=0.20,
    epochs=25,
    batch_size=64,
    callbacks=callbacks,
    verbose=2
)

_, acc_cnn = cnn.evaluate(Xs_test_norm, y_test_int, verbose=0)

cnn.save(CNN_MODEL_PATH)
np.save(CNN_MEAN_PATH, mean_ch)
np.save(CNN_STD_PATH, std_ch)

with open(CNN_CLASSES_PATH, "w", encoding="utf-8") as f:
    json.dump(classes, f, ensure_ascii=False)

print(f"Accuracy CNN 1D: {acc_cnn:.4f}")
print("Modelos ML + DL generados correctamente.")

shutil.rmtree(DATA_REPO, ignore_errors=True)
