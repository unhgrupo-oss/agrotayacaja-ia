import os
import shutil
import subprocess
import joblib
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

MODEL_PATH = "modelo_har_movil.pkl"
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
        np.percentile(x,25), np.percentile(x,75),
        np.sqrt(np.mean(x**2)), np.mean(np.abs(np.diff(x))),
        np.percentile(x,75)-np.percentile(x,25)
    ]

def extract_features(acc, gyro):
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    channels = [
        acc[:,0], acc[:,1], acc[:,2],
        gyro[:,0], gyro[:,1], gyro[:,2],
        acc_mag, gyro_mag
    ]
    feats = []
    for ch in channels:
        feats.extend(stats_1d(ch))
    return np.asarray(feats, dtype=float)

def find_raw_dir(root):
    for current, dirs, files in os.walk(root):
        if "labels.txt" in files:
            has_acc = any(name.startswith("acc_exp") for name in files)
            has_gyro = any(name.startswith("gyro_exp") for name in files)
            if has_acc and has_gyro:
                return current
    raise FileNotFoundError("No se encontró la carpeta RawData.")

def load_signal(raw_dir, exp, user, prefix):
    path = os.path.join(raw_dir, f"{prefix}_exp{int(exp):02d}_user{int(user):02d}.txt")
    return np.loadtxt(path)

if os.path.exists(MODEL_PATH):
    print("Modelo ya generado; se omite el entrenamiento.")
    raise SystemExit(0)

if os.path.exists(DATA_REPO):
    shutil.rmtree(DATA_REPO)

print("Descargando dataset de respaldo público...")
subprocess.run(
    ["git", "clone", "--depth", "1", DATA_URL, DATA_REPO],
    check=True
)

raw_dir = find_raw_dir(DATA_REPO)
labels = np.loadtxt(os.path.join(raw_dir, "labels.txt"), dtype=int)
if labels.ndim == 1:
    labels = labels.reshape(1, -1)

X_rows, y_rows, groups = [], [], []
cache = {}

print("Extrayendo ventanas de sensores...")
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
        X_rows.append(extract_features(acc[s:e], gyro[s:e]))
        y_rows.append(ACTIVITY_MAP[activity])
        groups.append(int(user))

X = np.vstack(X_rows)
y = np.asarray(y_rows)
groups = np.asarray(groups)

print("Ventanas:", len(X), "| Características:", X.shape[1])

splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(X_train_scaled, y_train)

tree = DecisionTreeClassifier(max_depth=14, min_samples_leaf=4, random_state=42)
tree.fit(X_train, y_train)

acc_knn = accuracy_score(y_test, knn.predict(X_test_scaled))
acc_tree = accuracy_score(y_test, tree.predict(X_test))

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

joblib.dump(bundle, MODEL_PATH, compress=3)
print(f"Modelo generado: {MODEL_PATH}")
print(f"Accuracy KNN: {acc_knn:.4f}")
print(f"Accuracy Árbol: {acc_tree:.4f}")

shutil.rmtree(DATA_REPO, ignore_errors=True)
print("Dataset temporal eliminado.")
