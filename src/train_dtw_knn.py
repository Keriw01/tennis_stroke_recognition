import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fastdtw import fastdtw
from rich.console import Console
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

console = Console()
console.print(
    "\n[bold cyan]>>> STARTING DTW + k-NN MODEL TRAINING AND EVALUATION <<<\n"
)


# * --- Step 1: Loading and Preparing Data ---
console.print("[bold]Step 1: Loading normalized data...[/bold]")

NORMALIZED_DATA_PATH = "data/processed_features/1_normalized_sequences.npz"

try:
    data = np.load(NORMALIZED_DATA_PATH, allow_pickle=True)
    sequences = data["sequences"]
    labels = data["labels"]
    console.print(f"[green]Successfully loaded {len(sequences)} sequences.[/green]")
except FileNotFoundError:
    console.print(
        f"[bold red]ERROR: File '{NORMALIZED_DATA_PATH}' not found. Run the normalization script first.[/bold red]"
    )
    exit()


# ? Optional: Aggregate labels to 5 main classes
# console.print("[yellow]Aggregating labels to main classes (e.g., 'FH', 'BH')...[/yellow]")
# labels = [label.split(" ")[0] for label in labels]


# * --- Step 2: Preparing Labels and Splitting Data ---
console.print("\n[bold]Step 2: Preparing labels and splitting data...[/bold]")

# Changing text labels to integer representation
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(labels)

console.print("[underline]Mapping labels to numbers:[/underline]")
for i, class_name in enumerate(label_encoder.classes_):
    console.print(f"  [magenta]{i}[/magenta]: {class_name}")

# Instead of splitting the data itself, we create a list of indices [0, 1, 2, ...] and split it
indices = np.arange(len(sequences))

# `stratify=y_numeric` ensures that the class proportions are the same in the training and test sets
X_train_indices, X_test_indices, y_train, y_test = train_test_split(
    indices, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
)

# Create the actual datasets based on the split indices
X_train = [sequences[i] for i in X_train_indices]
X_test = [sequences[i] for i in X_test_indices]

console.print(f"Training set size: [bold]{len(X_train)}[/bold] samples")
console.print(f"Test set size: [bold]{len(X_test)}[/bold] samples")


# * --- Step 3: Calculating the DTW Distance Matrix ---
console.print("\n[bold]Step 3: Calculating DTW distance matrix...[/bold]")


def dtw_distance(s1, s2):
    """Calculates the DTW distance between two sequences."""
    distance, _ = fastdtw(s1, s2, dist=euclidean)
    return distance


n_test = len(X_test)
n_train = len(X_train)

# This matrix will store the distances of each test sample from each training sample
distance_matrix = np.zeros((n_test, n_train))

for i in tqdm(range(n_test), desc="Calculating test-train distances"):
    for j in range(n_train):
        distance_matrix[i, j] = dtw_distance(X_test[i], X_train[j])

console.print("[green]Test-to-train distance matrix calculated.[/green]")


# * --- Step 4: Training and Predicting with k-NN Model ---
console.print(
    "\n[bold]Step 4: Training and predicting with k-NN (metric='precomputed')...[/bold]"
)

# Initialize the k-NN classifier `metric='precomputed'` means it will get a ready-made distance matrix
knn_clf = KNeighborsClassifier(n_neighbors=5, metric="precomputed")

# For the 'precomputed' metric, the .fit() method requires a distance matrix between the training samples themselves.
train_distance_matrix = np.zeros((n_train, n_train))
for i in tqdm(range(n_train), desc="Calculating train-train distances for .fit()"):
    for j in range(i, n_train):
        dist = dtw_distance(X_train[i], X_train[j])
        train_distance_matrix[i, j] = dist
        train_distance_matrix[j, i] = dist

# Training on the distance matrix of the training set
knn_clf.fit(train_distance_matrix, y_train)
console.print("[green]Model training complete.[/green]")

# Prediction is based on the distance matrix between the test and training sets
y_pred = knn_clf.predict(distance_matrix)
console.print("[green]Prediction complete.[/green]")


# * --- Step 5: Evaluating and Saving Results ---
console.print("\n[bold]Step 5: Evaluating and saving results...[/bold]")


# Overall percentage of correct classifications
accuracy = accuracy_score(y_test, y_pred)

# Detailed table with precision, recall, f1-score metrics for each class separately
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# A matrix showing how many times a given class was confused with another
conf_matrix = confusion_matrix(y_test, y_pred)

console.print(f"\n[bold]Accuracy: [cyan]{accuracy:.4f}[/cyan] ({accuracy * 100:.2f}%)")
console.print("\n[bold]Classification Report:[/bold]")
console.print(report)

# Create a unique folders for each run's results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
accuracy_str = f"acc_{accuracy:.4f}".replace(".", "_")
results_folder_name = f"{timestamp}_{accuracy_str}"
RESULTS_DIR = os.path.join("data/dtw_knn_results", results_folder_name)

os.makedirs(RESULTS_DIR, exist_ok=True)
console.print(f"\n[yellow]Results will be saved in: {RESULTS_DIR}[/yellow]")

# Save the classification report to a text file
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
    f.write("Classification Report:\n")
    f.write(report)
console.print("[green]Classification report saved.[/green]")

# Create and save the confusion matrix plot
plt.figure(figsize=(12, 10))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.title(f"Macierz Pomyłek - DTW+kNN (k=5)\nDokładność: {accuracy * 100:.2f}%")
plt.ylabel("Prawdziwa etykieta (True Label)")
plt.xlabel("Przewidziana etykieta (Predicted Label)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
console.print("[green]Confusion matrix plot saved.[/green]")

console.print("\n[bold green]>>> FINISHED <<<[/bold green]\n")
plt.show()
