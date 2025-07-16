import numpy as np
from rich.console import Console
from tqdm import tqdm

console = Console()
console.print("\n[bold cyan]>>> Starting normalization... <<<[/bold cyan]\n")

NORMALIZED_DATA_PATH = "data/processed_features/1_normalized_sequences.npz"
JOINT_INDICES_TO_USE = [11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28]

normalized_sequences = []

data_path = "data/processed_features/raw_sequences.npz"
data = np.load(
    data_path, allow_pickle=True
)  # Loading an array of objects with different shapes

sequences = data["sequences"]
labels = data["labels"]

# Iteration after each individual stroke sequence (e.g. one forehand)
for seq in tqdm(sequences, desc="Normalization", unit="sequence"):
    normalized_seq = []

    # Single frame processing
    for frame_landmarks in seq:
        # `frame_landmarks` is a flattened list of 132 coordinates
        landmarks_reshaped = frame_landmarks.reshape((33, 4))

        # `selected_landmarks` will now be an array of shape (14, 4)
        selected_landmarks = landmarks_reshaped[JOINT_INDICES_TO_USE]

        # Selecting all rows but only the first 3 columns (x, y, z), then calculating the average value for each column separately
        # The result is `center_point` an array containing 3 values [average_x, average_y, average_z]
        center_point = np.mean(selected_landmarks[:, :3], axis=0)

        normalized_landmarks = selected_landmarks.copy()
        # From the [x, y, z] coordinates of each point, we subtract the coordinates of the center of mass
        # This effectively shifts the entire pose so that its new center is at (0, 0, 0)
        # *Information about where the player was standing on the court is removed
        # *Only information about their body position remains
        normalized_landmarks[:, :3] = normalized_landmarks[:, :3] - center_point

        normalized_seq.append(normalized_landmarks.flatten())

    normalized_sequences.append(np.array(normalized_seq))

console.print(
    "\n[bold green]>>> Normalization process completed successfully! <<<[/bold green]\n"
)


console.print("[yellow]Saving data to a file...[/yellow]")

np.savez_compressed(
    NORMALIZED_DATA_PATH,
    sequences=np.array(normalized_sequences, dtype=object),
    labels=labels,
)

console.print(
    f"\n[bold green]Data was successfully saved to the file:[/bold green] [underline cyan]{NORMALIZED_DATA_PATH}[/underline cyan]\n"
)
