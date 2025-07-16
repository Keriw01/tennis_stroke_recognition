import os

import cv2
import numpy as np
import pandas as pd
from mediapipe.python.solutions.pose import Pose
from rich.console import Console
from tqdm import tqdm

console = Console()

VIDEOS_DIR = "data/processed_videos_30fps/"

pose = Pose(
    static_image_mode=False,  # If set to false, the solution treats the input images as a video stream.
    model_complexity=2,  # Complexity of the pose landmark model: 0, 1 or 2. Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1.
    min_detection_confidence=0.5,  # Minimum confidence value ([0.0, 1.0]) from the person-detection model for the detection to be considered successful.
)

annotations_df = pd.read_csv("data/annotations_csv/combined_annotations.csv")

sequences = []
labels = []

total_rows = len(annotations_df)
console.print(
    f"\n[bold cyan]>>> Starting processing {total_rows} annotations... <<<[/bold cyan]\n"
)

for index, row in tqdm(
    annotations_df.iterrows(), total=total_rows, desc="Annotation analysis"
):
    video_filename = row["fileName"]
    if not video_filename.endswith(".mp4"):
        video_filename += ".mp4"

    start_frame = int(row["startFrame"])
    end_frame = int(row["endFrame"])
    label = row["label"]

    video_path = os.path.join(VIDEOS_DIR, video_filename)

    if not os.path.exists(video_path):
        console.print(
            f"[bold red]Missing video file:[/bold red] [yellow]{video_path}[/yellow]"
        )
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        console.print(
            f"[bold red]Cannot open video file:[/bold red] [yellow]{video_path}[/yellow]"
        )
        continue

    cap.set(
        cv2.CAP_PROP_POS_FRAMES, start_frame
    )  # Set start frame with `CAP_PROP_POS_FRAMES`

    current_frame_num = start_frame

    sequence_landmarks = []

    while current_frame_num <= end_frame:
        success, frame = cap.read()  # Reading one frame from video
        if not success:
            break

        image_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )  # Changing colours format because CV2 using BGR, but MediaPipe require RGB

        results = pose.process(image_rgb)

        # Checking if a person was detected in this frame
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark

            # Taking pose_world_landmarks coordinates (33 points, each with x, y, z, visibility), flatten them into one long list of 132 numbers
            frame_landmarks = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
            ).flatten()
            sequence_landmarks.append(frame_landmarks)

        current_frame_num += 1

    cap.release()

    # Adding the entire collected sequence to the main sequences list, and also label to the labels list
    if sequence_landmarks:
        sequences.append(np.array(sequence_landmarks))
        labels.append(label)

console.print(
    f"\n[bold green]Processing ended.\nProcessed {len(sequences)} sequences.[/bold green]\n"
)

console.print("[yellow]Saving data to a file...[/yellow]")

# Saving to .npz format
output_dir = "data/processed_features/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.savez_compressed(
    os.path.join(output_dir, "raw_sequences.npz"),
    sequences=np.array(
        sequences, dtype=object
    ),  # Converting a list to an array of objects
    labels=np.array(labels),
)

console.print(
    f"\n[bold green]Data was successfully saved to the file:[/bold green] [underline cyan]{output_dir}[/underline cyan]\n"
)
