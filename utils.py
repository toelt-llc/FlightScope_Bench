import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def getBounds(geometry):
    try: 
        arr = np.array(geometry).T
        xmin = np.min(arr[0])
        ymin = np.min(arr[1])
        xmax = np.max(arr[0])
        ymax = np.max(arr[1])
        return (xmin, ymin, xmax, ymax)
    except:
        return np.nan

def getWidth(bounds):
    try: 
        (xmin, ymin, xmax, ymax) = bounds
        return np.abs(xmax - xmin)
    except:
        return np.nan

def getHeight(bounds):
    try: 
        (xmin, ymin, xmax, ymax) = bounds
        return np.abs(ymax - ymin)
    except:
        return np.nan
    


def plot_curves(directory, color='sienna', separate_subplots=True):
    # Get a list of CSV files in the directory and sort them
    csv_files = sorted([file for file in os.listdir(directory) if file.endswith('.csv')])

    if separate_subplots:
        # Calculate the number of subplots needed based on the number of CSV files
        num_files = len(csv_files)
        num_rows = (num_files // 2) + (1 if num_files % 2 != 0 else 0)
        num_cols = min(num_files, 2)

        # Create subplot grid
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        axs = axs.flatten()

        # Iterate over sorted CSV files and plot curves
        for i, csv_file in enumerate(csv_files):
            # Construct the full path to the CSV file
            csv_path = os.path.join(directory, csv_file)

            # Read CSV file into a DataFrame
            df = pd.read_csv(csv_path)

            # Extract relevant columns
            steps = df['Step']
            values = df['Value']

            # Plotting the curve with the specified color
            axs[i].plot(steps, values, color=color)
            axs[i].set_title(os.path.splitext(csv_file)[0])  # Use file name as title
            axs[i].grid(True)

        # Customize the plot
        plt.tight_layout()
        plt.show()
    else:
        # Create a single plot for all curves
        fig, ax = plt.subplots(figsize=(15, 10))

        # Iterate over CSV files and plot curves with different colors
        for i, csv_file in enumerate(csv_files):
            # Construct the full path to the CSV file
            csv_path = os.path.join(directory, csv_file)

            # Read CSV file into a DataFrame
            df = pd.read_csv(csv_path)

            # Extract relevant columns
            steps = df['Step']
            values = df['Value']

            # Plotting each curve with a different color
            ax.plot(steps, values, label=os.path.splitext(csv_file)[0], color=plt.cm.viridis(i / len(csv_files)))

        # Customize the plot
        plt.legend(fontsize=18)
        plt.xlabel("Epochs")
        plt.ylabel("Values")
        plt.ylim(0, 1)
        plt.xlim(0, 500)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True)
        plt.show()



def concatenate_videos(videos_folder, output_folder="./md_vizualiser"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of video files in the folder
    video_files = [f for f in os.listdir(videos_folder) if f.endswith(".mp4") and f.startswith("Barcelona_airport_")]

    # Sort the files to ensure consistent ordering
    video_files.sort()

    # Iterate through pairs of video files and concatenate them horizontally
    for i in range(0, len(video_files), 2):
        video1 = cv2.VideoCapture(os.path.join(videos_folder, video_files[i]))

        # Check if the video file was opened successfully
        if not video1.isOpened():
            print(f"Error: Could not open {video_files[i]}")
            continue

        video2 = cv2.VideoCapture(os.path.join(videos_folder, video_files[i + 1]))

        # Check if the video file was opened successfully
        if not video2.isOpened():
            print(f"Error: Could not open {video_files[i + 1]}")
            video1.release()
            continue

        # Get the video properties
        width = int(video1.get(3))
        height = int(video1.get(4))

        # Create an output video writer
        output_path = os.path.join(output_folder, f"concatenated_{i//2}.mp4")
        output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width * 2, height))

        while True:
            ret1, frame1 = video1.read()
            ret2, frame2 = video2.read()

            if not ret1 or not ret2:
                break

            # Concatenate frames horizontally
            concatenated_frame = cv2.hconcat([frame1, frame2])

            # Write the concatenated frame to the output video
            output_video.write(concatenated_frame)

        # Release video captures and writer
        video1.release()
        video2.release()
        output_video.release()

    print("Concatenation complete.")

# Example usage:
# concatenate_videos("path/to/your/videos/folder")
