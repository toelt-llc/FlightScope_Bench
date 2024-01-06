#https://www.kaggle.com/code/kmader/segmenting-buildings-in-satellite-images

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

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
    


def plot_curves(directory, color):
    # Get a list of CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # Calculate the number of subplots needed based on the number of CSV files
    num_files = len(csv_files)
    num_rows = (num_files // 3) + (1 if num_files % 3 != 0 else 0)
    num_cols = min(num_files, 3)

    # Create subplot grid
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axs = axs.flatten()

    # Iterate over CSV files and plot curves
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

    # Customize the plot
    plt.tight_layout()
    plt.show()