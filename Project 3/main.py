import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on the Earth (specified in decimal degrees)."""
    R = 6371000  # Radius of the Earth in meters
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def interpolate_height(locations, removed, filename, distance_interval=250, output_filename="interpolated_data.csv"):
    """Reads data, calculates distances, interpolates height, plots, and saves to CSV."""

    data = np.loadtxt(filename)
    latitudes = data[:, 0]
    longitudes = data[:, 1]
    heights = data[:, 2]

    distances = [0]
    total_distance = 0

    for i in range(1, len(latitudes)):
        dist = calculate_distance(latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i])
        total_distance += dist
        distances.append(total_distance)

    distances = np.array(distances)

    f = interp1d(distances, heights, kind='linear')
    max_distance = distances[-1]
    interpolated_distances = np.arange(0, max_distance, distance_interval)
    interpolated_heights = f(interpolated_distances) - removed

    # Save to CSV
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance (m)", "Height (m)"])  # Header row
        for dist, height in zip(interpolated_distances, interpolated_heights):
            writer.writerow([dist, height])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(interpolated_distances, interpolated_heights)

    plt.xlabel("Distance from Mediterranean Sea (m)")
    plt.ylabel("Height over Sea Level (m)")
    plt.title("Height vs. Distance from Mediterranean Sea")
    plt.grid(True)
    plt.scatter(locations * interpolated_distances, locations * interpolated_heights + 6, c='r', marker='v')
    plt.scatter(locations * interpolated_distances, locations * interpolated_heights + 10, c='r', marker='|')
    plt.show()

interpolate_height(0, 0, "channel_data.txt", output_filename="interpolated_height.csv")