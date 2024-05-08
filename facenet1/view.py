# ChatGPT
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# Load the CSV file into a numpy array
# Assuming the CSV has one column with indices
csv_path = 'pred/predicted_volume.csv'  # Replace with your CSV file path
# Use genfromtxt to read the CSV
data = np.genfromtxt(csv_path, dtype=np.uint8, delimiter=',', skip_header=0)

# Ensure data is a 1D array
if len(data.shape) == 2 and data.shape[1] == 1:
    data = data.flatten()  # Convert to 1D array if needed

indices = data.tolist()  # Convert to a list

# Convert indices to 3D coordinates
coords = []
grayscale_values = []  # Store grayscale values for coloring

for index in range(len(indices)):
    x = index % 32
    y = (index // 32) % 32
    z = index // 1024
    coords.append((x, y, z))
    grayscale_values.append(indices[index])  # Use index as the grayscale value

# Convert the list of coordinates to a numpy array
coords_np = np.array(coords)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points in 3D space with colors based on grayscale values
# Normalize grayscale values to a 0-1 range for the colormap
normalized_gray = 1 - (np.array(grayscale_values) / 255)
ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], c=normalized_gray, cmap='gray', marker='o')

# Set labels for axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Set axis ticks to integers using MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.zaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
