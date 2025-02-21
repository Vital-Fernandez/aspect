import numpy as np

# Example 1D array
array = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
waveR = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

# Identify where changes occur (edges of ones and zeros)
edges = np.diff(np.concatenate(([0], array, [0])))
start_indices = np.where(edges == 1)[0]
end_indices = np.where(edges == -1)[0] - 1

# Calculate central indices
central_indices = [(start + end) // 2 for start, end in zip(start_indices, end_indices)]

print("Central indices for each segment of ones:", central_indices)
print("Central indices for each segment of ones:", waveR[central_indices])