import numpy as np
import requests
import matplotlib.pyplot as plt
from io import StringIO


url = 'https://ics.uci.edu/~ihler/classes/cs178/data/nyc_housing.txt'
with requests.get(url) as link:
    datafile = StringIO(link.text)
    nych = np.genfromtxt(datafile, delimiter=',')
    nych_X, nych_y = nych[:, :-1], nych[:, -1]

#- What is the shape of `nych_X` and `nych_y`? 
shape_X = nych_X.shape
shape_y = nych_y.shape

#- How many datapoints are in our dataset, and how many features does each datapoint have? 
num_data_points = shape_X[0]
num_features = shape_X[1]

#- How many different classes (i.e. labels)  are there? 
unique_labels = np.unique(nych_y)
num_classes = len(unique_labels)

# - Print rows 3, 4, 5, and 6 of the feature matrix and their corresponding labels. Since Python is zero-indexed, we will count our rows starting at zero -- for example, by "row 0" we mean `nych_X[0, :]`, and "row 1" means `nych_X[1, :]`, etc. (Hint: you can do this in two lines of code with slicing).
rows_3_to_6 = nych_X[2:6, :]
labels_3_to_6 = nych_y[2:6]

# print results
# print(f"Shape of nych_X: {shape_X}")
# print(f"Shape of nych_y: {shape_y}")
# print(f"Number of data points: {num_data_points}")
# print(f"Number of features per data point: {num_features}")
# print(f"Unique classes (labels): {unique_labels}")
# print(f"Number of classes: {num_classes}")
# print(f"Rows 3 to 6 of nych_X: \n{rows_3_to_6}")
# print(f"Corresponding labels: {labels_3_to_6}")

# Problem 1.2
# # Calculate the mean of each feature
# mean_values = np.mean(nych_X, axis=0)
# print("Mean of each feature:", mean_values)

# # Calculate the variance of each feature
# variance_values = np.var(nych_X, axis=0)
# print("Variance of each feature:", variance_values)

# # Calculate the standard deviation of each feature
# std_dev_values = np.std(nych_X, axis=0)
# print("Standard Deviation of each feature:", std_dev_values)

# # Calculate the minimum value of each feature
# min_values = np.min(nych_X, axis=0)
# print("Minimum value of each feature:", min_values)

# # Calculate the maximum value of each feature
# max_values = np.max(nych_X, axis=0)
# print("Maximum value of each feature:", max_values)


# # Problem 1.3
# # Extract data for Manhattan (y = 0)
# manhattan_data = nych_X[nych_y == 0]

# # Compute the mean and standard deviation for Manhattan
# manhattan_mean = np.mean(manhattan_data, axis=0)
# manhattan_std = np.std(manhattan_data, axis=0)

# # Print Manhattan statistics
# print("Manhattan (y=0) mean of each feature:", manhattan_mean)
# print("Manhattan (y=0) standard deviation of each feature:", manhattan_std)

# # Extract data for Bronx (y = 1)
# bronx_data = nych_X[nych_y == 1]

# # Compute the mean and standard deviation for Bronx
# bronx_mean = np.mean(bronx_data, axis=0)
# bronx_std = np.std(bronx_data, axis=0)

# # Print Bronx statistics
# print("Bronx (y=1) mean of each feature:", bronx_mean)
# print("Bronx (y=1) standard deviation of each feature:", bronx_std)

# # Problem 1.4
# # Create a figure with 1 row and 3 columns
# fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# # Plot histogram for each feature
# for i in range(3):
#     axes[i].hist(nych_X[:, i], bins=20, edgecolor='black')
#     axes[i].set_title(f"Feature {i}")
#     axes[i].set_xlabel("Value")
#     axes[i].set_ylabel("Frequency")

# # Adjust layout to prevent overlap
# fig.tight_layout()

# # Show the plot
# plt.show()

# Create a figure with 3 rows and 3 columns
fig, axes = plt.subplots(3, 3, figsize=(8, 8))  

### YOUR CODE STARTS HERE ###
colors = ['blue', 'green', 'red']

# Loop through each pair of features and create a scatter plot
for i in range(3):
    for j in range(3):
        if i != j:
            # Create scatter plot for feature pairs (i, j)
            for label in np.unique(nych_y):
                axes[i, j].scatter(nych_X[nych_y == label, i], nych_X[nych_y == label, j],
                                   c=colors[int(label)], label=f'Class {int(label)}', s=10)
            # Set x and y labels
            axes[i, j].set_xlabel(f'Feature {i}')
            axes[i, j].set_ylabel(f'Feature {j}')
        else:
            # Empty plot when i == j (no need to plot feature against itself)
            axes[i, j].axis('off')

# Add a legend to one subplot (only needs to be displayed once)
axes[0, 2].legend()

###  YOUR CODE ENDS HERE  ###

fig.tight_layout()