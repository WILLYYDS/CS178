# Create a figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# Plot histogram for each feature
for i in range(3):
    axes[i].hist(nych_X[:, i], bins=20, edgecolor='black')
    axes[i].set_title(f"Feature {i}")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

# Adjust layout to prevent overlap
fig.tight_layout()

# Show the plot
plt.show()