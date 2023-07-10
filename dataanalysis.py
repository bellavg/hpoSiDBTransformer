import torch

# Load the tensor data from the .pth file
labels = torch.load('denselabels.pth')

# Count the total number of tensors
total_tensors = len(labels)

# Initialize counters for statistics
total_dangling_bonds = 0
total_zero_classifications = 0
total_one_classifications = 0
total_empty_space = 0

for tensor in labels:
    # Count the number of silicon dangling bonds
    total_dangling_bonds += (tensor == 1).sum().item()
    total_dangling_bonds += (tensor == 0).sum().item()

    # Count the number of 0 and 1 classifications
    total_zero_classifications += (tensor == 0).sum().item()
    total_one_classifications += (tensor == 1).sum().item()

    # Count the number of empty spaces
    total_empty_space += (tensor == -1).sum().item()

# Print the calculated statistics
print("Total Tensors:", total_tensors)
print("Total Silicon Dangling Bonds:", total_dangling_bonds)
print("Total Neutral SiDBs:", total_zero_classifications)
print("Total Negatively Charged SiDBs:", total_one_classifications)
print("Percentage Neutral SiDBs:", total_zero_classifications / total_dangling_bonds)
print("Percentage Negatively Charged SiDBs:", total_one_classifications / total_dangling_bonds)
print("Total Empty Spaces:", total_empty_space)
print("Percentage Empty Spaces:", total_empty_space / (total_tensors * 42 * 42))
print("Percentage Dangling Bonds:", total_dangling_bonds / (total_tensors * 42 * 42))
