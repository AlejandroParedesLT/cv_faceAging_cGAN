import os
import numpy as np
from random import random
from datetime import datetime
import scipy.io

# Load the .mat file
mat_file_path = "C:/Users/ALEJANDRO/Documents/7. DUKE/1. AIPI 590 - Computer Vision/assignment/3. third_project/wiki/wiki.mat"  
data = scipy.io.loadmat(mat_file_path)

meta_data = data['wiki']  # Change to 'imdb' if using the IMDb dataset

# Extract attributes
dob = meta_data['dob'][0][0].flatten()  # Date of birth
photo_taken = meta_data['photo_taken'][0][0].flatten()  # Year photo was taken
full_path = meta_data['full_path'][0][0].flatten()  # Image file paths
gender = meta_data['gender'][0][0].flatten()  # Gender
face_score = meta_data['face_score'][0][0].flatten()  # Face detection score
second_face_score = meta_data['second_face_score'][0][0].flatten()  # Second face score

# Convert byte strings to normal strings
full_path = [str(path[0]) for path in full_path]

# Calculate age (assuming photo taken in the middle of the year)
ages = np.array([
    datetime(photo_year, 7, 1).year - datetime.fromordinal(int(d)).year
    if not np.isnan(d) else np.nan
    for photo_year, d in zip(photo_taken, dob)
])

# Filter valid entries (e.g., face_score > threshold and age > 0)
valid_entries = (face_score > 1.0) & (np.isnan(second_face_score)) & (ages > 0)
filtered_paths = np.array(full_path)[valid_entries]
filtered_ages = ages[valid_entries]

# Define age groups
def age_to_group(age):
    if age <= 20:
        return 0
    elif age <= 30:
        return 1
    elif age <= 40:
        return 2
    elif age <= 50:
        return 3
    else:
        return 4

# Initialize lists for train and test sets
train_txt = []
test_txt = []

train_groups = {i: [] for i in range(5)}
test_groups = {i: [] for i in range(5)}

# Process the dataset
for path, age in zip(filtered_paths, filtered_ages):
    group = age_to_group(age)
    strline = f"{path} {group}\n"
    if random() < 0.9:
        train_txt.append(strline)
        train_groups[group].append(strline)
    else:
        test_txt.append(strline)
        test_groups[group].append(strline)

# Save the files
output_dir = "data/wiki-lists"
os.makedirs(output_dir, exist_ok=True)

# Save train and test group files
for group, lines in train_groups.items():
    with open(os.path.join(output_dir, f"train_age_group_{group}.txt"), 'w') as f:
        f.writelines(lines)

for group, lines in test_groups.items():
    with open(os.path.join(output_dir, f"test_age_group_{group}.txt"), 'w') as f:
        f.writelines(lines)

# Save combined train and test files
with open(os.path.join(output_dir, "train.txt"), 'w') as f:
    f.writelines(train_txt)

with open(os.path.join(output_dir, "test.txt"), 'w') as f:
    f.writelines(test_txt)
