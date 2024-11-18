from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm
from skimage import transform as trans
import scipy.io
from datetime import datetime

# Paths
root_data_dir = "C:/Users/ALEJANDRO/Documents/7. DUKE/1. AIPI 590 - Computer Vision/assignment/3. third_project"
image_root_dir = os.path.join(root_data_dir, "wiki")
mat_file_path = os.path.join(root_data_dir, "wiki/wiki.mat")
store_image_dir = os.path.join(root_data_dir, "wiki_preprocess/wiki")

# Create output directory if not exists
os.makedirs(store_image_dir, exist_ok=True)

# Load the .mat file and filter valid entries
data = scipy.io.loadmat(mat_file_path)
meta_data = data['wiki']
dob = meta_data['dob'][0][0].flatten()  # Date of birth
photo_taken = meta_data['photo_taken'][0][0].flatten()  # Year photo was taken
full_path = [str(path[0]) for path in meta_data['full_path'][0][0].flatten()]  # Image file paths
face_score = meta_data['face_score'][0][0].flatten()  # Face detection score
second_face_score = meta_data['second_face_score'][0][0].flatten()  # Second face score

# Calculate ages
ages = np.array([
    datetime(photo_year, 7, 1).year - datetime.fromordinal(int(d)).year
    if not np.isnan(d) else np.nan
    for photo_year, d in zip(photo_taken, dob)
])

# Filter valid entries
valid_entries = (face_score > 1.0) & (np.isnan(second_face_score)) & (ages > 0)
filtered_paths = np.array(full_path)[valid_entries]
filtered_ages = ages[valid_entries]

# MTCNN parameters
src = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)
threshold = [0.6, 0.7, 0.9]
factor = 0.85
minSize = 20
imgSize = [120, 100]
detector = MTCNN(steps_threshold=threshold, scale_factor=factor, min_face_size=minSize)
keypoint_list = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']

for image_path in tqdm(filtered_paths, desc="Processing Images"):
    full_image_path = os.path.join(image_root_dir, image_path)
    store_path = os.path.join(store_image_dir, image_path)

    # Ensure the directory for the processed image exists
    os.makedirs(os.path.dirname(store_path), exist_ok=True)

    try:
        npimage = np.array(Image.open(full_image_path))
        if npimage is None or len(npimage.shape) < 2:  # Check for valid image dimensions
            print(f"Skipping invalid image: {full_image_path}")
            continue
    except Exception as e:
        print(f"Failed to open image {full_image_path}: {e}")
        continue

    # Detect faces
    try:
        dictface_list = detector.detect_faces(npimage)
    except Exception as e:
        print(f"Face detection failed for {full_image_path}: {e}")
        continue

    if not dictface_list:
        print(f"No face detected in {full_image_path}")
        continue  # Skip if no face detected

    # If multiple faces, choose the one closest to the center
    if len(dictface_list) > 1:
        boxs = np.array([face['box'] for face in dictface_list])
        center = np.array(npimage.shape[:2]) / 2
        face_centers = np.column_stack([
            boxs[:, 1] + boxs[:, 3] / 2,  # y-center
            boxs[:, 0] + boxs[:, 2] / 2   # x-center
        ])
        distances = np.linalg.norm(face_centers - center, axis=1)
        dictface = dictface_list[np.argmin(distances)]
    else:
        dictface = dictface_list[0]

    # Align and crop face
    try:
        dst = np.array([dictface['keypoints'][k] for k in keypoint_list], dtype=np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[:2, :]
        warped = cv2.warpAffine(npimage, M, (imgSize[1], imgSize[0]), borderValue=0.0)
        warped = cv2.resize(warped, (400, 400))

        # Save cropped face
        Image.fromarray(warped.astype(np.uint8)).save(store_path)
    except Exception as e:
        print(f"Failed to process or save image {full_image_path}: {e}")
        continue