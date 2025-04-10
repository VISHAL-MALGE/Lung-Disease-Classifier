import os
import numpy as np
import cv2

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get project root directory
DATASET_DIR = os.path.join(BASE_DIR, "Lung X-Ray Image")  # Dataset folder
SAVE_DIR = os.path.join(BASE_DIR, "processed_data")  # Where processed files will be saved

# Ensure 'processed_data' exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"✅ Created folder: {SAVE_DIR}")
else:
    print(f"✅ Folder exists: {SAVE_DIR}")


# Define labels
LABELS = ["Lung_Opacity", "Normal", "Viral Pneumonia"]
IMG_SIZE = 128  # Resize images to 128x128

# Load and preprocess images
X_train, y_train = [], []

for label in LABELS:
    class_dir = os.path.join(DATASET_DIR, label)  # Path to class folder
    if not os.path.exists(class_dir):
        print(f"❌ Error: Folder '{class_dir}' not found!")
        continue
    
    print(f"📂 Processing {label}...")
    
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
            X_train.append(img)
            y_train.append(LABELS.index(label))  # Store label as a number (0,1,2)
        except Exception as e:
            print(f"⚠️ Skipping {img_name}: {e}")

# Convert lists to NumPy arrays
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Add channel dimension
y_train = np.array(y_train)

print(f"Total processed images: {len(X_train)}")
print(f"Total labels: {len(y_train)}")

# Debugging before saving
print(f"🔍 X_train type: {type(X_train)}")
if isinstance(X_train, np.ndarray):
    print(f"✅ X_train shape: {X_train.shape}")
else:
    print("❌ ERROR: X_train is not a NumPy array!")

# Check directory write permissions
if os.access(SAVE_DIR, os.W_OK):
    print("✅ Directory is writable")
else:
    print("❌ ERROR: No write permission!")

# Test writing to folder
try:
    with open(os.path.join(SAVE_DIR, "test.txt"), "w") as f:
        f.write("Test write successful!")
    print("✅ Test file written successfully!")
except Exception as e:
    print(f"❌ ERROR: Cannot write test file! {e}")

# Save processed data
np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)

print(f"✅ Saved processed data to {SAVE_DIR}")


