# import os

# img_path = "sample_image.jpg"  # Change this to your actual file path
# print(f"File exists: {os.path.exists(img_path)}")


import os

file_path = r"D:\Lung Disease Prediction\sample_image.jpg"

if os.path.exists(file_path):
    print("✅ File found:", file_path)
else:
    print("❌ Error: Image file does not exist at the given path.")

