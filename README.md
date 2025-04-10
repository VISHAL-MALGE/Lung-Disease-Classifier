Lung Disease Classifier  

The "Lung Disease Classifier" project is a comprehensive web application designed to 
classify lung diseases from X-ray images. This project leverages deep learning techniques to 
classify lung X-ray images into different disease categories and provides detailed information 
about the predictions, including confidence levels and affected lung areas. 


Directory and File Descriptions 

• app.py: The main Flask application file that serves the web interface, handles file uploads, processes images, and returns predictions. 
• error.py: A script to handle errors in the application. 
• lung_disease_model.h5: The trained deep learning model file used for predictions. 
• node_modules/: Directory containing dependencies for the project, typically managed by npm. 
• package-lock.json: Automatically generated file that records the exact versions of the dependencies installed. 
• package.json: File that records metadata about the project and its dependencies. 
• preprocess.py: Script to preprocess the X-ray images before training the model. 
• processed_data/: Directory containing preprocessed data files. 
  o X_train.npy: Numpy array file containing the training data. 
  o y_train.npy: Numpy array file containing the training labels. 
  
• requirements.txt: File listing all the Python dependencies required for the project. 
• sample_image.jpg: Sample image used for testing the application. 
• static/: Directory for storing static files like images, CSS, and JavaScript. 
o lung.png: Image used in the web application. 
• template/: Directory containing HTML templates for the web application. 
o index.html: The main HTML template for the home page. 
o uploads/: Directory for storing uploaded images. 
• test.txt: A test file used during development. 
• test_model.py: Script to test the trained model. 
• train_model.py: Script to train the deep learning model. 
• Lung X-Ray Image/: Directory containing raw X-ray images categorized into different classes. 
  o Lung_Opacity/: Directory containing images of lungs with opacity issues. 
  o Normal/: Directory containing images of normal lungs. 
  o Viral Pneumonia/: Directory containing images of lungs with viral pneumonia. 
  
Technologies and Libraries Used 
Flask 
  Flask is a lightweight WSGI web application framework in Python. It is used to create the web 
  interface for the project, handle HTTP requests, and serve HTML templates. 
  
TensorFlow 
  TensorFlow is an open-source machine learning framework. It is used to build and train the deep 
  learning model that predicts lung diseases from X-ray images. 
  
OpenCV 
  OpenCV (Open Source Computer Vision Library) is an open-source computer vision and 
  machine learning software library. It is used for image processing tasks in the project, such as 
  resizing and normalizing X-ray images. 
  
NumPy 
  NumPy is a library for the Python programming language, adding support for large, multi
  dimensional arrays and matrices, along with a large collection of high-level mathematical 
  functions to operate on these arrays. It is used to handle numerical data and perform array 
  operations. 
  
HTML/CSS/JavaScript 
  These technologies are used to create the front-end of the web application. HTML is used to 
  structure the content, CSS is used for styling, and JavaScript is used for client-side scripting. 
  
Markdown 
  Markdown is a lightweight markup language with plain text formatting syntax. It is used for 
  writing this documentation. 
  
Data Preprocessing 
The raw X-ray images are stored in the Lung X-Ray Image directory, categorized into different 
classes: Lung_Opacity, Normal, and Viral Pneumonia. 

Preprocessing Steps 
1. Loading Images: Images are loaded from their respective directories. 
2. Resizing: Images are resized to a uniform size (e.g., 128x128 pixels) to ensure 
consistency. 
3. Normalization: Pixel values are normalized to a range of 0 to 1 by dividing by 255.0. 
4. Splitting: Images are split into training and validation sets. 
5. Saving: Preprocessed images and labels are saved as NumPy arrays 
(X_train.npy and y_train.npy) for efficient loading during training. 
The preprocessing steps are implemented in the preprocess.py script.

Model Training 
Model Architecture 
The deep learning model used in this project is a Convolutional Neural Network (CNN). CNNs 
are particularly effective for image classification tasks due to their ability to capture spatial 
hierarchies in images. 

Training Process 
1. Loading Data: Preprocessed data is loaded from the NumPy array files. 
2. Model Definition: The CNN model is defined using TensorFlow's Keras API. 
3. Compilation: The model is compiled with an appropriate loss function, optimizer, and 
evaluation metric. 
4. Training: The model is trained on the training data with a specified number of epochs. 
5. Saving: The trained model is saved as lung_disease_model.h5. 
The training process is implemented in the train_model.py script.

Algorithms and Techniques 
• Convolutional Layers: Used to extract features from the input images. 
• Pooling Layers: Used to reduce the spatial dimensions of the feature maps. 
• Fully Connected Layers: Used to make final predictions based on the extracted features. 
• Activation Functions: ReLU activation is used for hidden layers, and Softmax activation 
is used for the output layer to produce probability distributions. 

Model Testing 
The trained model is tested using the test_model.py script. This script loads the trained model, 
makes predictions on sample images, and prints the results, including the predicted disease, 
confidence level, and affected lung areas. 

Web Application 
The web application is built using Flask. It provides a user interface for uploading X-ray images 
and displaying the prediction results. 
Key Components 
• Home Page: The home page (index.html) provides a form for uploading X-ray images. 
• File Upload Handling: The /predict route in app.py handles file uploads, processes the uploaded image, and returns the prediction results. 
• Prediction Display: The results, including the predicted disease, confidence level, and affected lung areas, are displayed on the web page. 
