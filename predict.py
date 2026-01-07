import os
import cv2
import numpy as np

from PIL import Image
from tensorflow import keras

# Define the classes dictionary
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

def preprocess_image(image_path, img_height=30, img_width=30):
    """
    Loads and preprocesses a single image for prediction.

    Parameters:
    -----------
    image_path : str
        Path to the image file.
    img_height : int
        Target height for the resized image.
    img_width : int
        Target width for the resized image.

    Returns:
    --------
    numpy.ndarray
        Processed image as a NumPy array, normalized to [0, 1].
    """
    try:
        # Read image directly with cv2 (BGR format)
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize with PIL for better quality, then convert back to numpy
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((img_width, img_height))
        processed_image = np.array(resize_image)

        # Normalize pixel values to [0, 1]
        processed_image = processed_image.astype('float32') / 255.0

        return processed_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_top_classes(model, image_data, classes, top_n=5):
    """
    Performs prediction on image data and returns top N predicted classes.

    Parameters:
    -----------
    model : keras.Model
        The loaded Keras model.
    image_data : numpy.ndarray
        Preprocessed image data (can be a single image or a batch).
    classes : dict
        Dictionary mapping class IDs to class names.
    top_n : int, optional
        Number of top predictions to retrieve. Defaults to 5.

    Returns:
    --------
    list of dict
        A list of dictionaries, each containing 'class_id', 'class_name',
        and 'probability' for the top N predictions for each input image.
    """
    if image_data.ndim == 3:
        # If single image, add batch dimension
        image_data = np.expand_dims(image_data, axis=0)

    predictions = model.predict(image_data)

    results = []
    for i, pred in enumerate(predictions):
        # Get top N class indices and probabilities
        top_n_indices = np.argsort(pred)[::-1][:top_n]
        top_n_probabilities = pred[top_n_indices]

        image_results = []
        for j in range(top_n):
            class_id = top_n_indices[j]
            class_name = classes.get(class_id, f'Unknown Class {class_id}')
            probability = top_n_probabilities[j]
            image_results.append({
                'class_id': int(class_id),
                'class_name': class_name,
                'probability': float(probability)
            })
        results.append(image_results)
    return results

if __name__ == "__main__":
    # Path to the trained model
    model_path = 'models/best_gtsrb_model.h5'

    # Load the trained model
    try:
        model = keras.models.load_model(model_path)
        print(f"Model '{model_path}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Example image paths for prediction (adjust these paths as needed)
    # These paths are examples assuming the dataset was downloaded to /kaggle/input/gtsrb-german-traffic-sign
    # You might need to change these if running locally or if your dataset path differs.
    kaggle_base_path = './dataset'
    example_image_paths = [
        os.path.join(kaggle_base_path, 'Test/00000.png'), # Example for Speed limit (20km/h)
        os.path.join(kaggle_base_path, 'Test/00001.png'), # Example for Speed limit (30km/h)
        os.path.join(kaggle_base_path, 'Test/00005.png'), # Example for Speed limit (80km/h)
        os.path.join(kaggle_base_path, 'Test/00010.png'), # Example for No passing veh over 3.5 tons
        os.path.join(kaggle_base_path, 'Test/00015.png')  # Example for No entry
    ]

    print("\n--- Starting Predictions ---")
    for img_path in example_image_paths:
        print(f"\nPredicting for image: {img_path}")
        processed_img = preprocess_image(img_path)

        if processed_img is not None:
            # Get top 5 predictions
            top_predictions = predict_top_classes(model, processed_img, classes, top_n=5)

            # Display predictions
            for pred_result in top_predictions[0]: # top_predictions is a list of results for each image in the batch
                class_id = pred_result['class_id']
                class_name = pred_result['class_name']
                probability = pred_result['probability']
                print(f"  Class ID: {class_id}, Name: {class_name}, Probability: {probability:.4f}")
        else:
            print(f"  Could not process image: {img_path}")
    print("\n--- Predictions Complete ---")