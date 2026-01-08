"""
Shared utilities for traffic sign recognition project.
Contains common functions and constants used across all scripts.
"""

import cv2
import numpy as np
from PIL import Image

# Define the classes dictionary (single source of truth)
CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

# Model constants
IMG_HEIGHT = 30
IMG_WIDTH = 30
NUM_CLASSES = 43


def preprocess_image(image, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    """
    Preprocess image for prediction.

    Parameters:
    -----------
    image : str, numpy.ndarray, or PIL.Image
        Image path, numpy array, or PIL Image object
    img_height : int
        Target height for resized image
    img_width : int
        Target width for resized image

    Returns:
    --------
    numpy.ndarray
        Processed image as a NumPy array, normalized to [0, 1]
    """
    try:
        # Handle different input types
        if isinstance(image, str):
            # Image path
            image_array = cv2.imread(image)
            if image_array is None:
                raise ValueError(f"Failed to load image at {image}")
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            # Numpy array
            image_array = image.copy()
            # Ensure RGB format
            if len(image_array.shape) == 2:  # Grayscale
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif hasattr(image, 'convert'):
            # PIL Image
            image_array = np.array(image.convert('RGB'))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Handle RGBA
        if image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # Resize image
        image_resized = cv2.resize(
            image_array,
            (img_width, img_height),
            interpolation=cv2.INTER_AREA
        )

        # Normalize pixel values to [0, 1]
        processed_image = image_resized.astype('float32') / 255.0

        return processed_image
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


def predict_top_classes(model, image_data, top_n=5):
    """
    Performs prediction on image data and returns top N predicted classes.

    Parameters:
    -----------
    model : keras.Model
        The loaded Keras model
    image_data : numpy.ndarray
        Preprocessed image data (can be a single image or a batch)
    top_n : int
        Number of top predictions to retrieve

    Returns:
    --------
    list of dict
        A list of dictionaries containing 'class_id', 'class_name',
        and 'probability' for the top N predictions
    """
    if image_data.ndim == 3:
        # If single image, add batch dimension
        image_data = np.expand_dims(image_data, axis=0)

    predictions = model.predict(image_data, verbose=0)

    results = []
    for pred in predictions:
        # Get top N class indices and probabilities
        top_n_indices = np.argsort(pred)[::-1][:top_n]
        top_n_probabilities = pred[top_n_indices]

        image_results = []
        for idx, prob in zip(top_n_indices, top_n_probabilities):
            image_results.append({
                'class_id': int(idx),
                'class_name': CLASSES.get(idx, f'Unknown Class {idx}'),
                'probability': float(prob)
            })
        results.append(image_results)

    return results


def get_class_name(class_id):
    """
    Get class name from class ID.

    Parameters:
    -----------
    class_id : int
        The class ID

    Returns:
    --------
    str
        The class name
    """
    return CLASSES.get(class_id, f'Unknown Class {class_id}')


def get_num_classes():
    """
    Get the total number of classes.

    Returns:
    --------
    int
        Number of classes
    """
    return NUM_CLASSES


def get_image_shape():
    """
    Get the expected image shape.

    Returns:
    --------
    tuple
        (height, width, channels)
    """
    return (IMG_HEIGHT, IMG_WIDTH, 3)