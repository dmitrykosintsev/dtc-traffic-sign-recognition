import os
import sys
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

# Define the classes dictionary
classes = {0: 'Speed limit (20km/h)',
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
           42: 'End no passing veh > 3.5 tons'}


def preprocess_image(image_path, img_height=30, img_width=30):
    """
    Loads and preprocesses a single image for prediction.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((img_width, img_height))
        processed_image = np.array(resize_image)
        processed_image = processed_image.astype('float32') / 255.0

        return processed_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def predict_top_classes(model, image_data, classes, top_n=5):
    """
    Performs prediction on image data and returns top N predicted classes.
    """
    if image_data.ndim == 3:
        image_data = np.expand_dims(image_data, axis=0)

    predictions = model.predict(image_data)

    results = []
    for i, pred in enumerate(predictions):
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


def get_user_input():
    """
    Get model path and image path(s) from user input.
    """
    print("=" * 60)
    print("Traffic Sign Recognition - Prediction")
    print("=" * 60)

    # Get model path
    while True:
        model_path = input("\nEnter the path to your trained model (.h5 file): ").strip()
        if os.path.exists(model_path):
            break
        else:
            print(f"Error: Model file not found at '{model_path}'. Please try again.")

    # Get image path(s)
    print("\nEnter image path(s) for prediction:")
    print("  - Single image: /path/to/image.png")
    print("  - Multiple images (comma-separated): /path/img1.png, /path/img2.png")
    print("  - Directory: /path/to/images/")

    image_input = input("\nImage path(s): ").strip()

    # Parse image paths
    image_paths = []
    if os.path.isdir(image_input):
        # If directory, get all image files
        valid_extensions = ('.png', '.jpg', '.jpeg', '.ppm', '.bmp')
        for file in os.listdir(image_input):
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(image_input, file))
        if not image_paths:
            print(f"Warning: No image files found in directory '{image_input}'")
    elif ',' in image_input:
        # Multiple comma-separated paths
        image_paths = [p.strip() for p in image_input.split(',')]
    else:
        # Single image path
        image_paths = [image_input]

    # Validate image paths
    valid_paths = []
    for path in image_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"Warning: Image not found at '{path}', skipping...")

    if not valid_paths:
        print("Error: No valid image paths provided.")
        sys.exit(1)

    return model_path, valid_paths


if __name__ == "__main__":
    # Get user input
    model_path, image_paths = get_user_input()

    # Load the trained model
    try:
        model = keras.models.load_model(model_path)
        print(f"\n✓ Model '{model_path}' loaded successfully.")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        sys.exit(1)

    # Perform predictions
    print("\n" + "=" * 60)
    print(f"Predicting {len(image_paths)} image(s)...")
    print("=" * 60)

    for idx, img_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] Image: {img_path}")
        print("-" * 60)

        processed_img = preprocess_image(img_path)

        if processed_img is not None:
            top_predictions = predict_top_classes(model, processed_img, classes, top_n=5)

            print("Top 5 Predictions:")
            for i, pred_result in enumerate(top_predictions[0], 1):
                class_id = pred_result['class_id']
                class_name = pred_result['class_name']
                probability = pred_result['probability']
                bar = "█" * int(probability * 30)
                print(f"  {i}. [{probability * 100:5.2f}%] {bar} {class_name}")
        else:
            print(f"  ✗ Could not process image")

    print("\n" + "=" * 60)
    print("Predictions Complete!")
    print("=" * 60)