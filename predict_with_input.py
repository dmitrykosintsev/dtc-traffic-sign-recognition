import os
import sys
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

# Import shared utilities
from utils import (
    CLASSES,              # Traffic sign dictionary
    preprocess_image,     # Handles file paths
    predict_top_classes   # Makes predictions
)

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
            top_predictions = predict_top_classes(model, processed_img, 5)

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