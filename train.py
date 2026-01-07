import os
import cv2
import kagglehub
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Function to load and preprocess GTSRB data
def load_gtsrb_data(path, img_height=30, img_width=30,
                    validation_split=0.3, shuffle=True, random_state=42):
    """
    Efficiently load and preprocess GTSRB training data.

    Parameters:
    -----------
    path : str
        Path to the data directory
    img_height : int
        Target height for resized images
    img_width : int
        Target width for resized images
    validation_split : float
        Fraction of data to use for validation (0.0 to 1.0)
    shuffle : bool
        Whether to shuffle the data before splitting
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X_train, X_val, y_train, y_val : numpy arrays
        Training and validation data and labels (shuffled if shuffle=True)
    """

    train_path = os.path.join(path, "train")

    # Get number of categories
    num_categories = len([d for d in os.listdir(train_path)
                         if os.path.isdir(os.path.join(train_path, d))])

    print(f"Found {num_categories} categories")

    # First pass: count total images for efficient array allocation
    total_images = 0
    for i in range(num_categories):
        class_path = os.path.join(train_path, str(i))
        if os.path.exists(class_path):
            total_images += len([f for f in os.listdir(class_path)
                               if f.lower().endswith(('.png'))])

    print(f"Total images to load: {total_images}")

    # Pre-allocate arrays (much more efficient than list appending)
    image_data = np.zeros((total_images, img_height, img_width, 3), dtype=np.uint8)
    image_labels = np.zeros(total_images, dtype=np.int32)

    # Load images with progress bar
    idx = 0
    failed_images = []

    with tqdm(total=total_images, desc="Loading images") as pbar:
        for class_id in range(num_categories):
            class_path = os.path.join(train_path, str(class_id))

            if not os.path.exists(class_path):
                print(f"Warning: Class {class_id} directory not found")
                continue

            images = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.png'))]

            for img_name in images:
                img_path = os.path.join(class_path, img_name)

                try:
                    # Read image directly with cv2 (BGR format)
                    image = cv2.imread(img_path)

                    if image is None:
                        raise ValueError(f"Failed to load image")

                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Resize with better interpolation
                    image_resized = cv2.resize(image, (img_width, img_height),
                                              interpolation=cv2.INTER_AREA)

                    # Store in pre-allocated array
                    image_data[idx] = image_resized
                    image_labels[idx] = class_id
                    idx += 1

                except Exception as e:
                    failed_images.append((img_path, str(e)))

                pbar.update(1)

    # Trim arrays if some images failed to load
    if idx < total_images:
        image_data = image_data[:idx]
        image_labels = image_labels[:idx]
        print(f"Warning: {total_images - idx} images failed to load")

    # Report failures
    if failed_images:
        print(f"\nFailed to load {len(failed_images)} images:")
        for path, error in failed_images[:5]:  # Show first 5
            print(f"  {path}: {error}")
        if len(failed_images) > 5:
            print(f"  ... and {len(failed_images) - 5} more")

    print(f"\nSuccessfully loaded: {len(image_data)} images")
    print(f"Data shape: {image_data.shape}")
    print(f"Labels shape: {image_labels.shape}")
    print(f"Memory usage: {image_data.nbytes / (1024**2):.2f} MB")

    # Shuffle data if requested (even without validation split)
    if shuffle and validation_split == 0:
        print("\nShuffling data...")
        np.random.seed(random_state)
        shuffle_indices = np.random.permutation(len(image_data))
        image_data = image_data[shuffle_indices]
        image_labels = image_labels[shuffle_indices]

    # Split into train and validation sets
    if validation_split > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            image_data, image_labels,
            test_size=validation_split,
            random_state=random_state,
            shuffle=shuffle,  # Explicitly control shuffling
            stratify=image_labels  # Maintains class distribution
        )

        print(f"\nTrain set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")

        X_train_normalized = X_train.astype('float32') / 255.0
        X_val_normalized = X_val.astype('float32') / 255.0

        return X_train, X_val, y_train, y_val, X_train_normalized, X_val_normalized
    else:
        return image_data, None, image_labels, None, None, None

# Function to create CNN model
def create_cnn_model(input_shape=(30, 30, 3), num_classes=43, learning_rate=0.001, drop_first=0.25, drop_second=0.5):
    """
    Create a CNN model for GTSRB traffic sign classification.

    This architecture is inspired by successful GTSRB submissions and includes:
    - Multiple convolutional layers for feature extraction
    - Batch normalization for stable training
    - Dropout for regularization
    - Dense layers for classification

    Parameters:
    -----------
    input_shape : tuple
        Shape of input images (height, width, channels)
    num_classes : int
        Number of traffic sign classes (43 for GTSRB)

    Returns:
    --------
    model : keras.Model
        Compiled CNN model ready for training
    """

    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(drop_first),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(drop_first),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(drop_first),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(drop_second),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    # 1. Download the dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    print(f"Dataset downloaded to: {path}")

    # 2. Load and split data
    print("Loading and splitting data...")
    X_train, X_val, y_train, y_val, X_train_normalized, X_val_normalized = load_gtsrb_data(path)

    # 3. One-hot encode labels
    print("One-hot encoding labels...")
    num_classes = 43 # GTSRB has 43 classes
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    # 4. Create the model
    print("Creating CNN model...")
    model = create_cnn_model(input_shape=(30, 30, 3), num_classes=num_classes)
    model.summary()

    # 5. Define callbacks
    print("Setting up training callbacks...")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_gtsrb_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # 6. Train the model
    print("Starting model training...")
    history = model.fit(
        X_train_normalized, y_train,
        validation_data=(X_val_normalized, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    print("\nModel training complete! Best model saved to 'best_gtsrb_model.h5'")