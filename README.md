# Traffic sign recognition using GTSRB and CNN
This repository contains a capstone project completed for LM Zoomcamp 2025. You can check the final result here: https://dtc-traffic-sign-recognition.streamlit.app/

## Problem description
Traffic sign recognition and classification is an important problem for self-driving cars as it provides invaluable information for the vehicles to follow the rules and predict possible changes on the road. This projects aims to provide a model that can accurately recognise and classify road signs in over 40 classes. 

### Dataset
GTSRB - German Traffic Sign Recognition Benchmark: www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
### Analysing the data
As mentioned on the Kaggle page of the dataset, it contains 50k+ images and 43 classes.
* Class analysis:
![Class analysis](./graphs/class_distribution.png)
* Picture size distribution:
![Picture size](./graphs/gtsrb_size_analysis.png)
* Sampling 25 random images:
![Random images](./graphs/test_images_sample.png)

## Model training

### Training
Tensorflow and Keras are used for creating a model.
Default parameters:
* learning rate: 0.001
* dropout for the first three convolutional blocks: 0.25
* dropout for the the last layer: 0.5
```python
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
```

### Output
The first trained model was saved with the following results:
```python
Epoch 34/50
425/429 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9994 - loss: 0.0019
Epoch 34: val_accuracy improved from 0.99932 to 0.99949, saving model to best_gtsrb_model.h5
```
![Training results](./graphs/training_history.png)
Though the result was pretty good, I tried to adjust a few parameters to see whether it can be improved further:
* learning rate: [0.0001, 0.001, 0.01, 0.1]
* 1st dropout rate (first three layers): [0, 0.25, 0.5]
* 2nd dropout rate (other layers): [0, 0.25, 0.5]

There was no improvement, so the model was saved with the default parameters.

Running the model on the test dataset:
```python
395/395 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step
Test Data accuracy:  98.68566904196358
```

## Notebook to script
* Running train.py:

![Train](./graphs/train.png)
* Running predict.py:

![Predict](./graphs/predict.png)

**IMPORTANT:** adjust kaggle_base_path = '/kaggle/input/gtsrb-german-traffic-sign' before running predict.py.

Alternatively, use the predict_with_input.py and enter the paths for the model and a file to analyse:
![Predict](./graphs/predict_with_input.png)

## How to reproduce
The model was trained and saved using Google Colab. If you want to reproduce the results, either:
* open the notebook in Google Colab and follow the steps;

OR
* download the dataset from Kaggle using any of the ways provided on the website and adjust the path variable from predict.py accordingly:
```python
kaggle_base_path = '/kaggle/input/gtsrb-german-traffic-sign'
```
Additionally, check the imports in the notebook.

## Model deployment
### Dependencies
See the [requirements.txt](requirements.txt) to see packages that are required to run the model in Docker.

### Containerization
See the [Dockerfile](Dockerfile).
1. Create the necessary directories:
```bash
mkdir -p dataset models output
```
2. Build the Docker Image
```bash
docker build -t traffic-sign-recognition:latest .
```
3. Run the Container for training
```bash
docker build -t traffic-sign-recognition:latest .
```
4. Run the Container for prediction
```bash
docker run -it --rm \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/output:/app/output \
  traffic-sign-recognition:latest python predict.py
```
You'll be prompted to enter:

* Model path (e.g., /app/models/best_gtsrb_model.h5)
* Image path(s) (e.g., /app/dataset/Test/00000.png)

### Running a local Streamlit app
1. Check whether the Streamlit config exists here: .streamlit/config.toml. If not, create it:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = false

[browser]
gatherUsageStats = false
```
2. Install requirements and run the app:
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```
3. Go to http://localhost:8501 in your browser.


### Cloud deployment
#### Interactive version on Streamlit
Go to https://dtc-traffic-sign-recognition.streamlit.app/

#### Deploy Docker with CLI on Google Cloud
The current model can be deployed to Google Cloud for prediction. Follow the steps.
1. Install prerequisites
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```
2. Enable required APIs
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```
3. Build and push to Google Container Registry
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/traffic-sign-recognition
```
4. Deploy to Cloud Run
```bash
gcloud run deploy traffic-sign-api \
  --image gcr.io/YOUR_PROJECT_ID/traffic-sign-recognition \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --allow-unauthenticated
```

## Challenges and learnings
* There seems to be a known bug for older versions of TF + ONNX. Unfortunately, Google Colab does not allow me update further: https://github.com/onnx/tensorflow-onnx/issues/2348. So I used a suggestion from ChatGPT to save the model in the ONNX format.
* Deploying models on Streamlit requires opencv-python-headless, not opencv-python