# Use official TensorFlow image with GPU support (or use tensorflow/tensorflow:latest for CPU only)
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train.py .
COPY predict.py .

# Create directories for models and data
RUN mkdir -p /app/models /app/dataset /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "predict.py"]