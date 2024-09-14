import sys
import os

print("Starting script execution...")

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

print(f"Current directory: {current_dir}")
print(f"Source directory added to path: {src_dir}")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import pandas as pd
    import numpy as np
    from utils import download_images
    from constants import ALLOWED_UNITS
    print("All modules imported successfully")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ... [rest of the code remains the same until the main execution part]

# Main execution
if __name__ == "__main__":
    print("Entering main execution block")
    
    # Set up paths
    dataset_dir = os.path.join(current_dir, 'dataset')
    train_csv = os.path.join(dataset_dir, 'train.csv')
    test_csv = os.path.join(dataset_dir, 'test.csv')
    train_images_dir = os.path.join(dataset_dir, 'train_images')
    test_images_dir = os.path.join(dataset_dir, 'test_images')

    print(f"Dataset directory: {dataset_dir}")
    print(f"Train CSV path: {train_csv}")
    print(f"Test CSV path: {test_csv}")

    # Load and preprocess data
    print("Loading and preprocessing train data...")
    train_images, train_labels = load_and_preprocess_data(train_csv, train_images_dir)
    print(f"Train data loaded. Shape: {train_images.shape}")

    print("Loading and preprocessing test data...")
    test_images, _ = load_and_preprocess_data(test_csv, test_images_dir)
    print(f"Test data loaded. Shape: {test_images.shape}")
    
    # Split train data into train and validation
    val_split = 0.2
    val_size = int(len(train_images) * val_split)
    val_images, val_labels = train_images[:val_size], train_labels[:val_size]
    train_images, train_labels = train_images[val_size:], train_labels[val_size:]
    print(f"Data split. Train shape: {train_images.shape}, Validation shape: {val_images.shape}")
    
    # Create and train the model
    print("Creating model...")
    model = create_model()
    print("Model created. Training model...")
    history = train_model(model, train_images, train_labels, val_images, val_labels)
    print("Model training completed")
    
    # Make predictions on test data
    print("Making predictions on test data...")
    test_df = pd.read_csv(test_csv)
    predictions_df = predict_and_format(model, test_images, test_df)
    
    # Save predictions
    output_file = os.path.join(current_dir, 'test_out.csv')
    predictions_df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")
    print("Script execution completed")