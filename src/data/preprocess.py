import os # Preprocess script for brain hemorrhage detection dataset
# This script processes raw brain CT images, resizes them, normalizes pixel values,
import pandas as pd # and saves them as numpy arrays for machine learning tasks.
# It also creates a CSV file with the labels for each image.
import numpy as np # For numerical operations and array handling
from PIL import Image # For image processing
# tqdm is used for displaying a progress bar during processing
from tqdm import tqdm # For progress bar
import shutil # For file operations

# --- 1. Define Constants ---
# Define paths relative to the project root directory
RAW_DATA_DIR = 'data/raw/' # Directory containing raw data
# This contain the 'hemorrhage_diagnosis.csv' and 'Patients_CT'
PROCESSED_DATA_DIR = 'data/processed/' # Directory to save processed data
# This will contain the processed images and labels CSV
IMAGE_SIZE = (128, 128) # Size to which images will be resized

def create_dataset_df(): # Create a DataFrame from the diagnosis CSV
    """
    Loads the diagnosis CSV, creates a binary label, and constructs the
    full path to each brain window image.
    """
    csv_path = os.path.join(RAW_DATA_DIR, 'hemorrhage_diagnosis.csv') # Path to the diagnosis CSV
    # Base directory for images
    image_base_dir = os.path.join(RAW_DATA_DIR, 'Patients_CT') # Directory containing patient images

    try: # Load the diagnosis CSV into a DataFrame
        df = pd.read_csv(csv_path)
    except FileNotFoundError: # If the CSV file is not found, print an error message and return an empty DataFrame
        print(f"Error: Diagnosis CSV not found at {csv_path}")
        return pd.DataFrame()

    # Create a simple binary label: 1 for any hemorrhage, 0 for normal
    hemorrhage_cols = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural'] # Columns indicating types of hemorrhage
    df['label'] = df[hemorrhage_cols].any(axis=1).astype(int) # Convert to binary label

    # Create the full filepath for each brain window image
    # The folder names are zero-padded to 3 digits (e.g., 49 -> '049')
    df['filepath'] = df.apply(
        lambda row: os.path.join(image_base_dir, f"{int(row['PatientNumber']):03d}", 'brain', f"{row['SliceNumber']}.jpg"), 
        axis=1
    )

    # Return only the necessary columns
    return df[['filepath', 'label']]

def preprocess_and_save_data(df): # Preprocess images and save them as numpy arrays
    """
    Processes each image (resize, normalize) and saves it as a .npy file
    in the processed directory. Also saves the labels to a new CSV.
    """
    # Define paths for the processed data
    processed_image_dir = os.path.join(PROCESSED_DATA_DIR, 'images') # Directory to save processed images

    # Ensure the directory for processed images exists
    os.makedirs(processed_image_dir, exist_ok=True)

    processed_data_records = []

    # Use tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Images"):
        try:
            # 1. Load Image
            img = Image.open(row['filepath']).convert('L') # 'L' for grayscale

            # 2. Resize Image
            img_resized = img.resize(IMAGE_SIZE, Image.LANCZOS)

            # 3. Normalize Pixel Values and add channel dimension
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=-1) # Shape becomes (128, 128, 1)

            # 4. Save the processed image as a numpy array
            base_filename = f"{os.path.splitext(os.path.basename(row['filepath']))[0]}_patient_{row.name}.npy"
            new_filepath = os.path.join(processed_image_dir, base_filename)
            np.save(new_filepath, img_array.astype(np.float32))

            # Add the new path and label to our list for the new CSV
            processed_data_records.append([new_filepath, row['label']])

        except FileNotFoundError:
            print(f"Warning: File not found, skipping: {row['filepath']}")

    # Create a new DataFrame for the processed data and save it
    processed_df = pd.DataFrame(processed_data_records, columns=['filepath', 'label'])
    processed_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'labels.csv'), index=False)

    print(f"\nProcessing complete.")
    print(f"{len(processed_df)} images and labels saved in '{PROCESSED_DATA_DIR}'")

# --- Main Execution Block ---
if __name__ == '__main__':
    # This block runs when you execute `python src/data/preprocess.py`
    print("Starting data preprocessing...")
    master_df = create_dataset_df()

    if not master_df.empty: # If the DataFrame is not empty, proceed with preprocessing
        print(f"Found {len(master_df)} records in the dataset.")
        preprocess_and_save_data(master_df)
        print("Data preprocessing finished successfully.")
    else: # If the DataFrame is empty, print a message and abort
        print("Could not create DataFrame. Aborting preprocessing.")
