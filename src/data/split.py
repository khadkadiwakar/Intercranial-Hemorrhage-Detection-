import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. Define Constants ---
PROCESSED_DATA_DIR = 'data/processed/'
LABELS_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'labels.csv')
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

def split_data():
    """
    Loads the processed labels CSV and splits it into train, validation,
    and test sets, then saves them as new CSV files.
    """
    # Load the master labels file
    try:
        df = pd.read_csv(LABELS_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {LABELS_CSV_PATH} not found. Please run the preprocessing script first.")
        return

    # First split: separate out the training set
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df['label'], # Stratify to maintain class balance
        random_state=42 # for reproducibility
    )

    # Second split: divide the remainder into validation and test sets
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)), # Calculate proportion for the test set
        stratify=temp_df['label'],
        random_state=42
    )

    # Save the new DataFrames to CSV files
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'), index=False)

    print("Data splitting complete.")
    print(f"Training set size: {len(train_df)} ({len(train_df)/len(df):.0%})")
    print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df):.0%})")
    print(f"Test set size: {len(test_df)} ({len(test_df)/len(df):.0%})")

# --- Main Execution Block ---
if __name__ == '__main__':
    split_data()