# src/preprocess.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def load_and_preprocess_data():

    print("🚀 Starting data preprocessing...")

    train_csv = '../data/fer2013_train_converted.csv'
    test_csv = '../data/fer2013_test_converted.csv'

    # Check if files exist
    if not os.path.exists(train_csv):
        print(f"Train CSV not found at: {train_csv}")
        print("Please make sure your train CSV is in the data folder")
        return None

    if not os.path.exists(test_csv):
        print(f"Test CSV not found at: {test_csv}")
        print("Please make sure your test CSV is in the data folder")
        return None


    print("\n1️⃣ Loading CSV files...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"   Train images: {len(train_df)}")
    print(f"   Test images: {len(test_df)}")

    print("\nValidating data...")

    # Check emotion labels (should be 0-6)
    invalid_train = train_df[~train_df['emotion'].isin(range(7))]
    invalid_test = test_df[~test_df['emotion'].isin(range(7))]

    if len(invalid_train) > 0:
        print(f"Found {len(invalid_train)} invalid labels in train, removing...")
        train_df = train_df[train_df['emotion'].isin(range(7))]

    if len(invalid_test) > 0:
        print(f"Found {len(invalid_test)} invalid labels in test, removing...")
        test_df = test_df[test_df['emotion'].isin(range(7))]

    # Step 3: Convert pixels to arrays
    print("\nConverting pixel strings to arrays...")

    def pixels_to_array(pixel_string):
        try:
            # Split string and convert to numbers
            pixels = np.array(pixel_string.split(), dtype=np.float32)

            # Check if we have correct number of pixels (48x48 = 2304)
            if len(pixels) != 2304:
                print(f"   ⚠️ Unexpected pixel count: {len(pixels)}")
                return None

            # Reshape to 48x48
            return pixels.reshape(48, 48)
        except Exception as e:
            print(f"   ⚠️ Error converting pixels: {e}")
            return None

    # Convert training images
    X_train = []
    valid_train_indices = []
    print("   Processing training images...")
    for idx, pixels in enumerate(train_df['pixels']):
        img = pixels_to_array(pixels)
        if img is not None:
            X_train.append(img)
            valid_train_indices.append(idx)

    # Convert test images
    X_test = []
    valid_test_indices = []
    print("   Processing test images...")
    for idx, pixels in enumerate(test_df['pixels']):
        img = pixels_to_array(pixels)
        if img is not None:
            X_test.append(img)
            valid_test_indices.append(idx)

    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Get corresponding labels
    y_train = train_df.iloc[valid_train_indices]['emotion'].values
    y_test = test_df.iloc[valid_test_indices]['emotion'].values

    print(f"\n   ✅ Conversion complete:")
    print(f"      Train: {X_train.shape}")
    print(f"      Test: {X_test.shape}")

    # Step 4: Check for anomalies (blank or extreme images)
    print("\n4️⃣ Checking for image anomalies...")

    # Check for blank images (low standard deviation)
    train_std = np.std(X_train, axis=(1, 2))
    test_std = np.std(X_test, axis=(1, 2))

    blank_train = np.where(train_std < 5)[0]  # Very low contrast
    blank_test = np.where(test_std < 5)[0]

    if len(blank_train) > 0:
        print(f"   ⚠️ Found {len(blank_train)} low-contrast images in train")
        # Remove them
        mask_train = train_std >= 5
        X_train = X_train[mask_train]
        y_train = y_train[mask_train]

    if len(blank_test) > 0:
        print(f"   ⚠️ Found {len(blank_test)} low-contrast images in test")
        # Remove them
        mask_test = test_std >= 5
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]

    # Step 5: Normalize pixel values to [0, 1]
    print("\n5️⃣ Normalizing pixel values...")
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print(f"   Pixel range: [{X_train.min():.3f}, {X_train.max():.3f}]")

    # Step 6: Add channel dimension for CNN
    print("\n6️⃣ Reshaping for CNN...")
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)

    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")

    # Step 7: Split train into train and validation
    print("\n7️⃣ Creating validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    print(f"   Final splits:")
    print(f"      Training:   {X_train.shape}")
    print(f"      Validation: {X_val.shape}")
    print(f"      Test:       {X_test.shape}")

    # Step 8: Class distribution
    print("\n8️⃣ Class distribution in training set:")
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    for i, label in enumerate(emotion_labels):
        count = np.sum(y_train == i)
        percentage = (count / len(y_train)) * 100
        print(f"      {label:8}: {count:4d} images ({percentage:5.2f}%)")

    print("\n✅ Preprocessing complete!")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Save preprocessed data to numpy files for faster loading later
    """
    # Create preprocessed directory if it doesn't exist
    os.makedirs('../data/preprocessed', exist_ok=True)

    print("\n💾 Saving preprocessed data...")

    np.save('../data/preprocessed/X_train.npy', X_train)
    np.save('../data/preprocessed/y_train.npy', y_train)
    np.save('../data/preprocessed/X_val.npy', X_val)
    np.save('../data/preprocessed/y_val.npy', y_val)
    np.save('../data/preprocessed/X_test.npy', X_test)
    np.save('../data/preprocessed/y_test.npy', y_test)

    print("   ✅ Data saved to data/preprocessed/")
    print("   Files:")
    print("      - X_train.npy, y_train.npy")
    print("      - X_val.npy, y_val.npy")
    print("      - X_test.npy, y_test.npy")


def load_preprocessed_data():
    """
    Load preprocessed data if it exists (much faster than reprocessing)
    """
    try:
        X_train = np.load('../data/preprocessed/X_train.npy')
        y_train = np.load('../data/preprocessed/y_train.npy')
        X_val = np.load('../data/preprocessed/X_val.npy')
        y_val = np.load('../data/preprocessed/y_val.npy')
        X_test = np.load('../data/preprocessed/X_test.npy')
        y_test = np.load('../data/preprocessed/y_test.npy')

        print("✅ Loaded preprocessed data from disk")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except FileNotFoundError:
        print("⚠️ Preprocessed data not found. Run preprocess first.")
        return None


if __name__ == "__main__":
    # Run the preprocessing
    train, val, test = load_and_preprocess_data()

    if train is not None:
        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test

        # Save for future use
        save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test)
