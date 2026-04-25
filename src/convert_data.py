import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def convert_folder_to_csv(folder_path, output_file, dataset_type):
    """
    Convert a folder of emotion images to FER-2013 format CSV

    Args:
        folder_path: Path to folder with emotion subfolders
        output_file: Where to save the CSV
        dataset_type: 'train' or 'test' (for display only)
    """

    emotion_map = {
        'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
        'sad': 4, 'surprise': 5, 'neutral': 6
    }

    data = []

    print(f"\n🔄 Converting {dataset_type} data from: {folder_path}")

    for emotion_folder in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, emotion_folder)

        if not os.path.isdir(folder_path_full):
            continue

        emotion_name = emotion_folder.lower()
        if emotion_name not in emotion_map:
            print(f"⚠️ Skipping unknown folder: {emotion_folder}")
            continue

        emotion_label = emotion_map[emotion_name]
        print(f"\n📁 Processing {emotion_folder}...")

        image_files = [f for f in os.listdir(folder_path_full)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in tqdm(image_files, desc=f"  {emotion_folder}"):
            img_path = os.path.join(folder_path_full, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48))

            pixels = ' '.join(map(str, img.flatten()))
            data.append({'emotion': emotion_label, 'pixels': pixels})

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"\n✅ {dataset_type} data saved to: {output_file}")
    print(f"   Total images: {len(df)}")

    return df


# Convert both train and test
if __name__ == "__main__":
    # Convert test data
    convert_folder_to_csv(
        folder_path=r'C:\Users\Hp\Downloads\archive\test',
        output_file='../../data/fer2013_test_converted.csv',
        dataset_type='test'
    )

    # Convert train data
    convert_folder_to_csv(
        folder_path=r'C:\Users\Hp\Downloads\archive\train',
        output_file='../../data/fer2013_train_converted.csv',
        dataset_type='train'
    )

    print("\n All data conversion complete!")