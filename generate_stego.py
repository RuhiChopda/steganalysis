import cv2
import os
import numpy as np

def text_to_bin(text):
    return ''.join(format(ord(i), '08b') for i in text)

def embed_text(img, text):
    bin_data = text_to_bin(text)
    data_index = 0

    # Ensure image is in uint8 format
    img = img.astype(np.uint8)
    img_flat = img.flatten()

    for i in range(len(img_flat)):
        if data_index < len(bin_data):
            # Bitwise LSB replacement
            img_flat[i] = (int(img_flat[i]) & ~1) | int(bin_data[data_index])
            data_index += 1
        else:
            break

    return img_flat.reshape(img.shape)

def create_stego_images(clean_folder, stego_folder, secret_text="HiddenMessage"):
    if not os.path.exists(stego_folder):
        os.makedirs(stego_folder)

    for file in os.listdir(clean_folder):
        img_path = os.path.join(clean_folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Skipping invalid image: {file}")
            continue

        stego_img = embed_text(img, secret_text)
        cv2.imwrite(os.path.join(stego_folder, file), stego_img)

    print("✅ Stego images generated successfully.")

create_stego_images('data/clean', 'data/stego', 'SecretData')
