"""Pre-processing of the exported POCUS frames (paper Sec. III-B1).

Each exported frame is cropped to the B-mode image region and resized to 581x446 pixels. To give the left- and
right-arm images of a landmark the same orientation, the middle frame ('_b') of the left arm is compared with the
middle frame of the right arm with and without a horizontal flip; if the flipped version is more similar (lower
mean squared error), all left-arm frames of that landmark are flipped.

Input layout:  INPUT_DIR/<subject ID>/<subject ID>_<L|R><landmark>/*.jpeg   (e.g. US010/US010_L2/US010_L2_a.jpeg)
Output:        OUTPUT_DIR/*_processed.jpeg, OUTPUT_DIR/mse_flip_results.csv and comparison plots.
"""
import os
import re
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_DIR = "./saved_frames"
OUTPUT_DIR = "./saved_frames/flipped_output"
OUTPUT_SIZE = (581, 446)  # (width, height)


def crop_and_resize(image):
    h, w = image.shape
    cropped = image[15:h - 290, 220:w - 237]
    return cv2.resize(cropped, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)


def process_image(image_path, flip=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = crop_and_resize(image)
    return cv2.flip(image, 1) if flip else image


def compute_mse(img1, img2):
    return np.mean((img1.astype(np.float32) / 255.0 - img2.astype(np.float32) / 255.0) ** 2)


def find_subfolder_pairs(base_path):
    """{'<subject>_<landmark>': (left folder, right folder)}"""
    folders = defaultdict(dict)
    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)
        if not os.path.isdir(subject_path):
            continue
        for subfolder in os.listdir(subject_path):
            match = re.match(rf"({subject})_([LR])(\d)$", subfolder)
            if match:
                _, side, landmark = match.groups()
                folders[f"{subject}_{landmark}"][side] = os.path.join(subject_path, subfolder)
    return {key: (v["L"], v["R"]) for key, v in folders.items() if "L" in v and "R" in v}


def save_comparison_plot(pair_id, left, right, mse_not_flipped, mse_flipped, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"{pair_id} | MSE (not flipped): {mse_not_flipped:.4f} | MSE (flipped): {mse_flipped:.4f}")
    for i, (title, img) in enumerate([("Left (not flipped)", left), ("Left (flipped)", cv2.flip(left, 1)),
                                      ("Right", right)]):
        plt.subplot(1, 3, i + 1)
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{pair_id}_comparison.png"))
    plt.close()


def middle_frame(folder):
    return next((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("_b.jpeg")), None)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pairs = find_subfolder_pairs(INPUT_DIR)
    print(f"Found {len(pairs)} left-right folder pairs.")

    results = []
    for key, (left_folder, right_folder) in pairs.items():
        left_b, right_b = middle_frame(left_folder), middle_frame(right_folder)
        if not left_b or not right_b:
            print(f"Missing _b image for {key}")
            continue
        left_img, right_img = process_image(left_b), process_image(right_b)
        if left_img is None or right_img is None:
            print(f"Could not load images for {key}")
            continue

        mse_not_flipped = compute_mse(left_img, right_img)
        mse_flipped = compute_mse(cv2.flip(left_img, 1), right_img)
        flip = mse_flipped < mse_not_flipped
        save_comparison_plot(key, left_img, right_img, mse_not_flipped, mse_flipped,
                             os.path.join(OUTPUT_DIR, "comparison_plots"))

        # Save all frames of both arms, flipping the left arm if needed
        for folder, should_flip in [(left_folder, flip), (right_folder, False)]:
            for f in os.listdir(folder):
                if f.endswith(".jpeg"):
                    img = process_image(os.path.join(folder, f), flip=should_flip)
                    if img is not None:
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f.replace(".jpeg", "_processed.jpeg")), img,
                                    [cv2.IMWRITE_JPEG_QUALITY, 100])

        results.append({"Pair": key, "Left_Folder": left_folder, "Right_Folder": right_folder,
                        "MSE_Not_Flipped": mse_not_flipped, "MSE_Flipped": mse_flipped, "Flipped_Left": flip})

    csv_path = os.path.join(OUTPUT_DIR, "mse_flip_results.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Flip results saved to: {csv_path}")
