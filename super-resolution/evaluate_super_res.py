import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def evaluate_super_res() -> None:
    super_res_dir = "path/to/super/res/images"
    original_dir = "path/to/orig/images"

    results = []

    for folder in os.listdir(super_res_dir):
        print(f"Evaluating {folder} Images")
        curr_path = os.path.join(super_res_dir, folder)
        for filename in tqdm(os.listdir(curr_path), desc="Evaluating Images"):
            if filename.endswith(".jpg"):
                super_res_img = cv2.imread(os.path.join(curr_path, filename))
                super_res_img = cv2.cvtColor(super_res_img, cv2.COLOR_BGR2RGB)

                original_img = cv2.imread(os.path.join(original_dir, folder, filename))
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                original_img = cv2.resize(
                    original_img,
                    (super_res_img.shape[1], super_res_img.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                results.append(tf.image.psnr(original_img, super_res_img, 255))

    print(f"Average PSNR over validation set {np.array(results).mean()}")


def main():
    evaluate_super_res()


if __name__ == "__main__":
    main()
