import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm


def upscale_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for dir in os.listdir(input_dir):
        curr_path = os.path.join(input_dir, dir)
        curr_out_path = os.path.join(output_dir, dir)
        os.makedirs(curr_out_path, exist_ok=True)
        for filename in tqdm(os.listdir(curr_path), desc="Processing Images"):
            if filename.endswith(".jpg"):
                img_path = os.path.join(curr_path, filename)

                # Load and preprocess image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255.0
                image = np.expand_dims(image, axis=0)

                sr_image = model(image)
                sr_image = np.clip(sr_image[0].numpy() * 255.0, 0, 255).astype(np.uint8)

                output_path = os.path.join(output_dir, dir, filename)
                cv2.imwrite(output_path, cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR))

    print(f"Processed images saved in: {output_dir}")


if __name__ == "__main__":

    model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

    with tf.device("/gpu:1"):
        upscale_images(
            "path/to/org/images",
            "path/to/super/res/images",
        )
