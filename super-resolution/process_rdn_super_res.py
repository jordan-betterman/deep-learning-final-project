import os

from tqdm import tqdm

from PIL import Image

import torch
from torchsr.models import rdn
from torchvision import transforms


def upscale_images(input_dir, output_dir, device):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    model = rdn(scale=4, pretrained=True).to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    for dir in os.listdir(input_dir):
        curr_path = os.path.join(input_dir, dir)
        curr_out_path = os.path.join(output_dir, dir)

        os.makedirs(curr_out_path, exist_ok=True)
        for filename in tqdm(os.listdir(curr_path)[:8142], desc="Processing Images"):
            if filename.endswith(".jpg"):
                img_path = os.path.join(curr_path, filename)

                image = Image.open(img_path)
                image = transform(image)
                image = image.unsqueeze(0)
                image = image.to(device)
                preds = model(image)

                pil_image = transforms.ToPILImage()(preds.squeeze(0))
                pil_image.save(os.path.join(curr_out_path, filename))

    print(f"Processed images saved in: {output_dir}")


if __name__ == "__main__":

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    upscale_images("path/to/org/images", "path/to/super/res/images", device)
