# Deepfake Detection Using CNNs & Image Super Resolution :robot:


# Overview
During the winter quarter of 2025 at Northwestern my team was given a final project where we had to take 1 publically available dataset and solve 2 problems. My team decided to use a deepfake dataset found on kaggle to classify between real and fake and experiment with image super resolution which we hadn't talked about much in class.

# Theme

The overarching theme of our work was how can we balance accuracy with speed. Using large deep learning models may be necessary depending on the task, but we wanted to see how performance changed between shallower and deeper models, full sized images and downsampled images.

# Classification Results

Our results showed that the best model we made was actually the smallest one (~84% test accuracy). The transfer learning models took a lot longer to train and didn't perform as well as our custom CNN.

# Image Super Resolution Results

Image super resolution is using deep learning to enlarge smaller images. For our sake, we super resolved our images from 256x256 to 1024x1024 using 2 pretrained models (ESRGAN & RDN). We found that the ESRGAN model didn't change the final image at all compared to a simple upsample without using deep learning. Where the RDN showed some odd pixelation is some areas of the image. So, if we were to do further work, we would focus on how super resolution affects upsampling the image when we had previously downsampled it before. 

# Developer Setup

## If you want to download the dataset we used:
```bash
curl -L -o deepfake-and-real-images.zip\
  https://www.kaggle.com/api/v1/datasets/download/manjilkarki/deepfake-and-real-images
```

## Create Virtual Environment & Install Depenencies
```bash
pip install uv
uv venv
uv sync
```

# Contributors
- [Yael Braverman](https://github.com/ybraverman)
- [Teddy Debreu](https://github.com/tdebreu)
- [John He](https://github.com/reigningforest)