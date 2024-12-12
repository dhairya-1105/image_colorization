# Image Colorization Using GANs

This repository contains the code and model for colorizing black-and-white images using a Generative Adversarial Network (GAN). The model leverages a U-Net architecture with a ResNet34 backbone for the generator and a standard discriminator to perform colorization.

## Project Overview
The goal of this project is to develop a deep learning-based model that can colorize grayscale images by predicting the "ab" channels in the LAB color space. The model was trained using the Deep Convolutional Generative Adversarial Network (DCGAN) architecture with modifications to better suit the image colorization task.


## Setup
1. **Clone the repository**:
   ```
   git clone https://github.com/dhairya-1105/image_colorization.git
   cd image_colorization
   ```
2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Download the pretrained model**: If you're not training the model yourself, download the pretrained model weights (net_G_epoch_20.pth), or you can use the provided weights.
4. **Run the Streamlit app**:
   ```
   streamlit run app.py
   ```
## Model Architecture
The generator is based on a U-Net architecture with a ResNet34 backbone. It predicts the "ab" channels of the LAB color space from grayscale "L" channel input. The model was trained with the following setup:

- **Input**: Grayscale images (L channel of LAB)
- **Output**: Colorized images (ab channels of LAB)
- **Loss**: A combination of adversarial loss and L1 loss.
