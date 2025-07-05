# Image Colorization Using GANs

This repository contains the code and model for colorizing black-and-white images using a Generative Adversarial Network (GAN). The model leverages a U-Net architecture with a ResNet34 backbone for the generator and a standard discriminator to perform colorization.

## Project Overview
The goal of this project is to develop a deep learning-based model that can colorize grayscale images by predicting the "ab" channels in the LAB color space. The model was trained using the Pix2Pix Conditional Generative Adversarial Network (cGAN) architecture.


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
3. **Run the Streamlit app**:
   ```
   streamlit run app.py
   ```
## Model Architecture
The generator is based on a U-Net architecture with a ResNet34 backbone. It predicts the "ab" channels of the LAB color space from grayscale "L" channel input. The model was trained with the following setup:

- **Input**: Grayscale images (L channel of LAB)
- **Output**: Colorized images (ab channels of LAB)
- **Loss**: A combination of adversarial loss and L1 loss.

The discriminator is a **PatchGAN** which uses convolutions to convert the Generator's 256 * 256 * 3 output into a 30 * 30 * 1 feature map, such that each pixel value of the feature map corresponds to a 70 * 70 patch of the generated image.

## Implementation Details
### Dataset Used
I used 10k randomly sampled images from the COCO Minitrain dataset, which originally consists of 25k images. The dataset can be accessed from [Kaggle](https://www.kaggle.com/datasets/trungit/coco25k)

### Training
The model uses a pre-trained Resnet-34 as the backbone for the U-Net architecture of the generator. The generator was trained independently for 20 epochs and then the entire model (Generator+Discriminator) was trained for 20 epochs. This is different from the strategy mentioned in the original [Pix2Pix](https://arxiv.org/abs/1611.07004) paper where the authors directly go for adverserial training. Pretraining the generator improves the results significantly. The parameters for adverserial training are kept the same as in the paper. The architectures for generator and discriminator can be accessed from the assets folder. A high level overview is given below:

![Pix2Pix Architecture](/assets/Google%20ChromeScreenSnapz096.jpg)

### Results
![Img1](/assets/Screenshot%202024-12-12%20122440.png)
![Img2](/assets/Screenshot%202024-12-12%20122503.png)
![Img3](/assets/Screenshot%202024-12-12%20122522.png)
![Img4](/assets/Screenshot%202024-12-12%20122601.png)
![Img5](/assets/Screenshot%202024-12-12%20122726.png)
![Img6](/assets/Screenshot%202024-12-12%20122749.png)

### Performance Comparison

| Model Variant             | PSNR (dB) | SSIM   |
|--------------------------|-----------|--------|
| Without Pretrained Generator | 22.18     | 0.878 |
| With Pretrained Generator    | **26.59** | **0.924** |

