import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from skimage.color import lab2rgb
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from huggingface_hub import hf_hub_download

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)

def build_res_unet(n_input=1, n_output=2, size=256):
    body = create_body(resnet34, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size))
    return net_G

IMG_DIM = 256
model_path = hf_hub_download(repo_id="dhairya-1105/image-colorization", filename="net_G_epoch_20.pth")
net_G = build_res_unet(n_input=1, n_output=2, size=IMG_DIM)
net_G.load_state_dict(torch.load(model_path))  # Adjust path
net_G.eval().to(device)

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_DIM, IMG_DIM)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# App UI
st.title("Image Colorization with Deep Learning")
st.write("Upload a grayscale image to see the colorized output.")

uploaded_file = st.file_uploader("Upload Grayscale Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_L = preprocess(input_image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output_ab = net_G(input_L)

    # De-normalize
    input_L = (input_L.squeeze(0).cpu().numpy() + 1.0) * 50.0
    output_ab = output_ab.squeeze(0).cpu().numpy() * 128.0

    # Combine and convert to RGB
    lab_image = np.concatenate([input_L, output_ab], axis=0).transpose(1, 2, 0)
    rgb_image = lab2rgb(lab_image)

    # Display the colorized image
    st.image(rgb_image, caption="Colorized Image", use_column_width=True)
