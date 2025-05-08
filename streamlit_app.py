import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from gradcam import generate_gradcam  
from model import load_model  

st.title("🩺 Chest X-ray Classification with ResNet18 + Grad-CAM")


st.sidebar.write("Upload a chest X-ray image and get a model prediction with visual explanation.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class_names = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]


@st.cache_resource
def load_trained_model():
    model = load_model("resnet_model.pth")
    model.eval()
    return model

model = load_trained_model()


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()

    st.success(f"🧠 Predicted: **{class_names[predicted_class]}** ({confidence*100:.2f}% confidence)")

  
    st.subheader("Explainable AI: Grad-CAM Visualization")
    cam_fig = generate_gradcam(model, input_tensor, class_idx=predicted_class)
    st.pyplot(cam_fig)
