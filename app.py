import streamlit as st
import torch
from PIL import Image
import io

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_objects(image):
    """Detect objects in an image using YOLOv5 and return the image with drawn bounding boxes."""
    results = model(image)
    rendered_images = results.render()  # This will return images with bounding boxes
    img_with_boxes = Image.fromarray(rendered_images[0])  # Convert the tensor to a PIL Image
    return img_with_boxes


# Streamlit UI
st.title("YOLOv5 Object Detection")

# Image uploader
uploaded_file = st.file_uploader("Upload an image to detect objects", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect objects button
    if st.button("Detect Objects"):
        with st.spinner('Detecting...'):
            detected_image = detect_objects(image)
            st.image(detected_image, caption="Detected Objects", use_column_width=True)
            st.success("Detection completed!")

st.write("Note: This app uses YOLOv5 for object detection.")

