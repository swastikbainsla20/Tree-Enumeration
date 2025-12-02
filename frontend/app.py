import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="Tree Enumeration and Path Optimization", layout="wide")

# --------------------------------------------------------------
# NAVBAR (PURE STREAMLIT)
# --------------------------------------------------------------
col1, col2, col3 = st.columns([2, 4, 2])

with col1:
    st.markdown("Tree Enumeration and Path Optimization")

with col2:
    st.write("")  # spacing
    nav = st.radio(
        "Navigation",
        ["Home", "About Us", "Documentation"],
        horizontal=True,
        label_visibility="collapsed",
    )

with col3:
    cA, cB = st.columns(2)
    with cA:
        st.button("Sign In")
    with cB:
        st.button("Sign Up")


st.write("---")  # thin line separator


# --------------------------------------------------------------
# HERO SECTION
# --------------------------------------------------------------
st.title("Tree Enumeration and Path Optimization")
st.subheader("Upload an image to begin processing.")

st.info(
    "**Problem Statement:** "
    "To build a deep learning-based system that can detect individual trees from aerial images "
    "and compute the largest safe region without trees using a spatial mask."
)

st.write("")


# --------------------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()


# --------------------------------------------------------------
# YOLO DETECTION FUNCTION
# --------------------------------------------------------------
def yolo_detect(img_np):
    results = model(img_np)[0]
    plot = results.plot()
    plot_rgb = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
    tree_count = len(results.boxes)
    return results, plot_rgb, tree_count


# --------------------------------------------------------------
# SAFE REGION COMPUTATION
# --------------------------------------------------------------
def compute_safe_area(img_np, boxes):
    H, W, _ = img_np.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        mask[y1:y2, x1:x2] = 1

    kernel = np.ones((240, 240), np.uint8)
    free_eroded = cv2.erode(1 - mask, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(free_eroded)
    max_area = 0
    best_label = 0

    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area > max_area:
            max_area = area
            best_label = label

    safe_region = (labels == best_label)

    overlay = img_np.copy()
    green = np.array([0, 255, 0], dtype=np.uint8)

    overlay[safe_region] = (
        0.3 * overlay[safe_region] +
        0.7 * green
    ).astype(np.uint8)

    return overlay


# --------------------------------------------------------------
# UPLOAD SECTION
# --------------------------------------------------------------
st.header("Upload Image")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img_pil = Image.open(uploaded)
    img_np = np.array(img_pil)

    # PREVIEW
    st.image(img_pil, caption="Uploaded Image", width=450)

    st.write("")

    # PROCESS BUTTON
    if st.button("Process Image"):
        with st.spinner("Analyzing image..."):
            results, yolo_img, count = yolo_detect(img_np)
            boxes = results.boxes.xyxy.cpu().numpy()
            safe_img = compute_safe_area(img_np, boxes)

        st.success(f"Total Trees Detected: **{count}**")

        # OUTPUT SIDE-BY-SIDE
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Detection Output")
            st.image(yolo_img, use_container_width=True)

        with c2:
            st.subheader("Safe Region Output")
            st.image(safe_img, use_container_width=True)

