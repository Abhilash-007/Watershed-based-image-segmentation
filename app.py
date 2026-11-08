import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from watershed_segmentation import make_demo_image, watershed_segment

st.set_page_config(page_title="Watershed Segmentation", layout="wide")
st.title("🌊 Watershed-Based Image Segmentation")
st.caption("Digital Image Processing mini project - marker-controlled watershed with OpenCV")

# ---------------- Sidebar ----------------
st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif"])
use_demo = st.sidebar.checkbox("Use demo image (touching coins)", value=uploaded is None)

st.sidebar.header("Parameters")
blur = st.sidebar.slider("Gaussian blur kernel", 1, 21, 5, step=2)
mode = st.sidebar.radio("Threshold", ["otsu", "manual"], horizontal=True)
manual_t = st.sidebar.slider("Manual threshold", 0, 255, 127, disabled=(mode == "otsu"))
invert = st.sidebar.checkbox("Invert (objects darker than background)", value=False)
morph_k = st.sidebar.slider("Morph kernel size", 1, 9, 3, step=2)
open_it = st.sidebar.slider("Opening iterations", 0, 6, 2)
bg_it = st.sidebar.slider("Background dilation iterations", 1, 8, 3)
fg_ratio = st.sidebar.slider("Sure-foreground ratio (× max distance)", 0.1, 0.9, 0.6, 0.05)
min_area = st.sidebar.slider("Minimum object area (px)", 0, 2000, 0, 10)

# ---------------- Load image ----------------
if uploaded is not None and not use_demo:
    image = np.array(Image.open(uploaded).convert("RGB"))
elif use_demo:
    image = make_demo_image()
else:
    st.info("Upload an image or tick 'Use demo image' in the sidebar.")
    st.stop()

# Limit size for speed
MAX_SIDE = 900
h, w = image.shape[:2]
if max(h, w) > MAX_SIDE:
    s = MAX_SIDE / max(h, w)
    image = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

# ---------------- Run ----------------
res = watershed_segment(
    image, blur_ksize=blur, threshold_mode=mode, manual_threshold=manual_t,
    invert=invert, morph_ksize=morph_k, open_iterations=open_it,
    bg_dilate_iterations=bg_it, fg_ratio=fg_ratio, min_area=min_area,
)

# ---------------- Results ----------------
c1, c2, c3 = st.columns(3)
c1.image(image, caption="Original", width="stretch")
c2.image(res.boundary_overlay, caption="Watershed boundaries (red)", width="stretch")
c3.image(res.blended, caption="Segmented regions", width="stretch")

m1, m2, m3 = st.columns(3)
m1.metric("Objects detected", res.num_objects)
m2.metric("Mean area (px)", f"{np.mean(res.areas):.0f}" if res.areas else "–")
m3.metric("Largest area (px)", max(res.areas) if res.areas else "–")

with st.expander("🔍 Show intermediate steps", expanded=True):
    steps = [
        ("1. Grayscale", res.gray),
        ("2. Threshold", res.binary),
        ("3. After opening", res.opened),
        ("4. Sure background", res.sure_bg),
        ("5. Distance transform", cv2.applyColorMap(res.dist_transform, cv2.COLORMAP_JET)[:, :, ::-1]),
        ("6. Sure foreground", res.sure_fg),
        ("7. Unknown region", res.unknown),
        ("8. Final labels", res.colored_segments),
    ]
    for row in range(0, len(steps), 4):
        cols = st.columns(4)
        for col, (title, img) in zip(cols, steps[row:row + 4]):
            col.image(img, caption=title, width="stretch", clamp=True)

if res.areas:
    with st.expander("📊 Object areas"):
        st.bar_chart(res.areas)

# ---------------- Download ----------------
buf = io.BytesIO()
Image.fromarray(res.blended).save(buf, format="PNG")
st.download_button("⬇️ Download segmented image", buf.getvalue(),
                   file_name="watershed_result.png", mime="image/png")

with st.expander("ℹ️ How it works"):
    st.markdown(
        """
        1. **Threshold** the blurred grayscale image (Otsu picks the level automatically).
        2. **Opening** removes small noise.
        3. **Sure background** = dilated mask; **sure foreground** = peaks of the *distance transform*.
        4. **Unknown** = background − foreground; the watershed decides these pixels.
        5. Each foreground blob becomes a **marker**; `cv2.watershed` floods from markers and
           the meeting lines (value −1) become object boundaries, splitting touching objects.
        """
    )
