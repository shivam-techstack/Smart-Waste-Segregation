"""Smart Waste Segregation - Streamlit app.

Run:  streamlit run app.py

"""
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from bins import get_bin

MODEL_PATH = os.environ.get(
    "WASTE_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
)
MAX_SIDE = 4000  # reject absurdly large images to keep the app responsive

st.set_page_config(page_title="Smart Waste Segregation", page_icon="♻️", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(path):
    """Load the YOLO weights once. Raises a clear error if the file is missing/incompatible."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model weights not found: {path}")
    from ultralytics import YOLO  # imported here so a missing install gives a clean message

    return YOLO(path)


def run_detection(model, image, conf):
    """Run YOLO on a PIL image. Returns (annotated RGB array, detections DataFrame, ms)."""
    start = time.perf_counter()
    # Pass the PIL image itself: Ultralytics treats raw numpy arrays as BGR, PIL as RGB.
    result = model.predict(image, conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        cls_name = model.names[int(box.cls)]
        rows.append(
            {
                "Class": cls_name,
                "Confidence (%)": round(float(box.conf) * 100, 1),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "Suggested bin": get_bin(cls_name)["bin"],
            }
        )
    df = pd.DataFrame(rows, columns=["Class", "Confidence (%)", "x1", "y1", "x2", "y2", "Suggested bin"])
    annotated = result.plot()[:, :, ::-1]  # plot() returns BGR; convert to RGB
    return annotated, df, elapsed_ms


def load_upload(uploaded):
    """Open an uploaded file as RGB. Returns (image, error message)."""
    try:
        img = Image.open(uploaded)
        img.load()
    except Exception:  # Ultralytics patches Image.open, so failures are not always PIL errors
        return None, "This file could not be read as an image. Please upload a valid JPG or PNG."
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        return None, f"Image is too large ({img.size[0]}x{img.size[1]}). Please use one under {MAX_SIDE}px per side."
    return img, None


# ---------------------------------------------------------------- UI
st.title("♻️ Smart Waste Segregation Using YOLO")
st.caption(
    "Detects Glass, Metal, Paper, Plastic and Waste in a photo with a YOLOv8-nano model, "
    "then suggests a bin for each item. "
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.info(
        "Confidence is the model's own score for a box. It is **not** the probability "
        "that the prediction is correct. Lower it to find more objects (more false alarms); "
        "raise it for fewer, safer detections. No retraining is needed."
    )
    st.divider()
    st.subheader("Session history")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history).fillna(0)
        st.dataframe(hist_df, hide_index=True, width="stretch")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No images analysed yet.")

try:
    model = load_model(MODEL_PATH)
except Exception as exc:  # missing file, corrupt or incompatible weights, missing package
    st.error(f"Could not load the model. {exc}")
    st.stop()

uploaded = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])
detect = st.button("Detect Waste", type="primary", disabled=uploaded is None)

if uploaded is not None:
    image, err = load_upload(uploaded)
    if err:
        st.error(err)
        st.stop()

    if detect:
        with st.spinner("Detecting waste..."):
            try:
                annotated, df, ms = run_detection(model, image, conf)
            except Exception as exc:
                st.error(f"Detection failed: {exc}")
                st.stop()
        st.session_state.last = (uploaded.name, image, annotated, df, ms, conf)
        counts = df["Class"].value_counts().to_dict()
        st.session_state.history.append(
            {"Time": datetime.now().strftime("%H:%M:%S"), "Image": uploaded.name, "Total": len(df), **counts}
        )
        st.rerun()

    last = st.session_state.get("last")
    if last and last[0] == uploaded.name:
        name, image, annotated, df, ms, used_conf = last
        left, right = st.columns(2)
        left.subheader("Original")
        left.image(image, width="stretch")
        right.subheader(f"Detections (threshold {used_conf:.2f})")
        right.image(annotated, width="stretch")
        st.caption(f"Inference time: {ms:.0f} ms (CPU timing varies; the first run is slower).")

        if df.empty:
            st.warning("No waste detected at this threshold. Try lowering the confidence slider and press Detect Waste again.")
        else:
            st.metric("Total objects detected", len(df))
            col_table, col_chart = st.columns([3, 2])
            with col_table:
                st.subheader("Detection table")
                st.dataframe(df, hide_index=True, width="stretch")
            with col_chart:
                st.subheader("Count by class")
                st.bar_chart(df["Class"].value_counts())

            st.subheader("Suggested bins")
            for cls_name, n in df["Class"].value_counts().items():
                rule = get_bin(cls_name)
                st.markdown(
                    f"<div style='border-left:6px solid {rule['color']};padding:6px 12px;margin:6px 0'>"
                    f"<b>{cls_name}</b> x {n} &rarr; <b>{rule['bin']}</b><br>"
                    f"<span style='opacity:.8'>{rule['note']}</span></div>",
                    unsafe_allow_html=True,
                )
            st.caption("Bin advice comes from simple rules in bins.py, separate from the model.")
    elif not detect:
        st.image(image, caption="Preview - press Detect Waste", width=400)
else:
    st.info("Upload a JPG or PNG image to begin.")
