import json
import cv2
import fitz  # PyMuPDF
import geojson
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon, mapping
import streamlit as st

st.set_page_config(
    page_title="Auto Map Tracer", page_icon="🗺️", layout="wide"
)

st.title("🗺️ Auto Map Vectorizer (Raster to Polygon)")
st.write("Upload a Map PDF/JPG to auto-trace boundaries and export GeoJSON.")


def load_image_from_upload(uploaded_file):
  if uploaded_file.type == "application/pdf":
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    page = doc.load_page(0)
    # 300 DPI Rendering for high precision
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  else:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# Sidebar Settings
st.sidebar.header("⚙️ Processing Controls")
threshold_val = st.sidebar.slider("Line Sensitivity (Threshold)", 50, 250, 180)
min_area = st.sidebar.slider("Min Plot Area (Pixels)", 100, 10000, 800)
line_smooth = st.sidebar.slider(
    "Line Simplification (Epsilon %)", 0.001, 0.02, 0.004, 0.001
)

uploaded_file = st.file_uploader(
    "Upload Map Image (JPG, PNG) or PDF", type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file is not None:
  img = load_image_from_upload(uploaded_file)
  h, w = img.shape[:2]

  # Preprocessing
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(
      gray, threshold_val, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )

  # Morphological Cleaning to bridge small line breaks
  kernel = np.ones((3, 3), np.uint8)
  closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

  # Find Contours
  contours, hierarchy = cv2.findContours(
      closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
  )

  features = []
  preview_img = img.copy()
  plot_count = 0

  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_area or area > (w * h * 0.95):
      continue

    epsilon = line_smooth * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) >= 3:
      plot_count += 1
      pts = [tuple(pt[0]) for pt in approx]
      pts.append(pts[0])

      poly = Polygon(pts)
      if poly.is_valid and not poly.is_empty:
        # Drawing bounding overlays on preview
        cv2.drawContours(preview_img, [approx], -1, (0, 255, 0), 2)

        # Centroid for Label
        M = cv2.moments(cnt)
        if M["m00"] != 0:
          cX = int(M["m10"] / M["m00"])
          cY = int(M["m01"] / M["m00"])
          cv2.putText(
              preview_img,
              str(plot_count),
              (cX, cY),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              (0, 0, 255),
              2,
          )

        feature = geojson.Feature(
            geometry=mapping(poly),
            properties={"Plot_No": plot_count, "Pixel_Area": area},
        )
        features.append(feature)

  feature_collection = geojson.FeatureCollection(features)
  geojson_str = geojson.dumps(feature_collection, indent=2)

  # Display Results
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Original & Vector Overlay")
    st.image(
        cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), use_container_width=True
    )

  with col2:
    st.subheader("Extracted Details")
    st.success(f"Detected **{len(features)}** closed polygon plots!")

    st.download_button(
        label="📥 Download Vector (GeoJSON)",
        data=geojson_str,
        file_name="traced_polygons.geojson",
        mime="application/json",
    )

    with st.expander("View GeoJSON Payload"):
      st.json(json.loads(geojson_str))
