import json
import cv2
import fitz  # PyMuPDF
import geojson
import numpy as np
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import snap, unary_union
import streamlit as st

st.set_page_config(
    page_title="Auto Map Tracer with Snapping", page_icon="🗺️", layout="wide"
)

st.title("🗺️ Auto Map Vectorizer & Snapping Tool")
st.write(
    "Upload a Map PDF/JPG to auto-trace boundaries with vertex snapping and"
    " export clean GeoJSON."
)


def load_image_from_upload(uploaded_file):
  if uploaded_file.type == "application/pdf":
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  else:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def apply_snapping_to_polygons(polygons_list, tolerance=5.0):
  """അടുത്തടുത്ത വെർട്ടിക്സുകൾ കൂട്ടിമുട്ടിക്കാൻ Shapely snap & union ഉപയോഗിക്കുന്നു"""
  if not polygons_list:
    return []

  # എല്ലാ അതിരുകളുടെയും യൂണിയൻ എടുത്ത് റഫറൻസ് ലൈൻ നെറ്റ്വർക്ക് ഉണ്ടാക്കുന്നു
  boundary_lines = unary_union(
      [poly.exterior for poly in polygons_list if poly.is_valid]
  )

  snapped_polygons = []
  for poly in polygons_list:
    if poly.is_valid:
      # ടോളറൻസ് പരിധിയിലുള്ള പോയിന്റുകൾ അടുത്ത ലൈനുകളിലേക്ക് സ്നാപ്പ് ചെയ്യുന്നു
      snapped = snap(poly, boundary_lines, tolerance=tolerance)
      # ചെറിയ വിടവുകൾ നികത്തി ടോപ്പോളജി കൃത്യമാക്കുന്നു
      snapped = snapped.buffer(0)
      if isinstance(snapped, MultiPolygon):
        snapped_polygons.extend(list(snapped.geoms))
      elif isinstance(snapped, Polygon) and not snapped.is_empty:
        snapped_polygons.append(snapped)

  return snapped_polygons


# Sidebar Controls
st.sidebar.header("⚙️ Processing & Snap Controls")
threshold_val = st.sidebar.slider("Line Sensitivity (Threshold)", 50, 250, 180)
min_area = st.sidebar.slider("Min Plot Area (Pixels)", 100, 10000, 800)
line_smooth = st.sidebar.slider(
    "Line Simplification (Epsilon %)", 0.001, 0.02, 0.004, 0.001
)

st.sidebar.markdown("---")
enable_snap = st.sidebar.checkbox("Enable Vertex Snapping", value=True)
snap_distance = st.sidebar.slider(
    "Snapping Distance (Pixels)", 1.0, 30.0, 6.0, 0.5
)

uploaded_file = st.file_uploader(
    "Upload Map Image (JPG, PNG) or PDF", type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file is not None:
  img = load_image_from_upload(uploaded_file)
  h, w = img.shape[:2]

  # 1. Image Preprocessing
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(
      gray, threshold_val, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )

  # Small gaps bridge cheyyan Morphological close
  kernel = np.ones((3, 3), np.uint8)
  closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

  # 2. Contours Detection
  contours, _ = cv2.findContours(
      closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
  )

  raw_polygons = []
  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_area or area > (w * h * 0.95):
      continue

    epsilon = line_smooth * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) >= 3:
      pts = [tuple(pt[0]) for pt in approx]
      pts.append(pts[0])
      poly = Polygon(pts)
      if poly.is_valid and not poly.is_empty:
        raw_polygons.append(poly)

  # 3. Apply Snapping
  final_polygons = (
      apply_snapping_to_polygons(raw_polygons, snap_distance)
      if enable_snap
      else raw_polygons
  )

  # 4. Draw preview & create GeoJSON
  preview_img = img.copy()
  features = []

  for idx, poly in enumerate(final_polygons, 1):
    int_coords = np.array(poly.exterior.coords, dtype=np.int32)
    # Bounding line വരയ്ക്കുന്നു
    cv2.polylines(
        preview_img, [int_coords], isClosed=True, color=(0, 255, 0), thickness=2
    )

    # ലേബൽ നൽകാൻ സെന്റർ പോയിന്റ് എടുക്കുന്നു
    centroid = poly.centroid
    cX, cY = int(centroid.x), int(centroid.y)
    cv2.putText(
        preview_img,
        str(idx),
        (cX, cY),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    feature = geojson.Feature(
        geometry=mapping(poly),
        properties={"Plot_No": idx, "Pixel_Area": round(poly.area, 2)},
    )
    features.append(feature)

  geojson_str = geojson.dumps(geojson.FeatureCollection(features), indent=2)

  # UI Display
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Traced & Snapped Overlay")
    st.image(
        cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), use_container_width=True
    )

  with col2:
    st.subheader("Output Details")
    st.success(f"Generated **{len(features)}** clean closed polygons!")
    if enable_snap:
      st.info(f"⚡ Snapping Applied: {snap_distance} px tolerance")

    st.download_button(
        label="📥 Download Snapped GeoJSON",
        data=geojson_str,
        file_name="snapped_map_polygons.geojson",
        mime="application/json",
    )

    with st.expander("View GeoJSON"):
      st.json(json.loads(geojson_str))
