import json
import cv2
import fitz  # PyMuPDF
import geojson
import numpy as np
from PIL import Image
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import snap, unary_union
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(
    page_title="Auto Map Vectorizer & Georeferencer",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Auto Map Vectorizer with GCP Georeferencing")
st.write(
    "Upload a Map PDF/JPG, click 3 Control Points to add Lat/Lon coordinates,"
    " snap boundaries, and export real-world GeoJSON."
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
  if not polygons_list:
    return []
  boundary_lines = unary_union(
      [poly.exterior for poly in polygons_list if poly.is_valid]
  )
  snapped_polygons = []
  for poly in polygons_list:
    if poly.is_valid:
      snapped = snap(poly, boundary_lines, tolerance=tolerance).buffer(0)
      if isinstance(snapped, MultiPolygon):
        snapped_polygons.extend(list(snapped.geoms))
      elif isinstance(snapped, Polygon) and not snapped.is_empty:
        snapped_polygons.append(snapped)
  return snapped_polygons


# Sidebar Controls
st.sidebar.header("⚙️ Processing Settings")
threshold_val = st.sidebar.slider("Line Sensitivity (Threshold)", 50, 250, 180)
min_area = st.sidebar.slider("Min Plot Area (Pixels)", 100, 10000, 800)
line_smooth = st.sidebar.slider(
    "Line Simplification (Epsilon %)", 0.001, 0.02, 0.004, 0.001
)
enable_snap = st.sidebar.checkbox("Enable Vertex Snapping", value=True)
snap_distance = st.sidebar.slider(
    "Snapping Distance (Pixels)", 1.0, 30.0, 6.0, 0.5
)

# Session state for 3 GCP points
if "gcp_points" not in st.session_state:
  st.session_state.gcp_points = []

uploaded_file = st.file_uploader(
    "Upload Map Image (JPG, PNG) or PDF", type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file is not None:
  img = load_image_from_upload(uploaded_file)
  h, w = img.shape[:2]

  st.markdown("---")
  st.subheader("📍 Step 1: Select 3 Ground Control Points (GCP)")
  st.caption(
      "Click anywhere on the map to set 3 reference points. Enter their real"
      " Longitude & Latitude below."
  )

  # Draw existing GCP points on interactive map
  gcp_preview = img.copy()
  for i, pt in enumerate(st.session_state.gcp_points):
    cv2.circle(gcp_preview, (int(pt["x"]), int(pt["y"])), 8, (0, 0, 255), -1)
    cv2.putText(
        gcp_preview,
        f"GCP {i+1}",
        (int(pt["x"]) + 10, int(pt["y"]) - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

  col_map, col_gcps = st.columns([2, 1])

  with col_map:
    # Click interaction
    value = streamlit_image_coordinates(
        Image.fromarray(cv2.cvtColor(gcp_preview, cv2.COLOR_BGR2RGB)),
        key="map_coord",
    )
    if value is not None:
      clicked_point = {"x": value["x"], "y": value["y"]}
      if (
          len(st.session_state.gcp_points) < 3
          and clicked_point not in st.session_state.gcp_points
      ):
        st.session_state.gcp_points.append(clicked_point)
        st.rerun()

  with col_gcps:
    st.markdown("**GCP Coordinate Inputs:**")
    gcp_coords = []
    for i in range(3):
      with st.expander(f"📌 Point {i+1}", expanded=(i == 0)):
        if i < len(st.session_state.gcp_points):
          pt = st.session_state.gcp_points[i]
          st.write(f"Pixel: `X={pt['x']}, Y={pt['y']}`")
          lon = st.number_input(
              f"P{i+1} Longitude (X)",
              value=76.000000 + (i * 0.001),
              format="%.6f",
              key=f"lon_{i}",
          )
          lat = st.number_input(
              f"P{i+1} Latitude (Y)",
              value=11.000000 + (i * 0.001),
              format="%.6f",
              key=f"lat_{i}",
          )
          gcp_coords.append({"px": pt["x"], "py": pt["y"], "lon": lon, "lat": lat})
        else:
          st.warning(f"Click on the map to select Point {i+1}")

    if st.button("🔄 Reset GCP Points"):
      st.session_state.gcp_points = []
      st.rerun()

  # ---------------- Vectorization & Transformation ----------------
  st.markdown("---")
  st.subheader("📐 Step 2: Auto-Traced Vector Output")

  # 1. Image Preprocessing & Contours
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(
      gray, threshold_val, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
  )
  kernel = np.ones((3, 3), np.uint8)
  closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
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

  final_polygons = (
      apply_snapping_to_polygons(raw_polygons, snap_distance)
      if enable_snap
      else raw_polygons
  )

  # Affine Georeferencing Matrix calculation
  use_georef = len(gcp_coords) == 3
  if use_georef:
    src_pts = np.float32([[p["px"], p["py"]] for p in gcp_coords])
    dst_pts = np.float32([[p["lon"], p["lat"]] for p in gcp_coords])
    # 2x3 Affine transform matrix
    affine_mat = cv2.getAffineTransform(src_pts, dst_pts)
    a, b, d = affine_mat[0]
    c, e, f_val = affine_mat[1]
    # Shapely transform matrix: [a, b, d, e, xoff, yoff]
    shapely_matrix = [a, b, c, e, d, f_val]

  features = []
  res_preview = img.copy()

  for idx, poly in enumerate(final_polygons, 1):
    int_coords = np.array(poly.exterior.coords, dtype=np.int32)
    cv2.polylines(
        res_preview, [int_coords], isClosed=True, color=(0, 255, 0), thickness=2
    )

    out_poly = (
        affine_transform(poly, shapely_matrix) if use_georef else poly
    )

    feature = geojson.Feature(
        geometry=mapping(out_poly),
        properties={
            "Plot_No": idx,
            "Area": (
                round(out_poly.area, 8) if use_georef else round(poly.area, 2)
            ),
            "CRS": "EPSG:4326 (WGS84)" if use_georef else "Pixel Coordinate",
        },
    )
    features.append(feature)

  geojson_str = geojson.dumps(geojson.FeatureCollection(features), indent=2)

  col_out1, col_out2 = st.columns(2)
  with col_out1:
    st.image(
        cv2.cvtColor(res_preview, cv2.COLOR_BGR2RGB), use_container_width=True
    )

  with col_out2:
    if use_georef:
      st.success("✅ Georeferenced using 3 GCPs (WGS84 Lat/Lon)!")
    else:
      st.info("ℹ️ Exporting in Pixel coordinates (Add 3 GCPs for Lat/Lon).")

    st.download_button(
        label="📥 Download GeoJSON",
        data=geojson_str,
        file_name="georeferenced_plots.geojson",
        mime="application/json",
    )
    with st.expander("View GeoJSON"):
      st.json(json.loads(geojson_str))
