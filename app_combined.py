import os
import sys
import gc
import time
import logging
import importlib.util
import asyncio
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import zipfile
import tempfile
import shutil

# Initialize asyncio event loop for Streamlit
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Configure Streamlit to disable file watcher that causes PyTorch issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Configure CUDA and PyTorch environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU if available

# Import torch first and configure it
import torch
torch.set_grad_enabled(False)  # Disable gradients globally for inference
torch._C._jit_set_profiling_mode(False)  # Disable JIT profiling
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# Rest of imports
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pydicom
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import regionprops, label
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KERNEL_PATH = os.path.join(CURRENT_DIR, 'dose_kernel.npy')
QRADPLAN_SCRIPT_PATH = os.path.join(CURRENT_DIR, "qradplan-may27-no-quantum.py")
TUMOR_DETECTION_SCRIPT_PATH = os.path.join(CURRENT_DIR, "tumor_detection_enhanced.py") # Using enhanced
TUMOR_MODEL_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "tumor_detector.pth") # Assuming weights file

# --- Configure Streamlit ---
st.set_page_config(
    page_title="Comprehensive RT Planning Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.config.set_option('server.maxMessageSize', 1000) # 1000MB limit

# --- Import Custom Modules ---
@st.cache_resource
def get_planning_and_detection_modules():
    """Load QRadPlan3D and TumorDetector modules with caching."""
    import os # Add import os here
    try:
        # Import QRadPlan3D
        qradplan_spec = importlib.util.spec_from_file_location("qradplan", QRADPLAN_SCRIPT_PATH)
        qradplan_module = importlib.util.module_from_spec(qradplan_spec)
        sys.modules["qradplan"] = qradplan_module
        qradplan_spec.loader.exec_module(qradplan_module)
        QRadPlan3D_cls = qradplan_module.QRadPlan3D

        # Import TumorDetector
        detection_spec = importlib.util.spec_from_file_location("tumor_detection", TUMOR_DETECTION_SCRIPT_PATH)
        detection_module = importlib.util.module_from_spec(detection_spec)
        sys.modules["tumor_detection"] = detection_module
        detection_spec.loader.exec_module(detection_module)
        TumorDetector_cls = detection_module.EnhancedTumorDetector # Use EnhancedTumorDetector

        # Initialize detector (can be moved if model path is dynamic)
        if os.path.exists(TUMOR_MODEL_WEIGHTS_PATH):
            detector_instance = TumorDetector_cls(model_weights_path=TUMOR_MODEL_WEIGHTS_PATH)
        else:
            st.warning(f"Tumor detector weights not found at {TUMOR_MODEL_WEIGHTS_PATH}. Detection might be suboptimal.")
            detector_instance = TumorDetector_cls() # Initialize without weights if not found

        return QRadPlan3D_cls, detector_instance
    except Exception as e:
        st.error(f"Error loading custom modules: {str(e)}")
        return None, None

QRadPlan3D, tumor_detector_instance = get_planning_and_detection_modules()

if not QRadPlan3D or not tumor_detector_instance:
    st.error("Failed to load core components. Application cannot start.")
    st.stop()


# --- DICOMViewer Class (adapted from starviewer_rt.py) ---
class DICOMViewer:
    def __init__(self):
        self.volume_data: Optional[np.ndarray] = None
        self.metadata: Optional[Dict] = None
        self.spacing: Optional[Tuple[float, float, float]] = None
        self.origin: Optional[List[float]] = None
        self.dicom_files_info: List[Dict] = [] # To store path and instance number

    def find_dicom_files(self, directory: str) -> List[str]:
        dicom_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        return dicom_files

    def load_dicom_series(self, directory_path: str) -> bool:
        try:
            dicom_files_paths = self.find_dicom_files(directory_path)
            if not dicom_files_paths:
                logger.error("No DICOM files found in directory")
                return False

            self.dicom_files_info = []
            slices_data = []
            for file_path in dicom_files_paths:
                try:
                    ds = pydicom.dcmread(file_path)
                    if hasattr(ds, 'pixel_array') and hasattr(ds, 'InstanceNumber'):
                         # Check for PhotometricInterpretation to handle MONOCHROME1
                        pixel_array = ds.pixel_array
                        if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == "MONOCHROME1":
                            # Invert MONOCHROME1 for consistent display (optional, depends on preference)
                            # pixel_array = np.invert(pixel_array) # Or use ds.WindowCenter/Width with LUT
                            pass # Keep as is, window/level will handle
                        
                        slices_data.append({
                            'data': pixel_array.astype(np.float32),
                            'instance_number': int(ds.InstanceNumber),
                            'slice_location': float(ds.SliceLocation) if hasattr(ds, 'SliceLocation') else int(ds.InstanceNumber),
                            'dataset': ds # Store dataset for metadata
                        })
                except Exception as e:
                    logger.warning(f"Could not read or process DICOM file {file_path}: {e}")
                    continue
            
            if not slices_data:
                logger.error("No valid DICOM slices could be processed.")
                return False

            # Sort slices by slice location (more robust) or instance number
            slices_data.sort(key=lambda s: s['slice_location'])
            
            first_slice_ds = slices_data[0]['dataset']
            self.metadata = self._extract_metadata(first_slice_ds)
            
            rows = first_slice_ds.Rows
            cols = first_slice_ds.Columns
            num_slices = len(slices_data)

            self.volume_data = np.zeros((rows, cols, num_slices), dtype=np.float32)
            for i, slice_info in enumerate(slices_data):
                self.volume_data[:, :, i] = slice_info['data']
            
            # Transpose to (slices, rows, cols) for easier iteration if preferred,
            # but Plotly Volume expects (x,y,z) which can map to (cols, rows, slices_idx)
            # For now, keep as (rows, cols, slices) and adjust access if needed.
            # Or, more consistently: (X, Y, Z) -> (cols, rows, slices_idx)
            # Let's re-orient to (cols, rows, slices_idx) for consistency with QRadPlan3D
            self.volume_data = self.volume_data.transpose(1, 0, 2) # Now (cols, rows, slices_idx)


            ps = first_slice_ds.PixelSpacing if hasattr(first_slice_ds, 'PixelSpacing') else [1.0, 1.0]
            st = first_slice_ds.SliceThickness if hasattr(first_slice_ds, 'SliceThickness') else 1.0
            self.spacing = (float(ps[0]), float(ps[1]), float(st)) # (x, y, z spacing)
            self.origin = first_slice_ds.ImagePositionPatient if hasattr(first_slice_ds, 'ImagePositionPatient') else [0.0, 0.0, 0.0]
            
            logger.info(f"Loaded DICOM series: {self.volume_data.shape}, Spacing: {self.spacing}")
            return True
        except Exception as e:
            logger.error(f"Error loading DICOM series: {str(e)}")
            self.volume_data = None
            return False

    def _extract_metadata(self, ds: pydicom.Dataset) -> Dict:
        return {
            'Patient Name': str(ds.PatientName) if hasattr(ds, 'PatientName') else "N/A",
            'Patient ID': str(ds.PatientID) if hasattr(ds, 'PatientID') else "N/A",
            'Study Date': str(ds.StudyDate) if hasattr(ds, 'StudyDate') else "N/A",
            'Modality': str(ds.Modality) if hasattr(ds, 'Modality') else "N/A",
            'Rows': ds.Rows if hasattr(ds, 'Rows') else "N/A",
            'Columns': ds.Columns if hasattr(ds, 'Columns') else "N/A",
            'Pixel Spacing': ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else "N/A",
            'Slice Thickness': ds.SliceThickness if hasattr(ds, 'SliceThickness') else "N/A",
            'Image Position (Patient)': ds.ImagePositionPatient if hasattr(ds, 'ImagePositionPatient') else "N/A",
            'Image Orientation (Patient)': ds.ImageOrientationPatient if hasattr(ds, 'ImageOrientationPatient') else "N/A",
            'Window Center': ds.WindowCenter if hasattr(ds, 'WindowCenter') else 40, # Default WC
            'Window Width': ds.WindowWidth if hasattr(ds, 'WindowWidth') else 400,    # Default WW
        }

    def get_slice_display_data(self, slice_idx: int, window_center: float, window_width: float) -> Optional[np.ndarray]:
        if self.volume_data is None or not (0 <= slice_idx < self.volume_data.shape[2]):
            return None
        
        # volume_data is (cols, rows, slice_idx)
        slice_data = self.volume_data[:, :, slice_idx].T # Transpose to (rows, cols) for imshow
        
        # Apply window/level
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        display_slice = np.clip(slice_data, min_val, max_val)
        display_slice = (display_slice - min_val) / (max_val - min_val + 1e-9) # Normalize to 0-1
        display_slice = (display_slice * 255).astype(np.uint8) # Scale to 0-255 for display
        return display_slice

    def get_volume_for_tumor_detection(self) -> Optional[np.ndarray]:
        """Returns the volume in (depth, height, width) or (z,y,x) for detection"""
        if self.volume_data is None:
            return None
        # self.volume_data is (cols, rows, slices_idx) -> (x, y, z)
        # Tumor detector might expect (z, y, x) or (depth, height, width)
        # Let's transpose to (slices_idx, rows, cols)
        return self.volume_data.transpose(2, 1, 0)


# --- TreatmentPlanner Class ---
class TreatmentPlanner:
    def __init__(self):
        self.q_rad_plan: Optional[QRadPlan3D] = None
        self.beam_weights: Optional[np.ndarray] = None
        self.dose_distribution: Optional[np.ndarray] = None
        self.plan_metrics: Optional[Dict] = None

    def initialize_planner(self, planner_params: Dict):
        try:
            self.q_rad_plan = QRadPlan3D(**planner_params)
            logger.info("Treatment planner initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Error initializing QRadPlan3D: {e}")
            st.error(f"Planner Initialization Error: {e}")
            self.q_rad_plan = None
            return False

    def set_tumor_data(self, tumor_mask_input: np.ndarray):
        if not self.q_rad_plan:
            st.error("Planner not initialized.")
            return False
        try:
            return self.q_rad_plan.set_tumor_data(tumor_mask_input=tumor_mask_input)
        except Exception as e:
            logger.error(f"Error setting tumor data in planner: {e}")
            st.error(f"Error setting tumor data: {e}")
            return False
            
    def optimize_and_calculate_dose(self):
        if not self.q_rad_plan:
            st.error("Planner not initialized.")
            return False
        try:
            with st.spinner("Optimizing beam weights..."):
                self.beam_weights = self.q_rad_plan.optimize_beams()
            st.success("Beam weights optimized.")
            
            with st.spinner("Calculating dose distribution..."):
                self.dose_distribution = self.q_rad_plan.calculate_dose(self.beam_weights)
            st.success("Dose distribution calculated.")

            with st.spinner("Calculating plan metrics..."):
                self.plan_metrics = self.q_rad_plan.calculate_plan_metrics(self.beam_weights)
            st.success("Plan metrics calculated.")
            return True
        except Exception as e:
            logger.error(f"Error during optimization or dose calculation: {e}")
            st.error(f"Optimization/Dose Calculation Error: {e}")
            return False

    def get_dvh_data(self):
        if not self.q_rad_plan or self.dose_distribution is None:
            st.warning("Dose distribution not available for DVH.")
            return None
        return self.q_rad_plan.get_dvh_data(self.dose_distribution)

# --- Helper Functions ---
def process_uploaded_files(uploaded_files) -> Optional[str]:
    """Process uploaded DICOM files (single, multiple, or ZIP) into a temporary directory."""
    if not uploaded_files:
        return None

    temp_dir = tempfile.mkdtemp()
    
    if isinstance(uploaded_files, list): # Multiple files
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.dcm'):
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            elif uploaded_file.name.lower().endswith('.zip'):
                 with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
        return temp_dir
    else: # Single file
        if uploaded_files.name.lower().endswith('.dcm'):
            with open(os.path.join(temp_dir, uploaded_files.name), "wb") as f:
                f.write(uploaded_files.getbuffer())
            return temp_dir
        elif uploaded_files.name.lower().endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            return temp_dir
    return None


# --- Streamlit UI ---
def main():
    # Initialize session state variables
    if 'viewer' not in st.session_state:
        st.session_state.viewer = DICOMViewer()
    if 'planner' not in st.session_state:
        st.session_state.planner = TreatmentPlanner()
    if 'dicom_load_path' not in st.session_state:
        st.session_state.dicom_load_path = None
    if 'temp_dir_path' not in st.session_state:
        st.session_state.temp_dir_path = None
    if 'current_slice_idx' not in st.session_state:
        st.session_state.current_slice_idx = 0
    if 'window_center' not in st.session_state:
        st.session_state.window_center = 40  # Default for CT soft tissue
    if 'window_width' not in st.session_state:
        st.session_state.window_width = 400 # Default for CT soft tissue
    if 'tumor_mask_display' not in st.session_state: # For display on 2D/3D views
        st.session_state.tumor_mask_display = None
    if 'detected_tumor_properties' not in st.session_state:
        st.session_state.detected_tumor_properties = None
    if 'rtstruct_path_input' not in st.session_state:
        st.session_state.rtstruct_path_input = ""
    if 'ct_path_input' not in st.session_state: # For static CT if not using 4D
        st.session_state.ct_path_input = ""
    if 'fourd_ct_path_input' not in st.session_state:
        st.session_state.fourd_ct_path_input = ""


    st.title("StarViewer RT - Enhanced Radiotherapy Planning")

    # --- Sidebar ---
    with st.sidebar:
        st.header("File Operations")
        dicom_files_uploaded = st.file_uploader(
            "Upload DICOM Series (DICOM files or ZIP archive)",
            type=['dcm', 'zip'],
            accept_multiple_files=True
        )

        if dicom_files_uploaded:
            if st.session_state.temp_dir_path and os.path.exists(st.session_state.temp_dir_path):
                try:
                    shutil.rmtree(st.session_state.temp_dir_path)
                except Exception as e:
                    logger.warning(f"Could not remove previous temp dir: {e}")
            
            st.session_state.temp_dir_path = process_uploaded_files(dicom_files_uploaded)
            if st.session_state.temp_dir_path:
                st.session_state.dicom_load_path = st.session_state.temp_dir_path
                st.success(f"Files processed into: {st.session_state.temp_dir_path}")
            else:
                st.error("Failed to process uploaded files.")
                st.session_state.dicom_load_path = None


        if st.button("Load DICOM Series from Processed Files"):
            if st.session_state.dicom_load_path:
                with st.spinner("Loading DICOM series..."):
                    if st.session_state.viewer.load_dicom_series(st.session_state.dicom_load_path):
                        st.success("DICOM series loaded.")
                        # Set initial window/level from DICOM if available
                        if st.session_state.viewer.metadata:
                            wc = st.session_state.viewer.metadata.get('Window Center', 40)
                            ww = st.session_state.viewer.metadata.get('Window Width', 400)
                            # DICOM can have multiple WC/WW values, take the first if list
                            st.session_state.window_center = wc[0] if isinstance(wc, list) else wc
                            st.session_state.window_width = ww[0] if isinstance(ww, list) else ww
                        st.session_state.current_slice_idx = st.session_state.viewer.volume_data.shape[2] // 2 if st.session_state.viewer.volume_data is not None else 0
                        st.session_state.tumor_mask_display = None # Reset tumor mask on new series load
                        st.session_state.detected_tumor_properties = None
                    else:
                        st.error("Failed to load DICOM series from the processed files.")
            else:
                st.warning("Please upload and process files first.")

        if st.session_state.viewer.volume_data is not None:
            st.header("Display Controls")
            st.session_state.window_center = st.slider(
                "Window Center", -1000, 3000, int(st.session_state.window_center), 1
            )
            st.session_state.window_width = st.slider(
                "Window Width", 1, 5000, int(st.session_state.window_width), 1
            )
        
        st.header("Planning Setup")
        st.session_state.rtstruct_path_input = st.text_input("RTStruct File Path (Optional)", st.session_state.rtstruct_path_input)
        st.session_state.ct_path_input = st.text_input("Static CT Directory Path (Optional, if not using loaded series for planning)", st.session_state.ct_path_input if st.session_state.ct_path_input else (st.session_state.dicom_load_path or ""))
        st.session_state.fourd_ct_path_input = st.text_input("4D CT Directory Path (Optional)", st.session_state.fourd_ct_path_input)
        
        motion_compensation_enabled = st.checkbox("Enable Motion Compensation (Requires 4D CT)", value=False)
        dir_method_options = ['simplified_sinusoidal', 'none'] # Add 'external_sitk' when ready
        dir_method_selected = st.selectbox("Deformable Registration Method", dir_method_options)


    # --- Main Area Tabs ---
    tab_dicom, tab_tumor, tab_planning, tab_results = st.tabs([
        "DICOM Viewer", "Tumor Detection", "Treatment Planning", "Results & Analysis"
    ])

    with tab_dicom:
        st.header("DICOM Viewer")
        if st.session_state.viewer.volume_data is not None:
            num_slices = st.session_state.viewer.volume_data.shape[2]
            st.session_state.current_slice_idx = st.slider(
                "Slice", 0, num_slices - 1, st.session_state.current_slice_idx
            )

            display_slice_data = st.session_state.viewer.get_slice_display_data(
                st.session_state.current_slice_idx,
                st.session_state.window_center,
                st.session_state.window_width
            )
            
            col1, col2 = st.columns([2,1])
            with col1:
                if display_slice_data is not None:
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(111)
                    ax.imshow(display_slice_data, cmap='gray')
                    ax.set_title(f"Slice {st.session_state.current_slice_idx + 1}/{num_slices}")
                    ax.axis('off')

                    if st.session_state.tumor_mask_display is not None:
                        # Ensure tumor_mask_display is (cols, rows, slices_idx) then take current slice and transpose
                        if st.session_state.tumor_mask_display.shape[2] > st.session_state.current_slice_idx:
                            tumor_overlay_slice = st.session_state.tumor_mask_display[:,:,st.session_state.current_slice_idx].T
                            ax.imshow(np.ma.masked_where(tumor_overlay_slice == 0, tumor_overlay_slice), cmap='Reds', alpha=0.4)
                    st.pyplot(fig)
                else:
                    st.info("Slice data not available.")
            
            with col2:
                st.subheader("DICOM Metadata")
                if st.session_state.viewer.metadata:
                    for key, value in st.session_state.viewer.metadata.items():
                        st.text(f"{key}: {value}")
                else:
                    st.text("No metadata available.")

            st.subheader("3D Volume View")
            if st.button("Render 3D Volume"):
                with st.spinner("Rendering 3D volume..."):
                    # Plotly Volume expects x, y, z to be 1D arrays defining grid, value to be flattened 3D array
                    vol_data = st.session_state.viewer.volume_data # (cols, rows, slices_idx)
                    
                    # Normalize for Plotly Volume display
                    vol_display = (vol_data - np.min(vol_data)) / (np.max(vol_data) - np.min(vol_data) + 1e-9)

                    x_coords = np.arange(vol_data.shape[0]) * st.session_state.viewer.spacing[0]
                    y_coords = np.arange(vol_data.shape[1]) * st.session_state.viewer.spacing[1]
                    z_coords = np.arange(vol_data.shape[2]) * st.session_state.viewer.spacing[2]

                    fig_3d = go.Figure(data=go.Volume(
                        x=x_coords, y=y_coords, z=z_coords,
                        value=vol_display.flatten('F'), # Flatten in Fortran order
                        isomin=0.15,  isomax=0.85, # Adjust these for better visualization
                        opacity=0.1,
                        surface_count=20,
                        colorscale='Gray',
                        caps=dict(x_show=False, y_show=False, z_show=False) # Hide caps
                    ))
                    fig_3d.update_layout(scene_aspectmode='data')
                    st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("Load a DICOM series to view slices and 3D volume.")

    with tab_tumor:
        st.header("Tumor Detection")
        if st.session_state.viewer.volume_data is None:
            st.info("Please load a DICOM series first.")
        else:
            if st.button("Detect Tumor"):
                with st.spinner("Detecting tumor..."):
                    # Use the middle slice of the volume for 2D detection as an example
                    # Or adapt EnhancedTumorDetector for 3D volumes if it supports it
                    # For now, using middle slice of the (cols, rows, slices_idx) volume,
                    # transposing it to (rows, cols) for the detector.
                    
                    # The EnhancedTumorDetector expects a 2D slice.
                    # Let's use the currently displayed slice for detection.
                    current_slice_for_detection_raw = st.session_state.viewer.volume_data[:,:,st.session_state.current_slice_idx]
                    
                    # The detector preprocesses, so pass the raw-ish slice data
                    detection_results = tumor_detector_instance.detect_tumors(current_slice_for_detection_raw)
                    
                    if detection_results and detection_results['mask'] is not None:
                        st.success("Tumor detection complete.")
                        st.session_state.detected_tumor_properties = detection_results
                        
                        # Create a 3D mask for display, placing the 2D detected mask on the current slice
                        full_3d_mask = np.zeros_like(st.session_state.viewer.volume_data, dtype=bool)
                        # detection_results['mask'] is (rows, cols), need to transpose back for (cols, rows)
                        full_3d_mask[:, :, st.session_state.current_slice_idx] = detection_results['mask'].T > 0.5
                        st.session_state.tumor_mask_display = full_3d_mask
                    else:
                        st.error("Tumor detection failed or no tumor found.")
                        st.session_state.tumor_mask_display = None
                        st.session_state.detected_tumor_properties = None
            
            if st.session_state.detected_tumor_properties:
                st.subheader("Detection Results")
                props = st.session_state.detected_tumor_properties
                st.write(f"Confidence: {props.get('confidence', 0):.2%}")
                st.write(f"Approx. Tumor Size (Volume from 2D mask): {props.get('tumor_size', 0):.2f} (units depend on pixel_spacing)")
                st.write(f"Number of distinct tumor regions found: {props.get('num_tumors',0)}")
                if props.get('tumor_properties'):
                    st.write("Properties of largest region:")
                    for key, val in props['tumor_properties'][0].items():
                        st.write(f"  {key}: {val}")

                # Display the detected mask on the current slice
                display_slice_data = st.session_state.viewer.get_slice_display_data(
                    st.session_state.current_slice_idx,
                    st.session_state.window_center,
                    st.session_state.window_width
                )
                if display_slice_data is not None and props['mask'] is not None:
                    fig_det = plt.figure(figsize=(8, 8))
                    ax_det = fig_det.add_subplot(111)
                    ax_det.imshow(display_slice_data, cmap='gray')
                    ax_det.imshow(np.ma.masked_where(props['mask'] == 0, props['mask']), cmap='Reds', alpha=0.5)
                    ax_det.set_title(f"Detected Tumor on Slice {st.session_state.current_slice_idx + 1}")
                    ax_det.axis('off')
                    st.pyplot(fig_det)


    with tab_planning:
        st.header("Treatment Planning")
        if st.session_state.viewer.volume_data is None:
            st.info("Load DICOM and detect tumor first.")
        elif st.session_state.tumor_mask_display is None : # Use the 3D mask for planning
            st.info("Please run tumor detection first.")
        else:
            st.subheader("Planning Parameters")
            num_beams_planner = st.slider("Number of Beams for Plan", 4, 16, 8, 2, key="planner_num_beams")
            
            # DICOM paths for QRadPlan3D initialization
            # If 4D CT path is provided, it takes precedence for density/masks
            # Otherwise, if static CT path is provided, it's used
            # Otherwise, if a DICOM series was loaded in the viewer, its path is used (if applicable for QRadPlan3D)
            # Otherwise, QRadPlan3D will use its simplified model.

            ct_for_planner = st.session_state.ct_path_input
            if not ct_for_planner and st.session_state.dicom_load_path and not st.session_state.fourd_ct_path_input:
                # If user loaded a series via UI and no specific static/4D path is given for planner
                ct_for_planner = st.session_state.dicom_load_path 
                st.caption(f"Using loaded DICOM series from viewer as CT input for planner: {ct_for_planner}")


            if st.button("Initialize Treatment Planner"):
                planner_grid_size = st.session_state.viewer.volume_data.shape # (cols, rows, slices)
                
                planner_params = {
                    "grid_size": planner_grid_size,
                    "num_beams": num_beams_planner,
                    "kernel_path": DEFAULT_KERNEL_PATH,
                    "dicom_rt_struct_path": st.session_state.rtstruct_path_input if st.session_state.rtstruct_path_input else None,
                    "ct_path": ct_for_planner if not motion_compensation_enabled and ct_for_planner else None,
                    "fourd_ct_path": st.session_state.fourd_ct_path_input if motion_compensation_enabled and st.session_state.fourd_ct_path_input else None,
                    "dir_method": dir_method_selected
                }
                if st.session_state.planner.initialize_planner(planner_params):
                    # Pass the 3D tumor mask to the planner
                    # The tumor_mask_display is (cols, rows, slices_idx)
                    if not st.session_state.planner.set_tumor_data(tumor_mask_input=st.session_state.tumor_mask_display):
                         st.error("Failed to set tumor data in planner.")
                else:
                    st.error("Failed to initialize treatment planner.")

            if st.session_state.planner.q_rad_plan is not None:
                st.success("Treatment Planner is active and configured.")
                if st.button("Optimize Plan & Calculate Dose"):
                    if st.session_state.planner.optimize_and_calculate_dose():
                        st.success("Plan optimized and dose calculated.")
                    else:
                        st.error("Failed to optimize plan or calculate dose.")
            else:
                st.info("Initialize planner to proceed.")


    with tab_results:
        st.header("Results and Analysis")
        if st.session_state.planner.q_rad_plan and st.session_state.planner.dose_distribution is not None:
            st.subheader("Dose Distribution")
            
            dose_map_to_display = st.session_state.planner.dose_distribution
            # Display a slice of the dose
            slice_idx_dose = st.slider("View Dose Slice (Z-axis)", 0, dose_map_to_display.shape[2] - 1, dose_map_to_display.shape[2] // 2, key="dose_slice_slider")
            
            fig_dose_res, ax_dose_res = plt.subplots(figsize=(8,8))
            # Dose map is (cols, rows, slices_idx), imshow expects (rows, cols)
            im = ax_dose_res.imshow(dose_map_to_display[:, :, slice_idx_dose].T, cmap='jet', origin='lower')
            plt.colorbar(im, ax=ax_dose_res, label="Dose (Gy)")
            ax_dose_res.set_title(f"Calculated Dose Distribution (Slice {slice_idx_dose})")
            ax_dose_res.set_xlabel("X-voxel")
            ax_dose_res.set_ylabel("Y-voxel")

            # Overlay tumor contour if available
            if st.session_state.tumor_mask_display is not None:
                 tumor_contour_slice = st.session_state.tumor_mask_display[:, :, slice_idx_dose].T
                 ax_dose_res.contour(tumor_contour_slice, colors='w', linewidths=0.8, origin='lower')
            st.pyplot(fig_dose_res)

            st.subheader("Plan Metrics")
            if st.session_state.planner.plan_metrics:
                st.json(st.session_state.planner.plan_metrics)
            else:
                st.info("Metrics not calculated yet.")

            st.subheader("Dose Volume Histogram (DVH)")
            dvh_data_df = st.session_state.planner.get_dvh_data()
            if dvh_data_df is not None and not dvh_data_df.empty:
                fig_dvh = go.Figure()
                for col in dvh_data_df.columns:
                    if col != 'Dose (Gy)':
                        fig_dvh.add_trace(go.Scatter(x=dvh_data_df['Dose (Gy)'], y=dvh_data_df[col], mode='lines', name=col.replace(" (%)","")))
                fig_dvh.update_layout(
                    title="Dose Volume Histogram",
                    xaxis_title="Dose (Gy)",
                    yaxis_title="Volume (%)",
                    legend_title="Structure"
                )
                st.plotly_chart(fig_dvh, use_container_width=True)
            else:
                st.info("DVH data not available.")
        else:
            st.info("Generate a treatment plan to view results.")
            
    # Cleanup temp directory on exit or reload
    # This is tricky with Streamlit's execution model.
    # A more robust way would be to clean up at the start of a new upload.

if __name__ == "__main__":
    main()
