"""Face recognition service using InsightFace (ArcFace) for high accuracy."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("onnxruntime not available. Install with: pip install onnxruntime")

# Set matplotlib backend before importing insightface to avoid conflicts
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

try:
    # Comprehensive workaround for matplotlib/insightface import issues
    import sys
    import subprocess
    
    # Save original state
    original_path = sys.path[:]
    original_modules = set(sys.modules.keys())
    
    # Strategy: Use subprocess to test import first, then import in isolated way
    user_site_packages = [p for p in sys.path if 'site-packages' in p and 'dist-packages' not in p]
    if not user_site_packages:
        # Fallback: use common user site location
        import site
        user_site_packages = [site.getusersitepackages()] if hasattr(site, 'getusersitepackages') else []
    
    # Ensure all required dependencies are available (required by matplotlib/insightface)
    required_deps = {
        'cycler': 'cycler',
        'dateutil': 'python-dateutil', 
        'six': 'six',
        'kiwisolver': 'kiwisolver',
        'pyparsing': 'pyparsing',
        'fonttools': 'fonttools'
    }
    missing_deps = []
    
    # Check each dependency
    for import_name, package_name in required_deps.items():
        try:
            if import_name == 'dateutil':
                import dateutil
            else:
                __import__(import_name)
        except ImportError:
            missing_deps.append(package_name)
    
    # Install missing dependencies to user site
    if missing_deps:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user'] + missing_deps,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            # Reload modules after installation
            for dep in missing_deps:
                if dep == 'python-dateutil':
                    try:
                        import importlib
                        if 'dateutil' in sys.modules:
                            importlib.reload(sys.modules['dateutil'])
                    except:
                        pass
        except Exception as e:
            logging.debug(f"Failed to install dependencies: {e}")
    
    # Reorder path: user packages first, but keep essential system packages
    # We need to keep some system packages (like basic Python libs) but exclude conflicting ones
    dist_packages = [p for p in sys.path if 'dist-packages' in p]
    essential_system = [p for p in sys.path if p not in dist_packages and p not in user_site_packages]
    
    if user_site_packages:
        # Prioritize user packages, then essential system, exclude problematic dist-packages
        sys.path = user_site_packages + essential_system
    else:
        # Fallback: just exclude dist-packages
        sys.path = [p for p in sys.path if p not in dist_packages]
    
    # Aggressively clean matplotlib/mpl_toolkits modules
    modules_to_remove = [k for k in sys.modules.keys() 
                        if (k.startswith('matplotlib') or k.startswith('mpl_toolkits')) 
                        and k not in original_modules]
    for mod in modules_to_remove:
        try:
            del sys.modules[mod]
        except KeyError:
            pass
    
    # Use import hook to block system mpl_toolkits and force user version
    class MplToolkitsImportHook:
        def __init__(self, user_site):
            self.user_site = user_site
            # Find mpl_toolkits in user matplotlib installation
            try:
                import matplotlib
                mpl_dir = os.path.dirname(matplotlib.__file__)
                # mpl_toolkits is usually a sibling of matplotlib
                self.mpl_toolkits_path = os.path.join(os.path.dirname(mpl_dir), 'mpl_toolkits')
            except:
                self.mpl_toolkits_path = os.path.join(user_site, 'mpl_toolkits')
        
        def find_spec(self, name, path, target=None):
            if name.startswith('mpl_toolkits'):
                # Check if path includes dist-packages (system version)
                if path and any('dist-packages' in str(p) for p in path):
                    # Try to find user version
                    if os.path.exists(self.mpl_toolkits_path):
                        import importlib.util
                        # For mpl_toolkits.mplot3d, find the specific module
                        if name == 'mpl_toolkits.mplot3d':
                            mod_path = os.path.join(self.mpl_toolkits_path, 'mplot3d', '__init__.py')
                        else:
                            mod_path = os.path.join(self.mpl_toolkits_path, '__init__.py')
                        if os.path.exists(mod_path):
                            return importlib.util.spec_from_file_location(name, mod_path)
            return None
    
    # Install the hook
    hook_instance = None
    if user_site_packages:
        hook_instance = MplToolkitsImportHook(user_site_packages[0])
        sys.meta_path.insert(0, hook_instance)
    
    # Now try importing insightface
    try:
        from insightface import app as insightface_app
        from insightface.model_zoo import model_zoo
        INSIGHTFACE_AVAILABLE = True
        logging.info("InsightFace imported successfully")
    finally:
        # Remove the hook
        if hook_instance and hook_instance in sys.meta_path:
            sys.meta_path.remove(hook_instance)
    
    # Restore original path (but keep user packages accessible)
    sys.path = original_path + user_site_packages
        
except (ImportError, Exception) as e:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("insightface not available: %s. Install with: pip install insightface", e)
    # Restore path on error
    try:
        sys.path = original_path
    except:
        pass

LOGGER = logging.getLogger("uvicorn.error").getChild(__name__)


class FaceRecognitionError(RuntimeError):
    """Raised when face recognition operations fail."""


class FaceRecognitionService:
    """Face recognition service using InsightFace ArcFace model for high accuracy."""
    
    def __init__(
        self,
        known_faces_dir: str | Path = "known-faces",
        model_name: str = "arcface_r100_v1",
        threshold: float = 0.6,
    ):
        """
        Initialize face recognition service.
        
        Args:
            known_faces_dir: Directory containing known face images
            model_name: InsightFace model name (default: arcface_r100_v1 for best accuracy)
            threshold: Similarity threshold (lower = more strict, default 0.6)
        """
        if not INSIGHTFACE_AVAILABLE:
            raise FaceRecognitionError(
                "InsightFace not available. Install with: pip install insightface onnxruntime"
            )
        
        self.known_faces_dir = Path(known_faces_dir)
        self.threshold = threshold
        self.known_faces: Dict[str, np.ndarray] = {}  # name -> embedding
        self.face_detector = None
        self.face_recognizer = None
        
        # Initialize InsightFace models
        self._initialize_models(model_name)
        
        # Load known faces
        self._load_known_faces()
    
    def _initialize_models(self, model_name: str) -> None:
        """Initialize InsightFace face detection and recognition models."""
        # Try multiple model names if the specified one fails
        model_names_to_try = [model_name, 'buffalo_l', 'buffalo_s', 'antelopev2']
        
        last_error = None
        for model in model_names_to_try:
            try:
                # Initialize face analysis app (includes detection and recognition)
                self.face_analyzer = insightface_app.FaceAnalysis(
                    name=model,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # Try CUDA first, fallback to CPU
                )
                self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                LOGGER.info(f"Initialized InsightFace model: {model}")
                return  # Success!
            except Exception as exc:
                last_error = exc
                LOGGER.debug(f"Failed to initialize model {model}: {exc}")
                continue
        
        # If all models failed, raise error
        raise FaceRecognitionError(f"Failed to initialize InsightFace models. Tried: {model_names_to_try}. Last error: {last_error}") from last_error
    
    def _load_known_faces(self) -> None:
        """Load all known faces from the known_faces directory."""
        if not self.known_faces_dir.exists():
            LOGGER.warning(f"Known faces directory does not exist: {self.known_faces_dir}")
            return
        
        self.known_faces.clear()
        loaded_count = 0
        
        # Support both flat structure (image files directly) and folder structure
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # First, try folder structure: known_faces/person_name/image1.jpg
        for person_dir in self.known_faces_dir.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                embeddings = []
                
                for image_file in person_dir.iterdir():
                    if image_file.suffix.lower() in image_extensions:
                        embedding = self._extract_face_embedding_from_file(image_file)
                        if embedding is not None:
                            embeddings.append(embedding)
                
                if embeddings:
                    # Average multiple embeddings for better accuracy
                    avg_embedding = np.mean(embeddings, axis=0)
                    self.known_faces[person_name] = avg_embedding
                    loaded_count += 1
                    LOGGER.info(f"Loaded {len(embeddings)} face(s) for '{person_name}'")
        
        # Also check for flat structure: known_faces/person_name.jpg
        for image_file in self.known_faces_dir.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                # Extract name from filename (remove extension)
                person_name = image_file.stem
                
                # Skip if already loaded from folder structure
                if person_name in self.known_faces:
                    continue
                
                embedding = self._extract_face_embedding_from_file(image_file)
                if embedding is not None:
                    self.known_faces[person_name] = embedding
                    loaded_count += 1
                    LOGGER.info(f"Loaded face for '{person_name}' from {image_file.name}")
        
        LOGGER.info(f"Loaded {loaded_count} known face(s) from {self.known_faces_dir}")
    
    def _extract_face_embedding_from_file(self, image_path: Path) -> Optional[np.ndarray]:
        """Extract face embedding from an image file."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                LOGGER.warning(f"Could not read image: {image_path}")
                return None
            
            # Convert BGR to RGB (InsightFace expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect and extract face
            faces = self.face_analyzer.get(image_rgb)
            
            if not faces:
                LOGGER.warning(f"No face detected in: {image_path}")
                return None
            
            # Use the largest face if multiple detected
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            # Return the embedding (norm vector)
            embedding = largest_face.normed_embedding
            return embedding
            
        except Exception as exc:
            LOGGER.error(f"Error extracting face from {image_path}: {exc}", exc_info=True)
            return None
    
    def recognize_faces(
        self,
        image: np.ndarray,
        return_locations: bool = True,
    ) -> List[Dict]:
        """
        Recognize faces in an image.
        
        Args:
            image: Input image (BGR format, as from OpenCV)
            return_locations: If True, return face bounding boxes
        
        Returns:
            List of recognition results, each containing:
            - name: Recognized person name or "Unknown"
            - confidence: Similarity score (0-1, higher is better)
            - bbox: Bounding box [x1, y1, x2, y2] if return_locations=True
        """
        if not self.known_faces:
            LOGGER.warning("No known faces loaded. Recognition will return 'Unknown' for all faces.")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_analyzer.get(image_rgb)
        
        if not faces:
            return []
        
        results = []
        
        for face in faces:
            # Get face embedding
            embedding = face.normed_embedding
            
            # Find best match
            best_match = None
            best_similarity = 0.0
            
            for name, known_embedding in self.known_faces.items():
                # Compute cosine similarity (dot product for normalized vectors)
                similarity = np.dot(embedding, known_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # Determine if match is above threshold
            if best_match and best_similarity >= self.threshold:
                name = best_match
                confidence = float(best_similarity)
            else:
                name = "Unknown"
                confidence = float(best_similarity) if best_similarity > 0 else 0.0
            
            result = {
                "name": name,
                "confidence": confidence,
            }
            
            if return_locations:
                bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
                result["bbox"] = bbox
            
            results.append(result)
        
        return results
    
    def add_known_face(
        self,
        name: str,
        image: np.ndarray,
    ) -> bool:
        """
        Add a new known face from an image.
        
        Args:
            name: Person's name
            image: Face image (BGR format)
        
        Returns:
            True if face was successfully added
        """
        embedding = self._extract_face_embedding_from_image(image)
        
        if embedding is None:
            return False
        
        # If person already exists, average with existing embedding
        if name in self.known_faces:
            existing = self.known_faces[name]
            self.known_faces[name] = (existing + embedding) / 2.0
            LOGGER.info(f"Updated embedding for '{name}' (averaged with existing)")
        else:
            self.known_faces[name] = embedding
            LOGGER.info(f"Added new face for '{name}'")
        
        return True
    
    def _extract_face_embedding_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from an image array."""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            faces = self.face_analyzer.get(image_rgb)
            
            if not faces:
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return largest_face.normed_embedding
            
        except Exception as exc:
            LOGGER.error(f"Error extracting face embedding: {exc}", exc_info=True)
            return None
    
    def get_known_faces(self) -> List[str]:
        """Get list of all known face names."""
        return list(self.known_faces.keys())
    
    def reload_known_faces(self) -> None:
        """Reload known faces from disk."""
        self._load_known_faces()
    
    def draw_recognitions(
        self,
        image: np.ndarray,
        recognitions: List[Dict],
    ) -> np.ndarray:
        """
        Draw recognition results on image.
        
        Args:
            image: Input image (BGR format)
            recognitions: Recognition results from recognize_faces()
        
        Returns:
            Image with drawn annotations
        """
        result_image = image.copy()
        
        for rec in recognitions:
            if "bbox" not in rec:
                continue
            
            x1, y1, x2, y2 = rec["bbox"]
            name = rec["name"]
            confidence = rec["confidence"]
            
            # Choose color: green for recognized, red for unknown
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{name} ({confidence:.2f})"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                result_image,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1,
            )
            
            # Draw label text
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        
        return result_image


__all__ = [
    "FaceRecognitionService",
    "FaceRecognitionError",
]

