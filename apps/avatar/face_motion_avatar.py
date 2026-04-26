from __future__ import annotations

import base64
import io
import importlib
import json
import math
import random
import struct
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np
import pyglet
from pyglet import shapes
from pyglet.display import get_display
from PIL import Image
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from pyglet.gl import (
    GL_BLEND,
    GL_DEPTH_TEST,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_REPEAT,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    glDisable,
    glActiveTexture,
    glBlendFunc,
    glBindTexture,
    glClearColor,
    glEnable,
    glTexParameteri,
    glViewport,
)
from pyglet.graphics.shader import Shader, ShaderProgram

try:
    _face_mesh_connections = importlib.import_module("mediapipe.python.solutions.face_mesh_connections")
    FACEMESH_TESSELATION = tuple(getattr(_face_mesh_connections, "FACEMESH_TESSELATION", ()))
    FACEMESH_LIPS = tuple(getattr(_face_mesh_connections, "FACEMESH_LIPS", ()))
except Exception:
    FACEMESH_TESSELATION = ()
    FACEMESH_LIPS = ()


WINDOW_TITLE = "DANYA Avatar"
DISPLAY_ROTATION_DEG = 90.0
WARP_GRID_COLS = 5
WARP_GRID_ROWS = 5
MAX_WARP_POINTS = 64
WEIGHT_DEADZONE = 0.035
WEIGHT_CURVE_POWER = 1.5
ONE_EURO_MIN_CUTOFF = 1.2
ONE_EURO_BETA = 0.015
ONE_EURO_D_CUTOFF = 1.0
PIP_WIDTH = 240
PIP_HEIGHT = 180
PIP_PADDING = 16
PIP_UPDATE_HZ = 15.0
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
BASE_DIR = PROJECT_ROOT
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "avatar.glb"
MODEL_CACHE = PROJECT_ROOT / ".cache" / "mediapipe_face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

FACE_MESHES = {
    "Head_Mesh",
    "Eye_Mesh",
    "Teeth_Mesh",
    "Tongue_Mesh",
}

ALIAS_WEIGHTS = {
    "jawopen": ["mouthopen"],
    "mouthopen": ["jawopen"],
    "mouthsmile": ["mouthsmileleft", "mouthsmileright"],
    "mouthlowerdownleft": ["mouthfrownleft"],
    "mouthlowerdownright": ["mouthfrownright"],
    "eyewideleft": ["eyesclosed"],
    "eyewideright": ["eyesclosed"],
    "eyelookinleft": ["eyelookinright"],
    "eyelookinright": ["eyelookinleft"],
    "eyelookoutleft": ["eyelookoutright"],
    "eyelookoutright": ["eyelookoutleft"],
    "eyelookupleft": ["eyeslookup"],
    "eyelookupright": ["eyeslookup"],
    "eyelookdownleft": ["eyeslookdown"],
    "eyelookdownright": ["eyeslookdown"],
    "eyesclosed": ["eyeblinkleft", "eyeblinkright"],
    "eyeblinkleft": ["eyesclosed"],
    "eyeblinkright": ["eyesclosed"],
    "browinnerup": ["browraise"],
    "browraise": ["browinnerup"],
}

FAST_EYE_LEFT_KEYS = {
    "eyeblinkleft",
    "eyesquintleft",
    "eyewideleft",
    "eyelookinleft",
    "eyelookoutleft",
    "eyelookupleft",
    "eyelookdownleft",
}

FAST_EYE_RIGHT_KEYS = {
    "eyeblinkright",
    "eyesquintright",
    "eyewideright",
    "eyelookinright",
    "eyelookoutright",
    "eyelookupright",
    "eyelookdownright",
}

FAST_RESPONSE_KEYS = FAST_EYE_LEFT_KEYS | FAST_EYE_RIGHT_KEYS | {"eyesclosed"}

FAST_MOUTH_KEYS = {
    "jawopen",
    "mouthopen",
    "mouthclose",
    "mouthfunnel",
    "mouthpucker",
    "mouthsmileleft",
    "mouthsmileright",
    "mouthstretchleft",
    "mouthstretchright",
    "mouthpressleft",
    "mouthpressright",
    "mouthlowerdownleft",
    "mouthlowerdownright",
    "viseme_aa",
    "viseme_e",
    "viseme_i",
    "viseme_o",
    "viseme_u",
    "viseme_ff",
    "viseme_rr",
    "viseme_nn",
    "viseme_dd",
    "viseme_th",
    "viseme_sil",
}

FAST_VISEME_KEYS = {
    "viseme_aa",
    "viseme_e",
    "viseme_i",
    "viseme_o",
    "viseme_u",
    "viseme_ff",
    "viseme_rr",
    "viseme_nn",
    "viseme_dd",
    "viseme_th",
    "viseme_sil",
}

EYE_DIRECT_KEYS = {
    "eyeblinkleft",
    "eyeblinkright",
    "eyesclosed",
    "eyesquintleft",
    "eyesquintright",
    "eyewideleft",
    "eyewideright",
}

EYE_GAZE_KEYS = {
    "eyelookinleft",
    "eyelookinright",
    "eyelookoutleft",
    "eyelookoutright",
    "eyelookupleft",
    "eyelookupright",
    "eyelookdownleft",
    "eyelookdownright",
}

MOUTH_DIRECT_KEYS = {
    "jawopen",
    "mouthopen",
    "mouthclose",
    "mouthfunnel",
    "mouthpucker",
    "mouthsmile",
    "mouthsmileleft",
    "mouthsmileright",
    "mouthstretchleft",
    "mouthstretchright",
    "mouthpressleft",
    "mouthpressright",
    "mouthlowerdownleft",
    "mouthlowerdownright",
    "mouthdimpleleft",
    "mouthdimpleright",
    "mouthrolllower",
    "mouthrollupper",
    "mouthshruglower",
    "mouthshrugupper",
    "mouthright",
    "mouthleft",
    "mouthroundleft",
    "mouthroundright",
    "mouthupperupleft",
    "mouthupperupright",
    "tongueout",
    "tongueinout",
    "tongueleft",
    "tongueright",
    "tongueup",
    "tonguedown",
    "tongueroll",
    "viseme_aa",
    "viseme_ch",
    "viseme_dd",
    "viseme_e",
    "viseme_ff",
    "viseme_i",
    "viseme_kk",
    "viseme_nn",
    "viseme_o",
    "viseme_pp",
    "viseme_rr",
    "viseme_sil",
    "viseme_ss",
    "viseme_th",
    "viseme_u",
}


@dataclass
class MeshPart:
    name: str
    vertex_list: Any
    texture: Optional[Any]
    alpha_mode: str
    base_positions: np.ndarray
    base_normals: np.ndarray
    base_texcoords: np.ndarray
    indices: np.ndarray
    morph_names: list[str]
    morph_positions: np.ndarray
    morph_normals: np.ndarray
    mesh_transform: np.ndarray
    normal_transform: np.ndarray


@dataclass
class TrackingResult:
    weights: dict[str, float]
    pose: tuple[float, float, float]
    pose_matrix: np.ndarray
    face_landmarks: list[tuple[float, float]]


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._prev_value: Optional[float] = None
        self._prev_timestamp: Optional[float] = None
        self._prev_derivative: float = 0.0
        self._filtered_value: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, te: float) -> float:
        cutoff = max(float(cutoff), 1e-6)
        te = max(float(te), 1e-6)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    @staticmethod
    def _low_pass(current: float, previous: float, alpha: float) -> float:
        return alpha * current + (1.0 - alpha) * previous

    def reset(self, value: Optional[float] = None, timestamp: Optional[float] = None) -> None:
        self._prev_value = value
        self._prev_timestamp = timestamp
        self._prev_derivative = 0.0
        self._filtered_value = value

    def filter(self, value: float, timestamp: float) -> float:
        value = float(value)
        timestamp = float(timestamp)

        if self._prev_timestamp is None or self._filtered_value is None or self._prev_value is None:
            self._prev_timestamp = timestamp
            self._prev_value = value
            self._filtered_value = value
            self._prev_derivative = 0.0
            return value

        te = max(timestamp - self._prev_timestamp, 1e-4)
        derivative = (value - self._prev_value) / te
        derivative_alpha = self._alpha(self.d_cutoff, te)
        self._prev_derivative = self._low_pass(derivative, self._prev_derivative, derivative_alpha)
        cutoff = self.min_cutoff + self.beta * abs(self._prev_derivative)
        filtered_alpha = self._alpha(cutoff, te)
        self._filtered_value = self._low_pass(value, self._filtered_value, filtered_alpha)
        self._prev_value = value
        self._prev_timestamp = timestamp
        return float(self._filtered_value)


def _build_rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _build_rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _build_rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class CameraStream:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480) -> None:
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            with self.lock:
                self.frame = frame

    def latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def close(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.capture.isOpened():
            self.capture.release()


class FaceExpressionTracker:
    def __init__(self, model_path: Path) -> None:
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = face_landmarker.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = face_landmarker.FaceLandmarker.create_from_options(options)
        self.start_time = time.monotonic()

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def infer(self, frame_bgr: np.ndarray) -> TrackingResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.monotonic() - self.start_time) * 1000.0)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        weights: dict[str, float] = {}

        if result.face_blendshapes:
            for item in result.face_blendshapes[0]:
                if item.category_name:
                    weights[item.category_name.lower()] = float(item.score or 0.0)

        landmarks: list[tuple[float, float]] = []
        if getattr(result, "face_landmarks", None):
            first_face = result.face_landmarks[0]
            for lm in first_face:
                landmarks.append((float(lm.x), float(lm.y)))

        self._apply_aliases(weights)
        weights = self._map_to_avatar_weights(weights)
        pose_matrix = self._extract_pose_matrix(result)
        pose = self._matrix_to_pose(pose_matrix)
        return TrackingResult(weights=weights, pose=pose, pose_matrix=pose_matrix, face_landmarks=landmarks)

    @staticmethod
    def _apply_aliases(weights: dict[str, float]) -> None:
        for source, targets in ALIAS_WEIGHTS.items():
            if source in EYE_GAZE_KEYS:
                continue
            if source in weights:
                for target in targets:
                    if target in EYE_GAZE_KEYS:
                        continue
                    weights[target] = max(weights.get(target, 0.0), weights[source])

    @staticmethod
    def _map_to_avatar_weights(weights: dict[str, float]) -> dict[str, float]:
        mapped = dict(weights)

        # Pass eye tracking through directly so left/right motion and blinks
        # stay as close as possible to MediaPipe output.
        for key in EYE_DIRECT_KEYS:
            if key in weights:
                mapped[key] = weights[key]

        if "eyesclosed" not in mapped:
            mapped["eyesclosed"] = max(weights.get("eyeblinkleft", 0.0), weights.get("eyeblinkright", 0.0))

        gaze_gain = 1.3
        gaze_power = 0.72
        for key in EYE_GAZE_KEYS:
            raw = float(weights.get(key, 0.0))
            if raw > 0.0:
                mapped[key] = float(np.clip((raw ** gaze_power) * gaze_gain, 0.0, 1.0))

        # Mouth expressions are passed through as directly as possible.
        for key in MOUTH_DIRECT_KEYS:
            if key in weights:
                mapped[key] = weights[key]

        # Minimal fallback for the common open-mouth coupling.
        if "jawopen" not in mapped and "mouthopen" in mapped:
            mapped["jawopen"] = mapped["mouthopen"]
        if "mouthopen" not in mapped and "jawopen" in mapped:
            mapped["mouthopen"] = mapped["jawopen"]

        mapped["browinnerup"] = max(weights.get("browinnerup", 0.0), weights.get("browraise", 0.0))
        mapped["browdownleft"] = weights.get("browdownleft", 0.0)
        mapped["browdownright"] = weights.get("browdownright", 0.0)
        mapped["browouterupleft"] = weights.get("browouterupleft", 0.0)
        mapped["browouterupright"] = weights.get("browouterupright", 0.0)

        return mapped

    @staticmethod
    def _extract_pose_matrix(result: Any) -> np.ndarray:
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if not matrices:
            matrices = getattr(result, "face_transformation_matrixes", None)
        if not matrices:
            return np.eye(4, dtype=np.float32)

        matrix = np.array(matrices[0], dtype=np.float32)
        if matrix.size == 16:
            matrix = matrix.reshape(4, 4)
        if matrix.shape != (4, 4):
            return np.eye(4, dtype=np.float32)
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: np.ndarray) -> tuple[float, float, float]:
        rotation = np.array(matrix[:3, :3], dtype=np.float32)
        sy = math.sqrt(float(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]))
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            yaw = math.atan2(float(-rotation[2, 0]), sy)
            roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        else:
            pitch = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
            yaw = math.atan2(float(-rotation[2, 0]), sy)
            roll = 0.0

        return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


class GLBAvatar:
    def __init__(self, glb_path: Path) -> None:
        if not glb_path.exists():
            raise FileNotFoundError(f"Missing GLB model: {glb_path}")
        self.data = glb_path.read_bytes()
        self.json_data, self.bin_chunk = self._parse_glb(self.data)
        self.nodes = self.json_data.get("nodes", [])
        self.meshes = self.json_data.get("meshes", [])
        self.materials = self.json_data.get("materials", [])
        self.textures = self.json_data.get("textures", [])
        self.images = self.json_data.get("images", [])
        self.scene_index = int(self.json_data.get("scene", 0))
        self.display_rotation = self._rotation_z_matrix(math.radians(DISPLAY_ROTATION_DEG))
        self.node_world_matrices = self._compute_node_world_matrices()
        self.parts = self._build_mesh_parts()
        self._center_and_scale_model()

    @staticmethod
    def _parse_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
        magic, version, _length = struct.unpack_from("<III", data, 0)
        if magic != 0x46546C67 or version != 2:
            raise ValueError("Unsupported GLB file")
        offset = 12
        json_chunk = None
        bin_chunk = b""
        while offset < len(data):
            chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
            offset += 8
            chunk = data[offset : offset + chunk_length]
            offset += chunk_length
            if chunk_type == 0x4E4F534A:
                json_chunk = chunk
            elif chunk_type == 0x004E4942:
                bin_chunk = chunk
        if json_chunk is None:
            raise ValueError("GLB missing JSON chunk")
        return json.loads(json_chunk.decode("utf-8")), bin_chunk

    @staticmethod
    def _component_dtype(component_type: int) -> np.dtype[Any]:
        mapping = {
            5120: np.int8,
            5121: np.uint8,
            5122: np.int16,
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }
        return np.dtype(mapping[component_type]).newbyteorder("<")

    @staticmethod
    def _num_components(type_name: str) -> int:
        return {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT4": 16,
        }[type_name]

    def _read_accessor(self, accessor_index: int) -> np.ndarray:
        accessor = self.json_data["accessors"][accessor_index]
        buffer_view = self.json_data["bufferViews"][accessor["bufferView"]]
        dtype = self._component_dtype(accessor["componentType"])
        count = int(accessor["count"])
        components = self._num_components(accessor["type"])
        base_offset = int(buffer_view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        byte_stride = int(buffer_view.get("byteStride", components * dtype.itemsize))
        normalized = bool(accessor.get("normalized", False))

        if byte_stride == components * dtype.itemsize:
            arr = np.frombuffer(
                self.bin_chunk,
                dtype=dtype,
                count=count * components,
                offset=base_offset,
            ).reshape(count, components)
        else:
            arr = np.ndarray(
                shape=(count, components),
                dtype=dtype,
                buffer=self.bin_chunk,
                offset=base_offset,
                strides=(byte_stride, dtype.itemsize),
            )
        arr = np.array(arr, copy=True)

        if normalized and arr.dtype.kind in {"i", "u"}:
            if arr.dtype.kind == "u":
                arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
            else:
                info = np.iinfo(arr.dtype)
                arr = np.maximum(arr.astype(np.float32) / max(abs(info.min), info.max), -1.0)
        return arr

    def _decode_image(self, image_index: int) -> Image.Image:
        image = self.images[image_index]
        if "bufferView" in image:
            view = self.json_data["bufferViews"][image["bufferView"]]
            start = int(view.get("byteOffset", 0))
            end = start + int(view["byteLength"])
            payload = self.bin_chunk[start:end]
        elif "uri" in image:
            uri = image["uri"]
            if uri.startswith("data:"):
                payload = base64.b64decode(uri.split(",", 1)[1])
            else:
                payload = Path(uri).read_bytes()
        else:
            raise ValueError("Unsupported image source")
        return Image.open(io.BytesIO(payload)).convert("RGBA")

    def _load_texture(self, image_index: Optional[int]) -> Optional[Any]:
        if image_index is None:
            return None
        pil_image = self._decode_image(image_index)
        raw = pil_image.tobytes()
        texture = pyglet.image.ImageData(
            pil_image.width,
            pil_image.height,
            "RGBA",
            raw,
            pitch=-pil_image.width * 4,
        ).get_texture()
        glBindTexture(GL_TEXTURE_2D, texture.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        return texture

    @staticmethod
    def _translation_matrix(t: np.ndarray) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = t[:3]
        return m

    @staticmethod
    def _scale_matrix(s: np.ndarray) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[0, 0], m[1, 1], m[2, 2] = s[:3]
        return m

    @staticmethod
    def _rotation_z_matrix(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _quat_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def _local_matrix(self, node: dict[str, Any]) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], dtype=np.float32).reshape(4, 4).T
        matrix = np.eye(4, dtype=np.float32)
        if "translation" in node:
            matrix = matrix @ self._translation_matrix(np.array(node["translation"], dtype=np.float32))
        if "rotation" in node:
            matrix = matrix @ self._quat_matrix(np.array(node["rotation"], dtype=np.float32))
        if "scale" in node:
            matrix = matrix @ self._scale_matrix(np.array(node["scale"], dtype=np.float32))
        return matrix

    def _compute_node_world_matrices(self) -> list[np.ndarray]:
        world = [np.eye(4, dtype=np.float32) for _ in self.nodes]

        def visit(index: int, parent: np.ndarray) -> None:
            local = self._local_matrix(self.nodes[index])
            world[index] = parent @ local
            for child in self.nodes[index].get("children", []):
                visit(child, world[index])

        scene = self.json_data.get("scenes", [])[self.scene_index]
        for root in scene.get("nodes", []):
            visit(root, np.eye(4, dtype=np.float32))
        return world

    def _material_texture(self, material_index: Optional[int]) -> tuple[Optional[Any], str]:
        if material_index is None:
            return None, "OPAQUE"
        material = self.materials[material_index]
        alpha_mode = material.get("alphaMode", "OPAQUE")
        pbr = material.get("pbrMetallicRoughness", {})
        tex_index = None
        if "baseColorTexture" in pbr:
            tex_index = pbr["baseColorTexture"].get("index")
        texture = None
        if tex_index is not None:
            image_index = self.textures[tex_index].get("source")
            texture = self._load_texture(image_index)
        return texture, alpha_mode

    def _build_mesh_parts(self) -> list[MeshPart]:
        vertex_shader = """
            #version 330 core
            in vec3 position;
            in vec3 normal;
            in vec2 texcoord;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            uniform mat3 global_warp;
            uniform vec2 warp_src_min;
            uniform vec2 warp_src_max;
            uniform int warp_cols;
            uniform int warp_rows;
            uniform float warp_points[128];
            out vec3 v_normal;
            out vec2 v_texcoord;

            vec2 get_warp_point(int index) {
                int base = index * 2;
                return vec2(warp_points[base], warp_points[base + 1]);
            }

            vec2 warp_ndc_mesh(vec2 ndc) {
                vec2 span = max(warp_src_max - warp_src_min, vec2(1e-5));
                vec2 uv = (ndc - warp_src_min) / span;
                if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                    return ndc;
                }

                float gx = uv.x * float(warp_cols - 1);
                float gy = uv.y * float(warp_rows - 1);
                int x0 = int(floor(gx));
                int y0 = int(floor(gy));
                x0 = clamp(x0, 0, warp_cols - 2);
                y0 = clamp(y0, 0, warp_rows - 2);
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                float tx = gx - float(x0);
                float ty = gy - float(y0);

                int i00 = y0 * warp_cols + x0;
                int i10 = y0 * warp_cols + x1;
                int i01 = y1 * warp_cols + x0;
                int i11 = y1 * warp_cols + x1;

                vec2 p00 = get_warp_point(i00);
                vec2 p10 = get_warp_point(i10);
                vec2 p01 = get_warp_point(i01);
                vec2 p11 = get_warp_point(i11);
                vec2 top = mix(p00, p10, tx);
                vec2 bottom = mix(p01, p11, tx);
                return mix(top, bottom, ty);
            }

            void main() {
                vec4 world_pos = model * vec4(position, 1.0);
                v_normal = mat3(transpose(inverse(model))) * normal;
                v_texcoord = vec2(texcoord.x, 1.0 - texcoord.y);
                vec4 clip = projection * view * world_pos;
                vec3 ndc = clip.xyz / clip.w;
                vec2 local_warped_ndc = warp_ndc_mesh(ndc.xy);
                vec3 global_warped_ndc = global_warp * vec3(local_warped_ndc, 1.0);
                gl_Position = vec4(global_warped_ndc.xy * clip.w, clip.z, clip.w);
            }
        """
        fragment_shader = """
            #version 330 core
            in vec3 v_normal;
            in vec2 v_texcoord;
            uniform sampler2D tex0;
            uniform vec3 light_dir;
            uniform int use_texture;
            uniform vec4 color_tint;
            out vec4 fragColor;
            void main() {
                vec3 n = normalize(v_normal);
                float diff = max(dot(n, normalize(-light_dir)), 0.2);
                vec4 base = color_tint;
                if (use_texture == 1) {
                    base *= texture(tex0, v_texcoord);
                }
                float brightness = 0.82 + 0.38 * diff;
                fragColor = vec4(base.rgb * brightness, base.a);
            }
        """
        self.program = ShaderProgram(Shader(vertex_shader, "vertex"), Shader(fragment_shader, "fragment"))
        self.program.use()
        self.program["tex0"] = 0
        self.program["light_dir"] = (0.3, 0.7, 0.8)
        self.program["use_texture"] = 0
        self.program["color_tint"] = (1.0, 1.0, 1.0, 1.0)
        self.program["global_warp"] = tuple(np.eye(3, dtype=np.float32).T.reshape(-1))
        self.program["warp_src_min"] = (-0.5, -0.5)
        self.program["warp_src_max"] = (0.5, 0.5)
        self.program["warp_cols"] = 2
        self.program["warp_rows"] = 2
        identity_points = np.zeros((MAX_WARP_POINTS, 2), dtype=np.float32)
        self.program["warp_points"] = tuple(identity_points.reshape(-1))

        parts: list[MeshPart] = []
        for node_index, node in enumerate(self.nodes):
            mesh_index = node.get("mesh")
            if mesh_index is None:
                continue
            mesh = self.meshes[int(mesh_index)]
            mesh_name = node.get("name") or mesh.get("name") or f"mesh_{mesh_index}"
            if mesh_name not in FACE_MESHES:
                continue
            primitive = mesh["primitives"][0]
            attributes = primitive["attributes"]
            base_positions = self._read_accessor(attributes["POSITION"]).astype(np.float32)
            base_normals = self._read_accessor(attributes["NORMAL"]).astype(np.float32)
            if "TEXCOORD_0" in attributes:
                base_texcoords = self._read_accessor(attributes["TEXCOORD_0"]).astype(np.float32)
            else:
                base_texcoords = np.zeros((base_positions.shape[0], 2), dtype=np.float32)
            indices = self._read_accessor(int(primitive["indices"]))[:, 0].astype(np.uint32)

            morph_names = [name.lower() for name in mesh.get("extras", {}).get("targetNames", [])]
            morph_positions = []
            morph_normals = []
            for target in primitive.get("targets", []):
                if "POSITION" in target:
                    morph_positions.append(self._read_accessor(target["POSITION"]).astype(np.float32))
                else:
                    morph_positions.append(np.zeros_like(base_positions))
                if "NORMAL" in target:
                    morph_normals.append(self._read_accessor(target["NORMAL"]).astype(np.float32))
                else:
                    morph_normals.append(np.zeros_like(base_normals))
            if morph_positions:
                morph_positions_arr = np.stack(morph_positions, axis=0)
                morph_normals_arr = np.stack(morph_normals, axis=0)
            else:
                morph_positions_arr = np.zeros((0, *base_positions.shape), dtype=np.float32)
                morph_normals_arr = np.zeros((0, *base_normals.shape), dtype=np.float32)

            texture, alpha_mode = self._material_texture(primitive.get("material"))
            mesh_transform = self.node_world_matrices[node_index].astype(np.float32)
            normal_transform = np.linalg.inv(mesh_transform).T.astype(np.float32)

            vlist = self.program.vertex_list_indexed(
                int(base_positions.shape[0]),
                pyglet.gl.GL_TRIANGLES,
                indices.tolist(),
                position=("f", base_positions.reshape(-1)),
                normal=("f", base_normals.reshape(-1)),
                texcoord=("f", base_texcoords.reshape(-1)),
            )

            parts.append(
                MeshPart(
                    name=mesh_name,
                    vertex_list=vlist,
                    texture=texture,
                    alpha_mode=alpha_mode,
                    base_positions=base_positions,
                    base_normals=base_normals,
                    base_texcoords=base_texcoords,
                    indices=indices,
                    morph_names=morph_names,
                    morph_positions=morph_positions_arr,
                    morph_normals=morph_normals_arr,
                    mesh_transform=mesh_transform,
                    normal_transform=normal_transform,
                )
            )
        return parts

    def _center_and_scale_model(self) -> None:
        points = []
        for part in self.parts:
            pts = np.c_[part.base_positions, np.ones((part.base_positions.shape[0], 1), dtype=np.float32)]
            pts = (part.mesh_transform @ pts.T).T[:, :3]
            points.append(pts)
        if not points:
            self.model_matrix = np.eye(4, dtype=np.float32)
            return
        cloud = np.concatenate(points, axis=0)
        mins = cloud.min(axis=0)
        maxs = cloud.max(axis=0)
        center = (mins + maxs) * 0.5
        extent = float(np.max(maxs - mins))
        scale = 2.25 / extent if extent > 1e-6 else 1.0
        translate = np.eye(4, dtype=np.float32)
        translate[:3, 3] = -center
        scale_m = np.eye(4, dtype=np.float32)
        scale_m[:3, :3] *= scale
        self.model_matrix = self.display_rotation @ scale_m @ translate

    def apply_expression(self, weights: dict[str, float]) -> None:
        for part in self.parts:
            vertices = part.base_positions.copy()
            normals = part.base_normals.copy()
            if part.morph_names and part.morph_positions.size:
                for index, morph_name in enumerate(part.morph_names):
                    weight = weights.get(morph_name, 0.0)
                    if weight > 1e-4:
                        vertices += part.morph_positions[index] * weight
                        normals += part.morph_normals[index] * weight

            vertex_h = np.c_[vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)]
            vertex_h = (part.mesh_transform @ vertex_h.T).T[:, :3]
            normal_matrix = part.normal_transform[:3, :3]
            normal_h = (normal_matrix @ normals.T).T
            normal_h = normal_h / np.linalg.norm(normal_h, axis=1, keepdims=True).clip(min=1e-6)
            self._update_vertex_list(part, vertex_h, normal_h)

    @staticmethod
    def _update_vertex_list(part: MeshPart, positions: np.ndarray, normals: np.ndarray) -> None:
        part.vertex_list.set_attribute_data("position", positions.astype(np.float32).reshape(-1))
        part.vertex_list.set_attribute_data("normal", normals.astype(np.float32).reshape(-1))

    def draw(
        self,
        weights: dict[str, float],
        view: np.ndarray,
        projection: np.ndarray,
        pose_matrix: Optional[np.ndarray],
        global_warp: np.ndarray,
        warp_src_min: np.ndarray,
        warp_src_max: np.ndarray,
        warp_cols: int,
        warp_rows: int,
        warp_points: np.ndarray,
    ) -> None:
        self.program.use()
        model_matrix = self.model_matrix if pose_matrix is None else np.array(pose_matrix, dtype=np.float32) @ self.model_matrix
        self.program["model"] = tuple(model_matrix.T.reshape(-1))
        self.program["view"] = tuple(view.T.reshape(-1))
        self.program["projection"] = tuple(projection.T.reshape(-1))
        self.program["light_dir"] = (0.3, 0.7, 0.8)
        self.program["global_warp"] = tuple(np.array(global_warp, dtype=np.float32).T.reshape(-1))
        self.program["warp_src_min"] = tuple(np.array(warp_src_min, dtype=np.float32).reshape(-1))
        self.program["warp_src_max"] = tuple(np.array(warp_src_max, dtype=np.float32).reshape(-1))
        self.program["warp_cols"] = int(warp_cols)
        self.program["warp_rows"] = int(warp_rows)
        self.program["warp_points"] = tuple(np.array(warp_points, dtype=np.float32).reshape(-1))
        for part in self.parts:
            self._update_part_for_draw(part, weights)
            self.program["use_texture"] = 1 if part.texture is not None else 0
            self.program["color_tint"] = (1.0, 1.0, 1.0, 1.0)
            if part.texture is not None:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(part.texture.target, part.texture.id)
            part.vertex_list.draw(pyglet.gl.GL_TRIANGLES)

    def _update_part_for_draw(self, part: MeshPart, weights: dict[str, float]) -> None:
        vertices = part.base_positions.copy()
        normals = part.base_normals.copy()
        if part.morph_names and part.morph_positions.size:
            for index, morph_name in enumerate(part.morph_names):
                weight = weights.get(morph_name, 0.0)
                if weight > 1e-4:
                    vertices += part.morph_positions[index] * weight
                    normals += part.morph_normals[index] * weight
        vertex_h = np.c_[vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)]
        vertex_h = (part.mesh_transform @ vertex_h.T).T[:, :3]
        normal_matrix = part.normal_transform[:3, :3]
        normal_h = (normal_matrix @ normals.T).T
        normal_h = normal_h / np.linalg.norm(normal_h, axis=1, keepdims=True).clip(min=1e-6)
        self._update_vertex_list(part, vertex_h, normal_h)


class AvatarApp(pyglet.window.Window):
    def __init__(self) -> None:
        config = pyglet.gl.Config(double_buffer=True, depth_size=24)
        super().__init__(caption=WINDOW_TITLE, fullscreen=False, config=config, vsync=True, width=1280, height=720)
        self.set_mouse_visible(True)
        self._last_viewport_size = (float(self.width), float(self.height))
        self._windowed_size = (int(self.width), int(self.height))
        try:
            self._windowed_location = tuple(self.get_location())
        except Exception:
            self._windowed_location = (0, 0)
        self.camera = CameraStream()
        self.tracker = self._create_tracker()
        self.avatar = GLBAvatar(MODEL_PATH)
        self.raw_weights: dict[str, float] = {}
        self.smoothed_weights: dict[str, float] = {}
        self.neutral_weights: dict[str, float] = {}
        self.weight_filters: dict[str, OneEuroFilter] = {}
        self.current_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.current_pose_matrix = np.eye(4, dtype=np.float32)
        self.current_face_landmarks: list[tuple[float, float]] = []
        self.view_rotation_index = 0
        self.app_start_time = time.monotonic()
        self.last_update_time = self.app_start_time
        self.breath_phase = 0.0
        self.breath_offset = 0.0
        self.saccade_offset = np.zeros(2, dtype=np.float32)
        self.next_saccade_time = self.app_start_time + random.uniform(1.0, 3.0)
        self.recording_active = False
        self.recording_frames: list[dict[str, Any]] = []
        self.recording_started_at = 0.0
        self.recording_save_prompt_active = False
        self.recording_lock = threading.Lock()
        self.pip_size = (PIP_WIDTH, PIP_HEIGHT)
        self.pip_update_interval = 1.0 / PIP_UPDATE_HZ
        self.last_pip_update = 0.0
        self.pip_texture = pyglet.image.Texture.create(PIP_WIDTH, PIP_HEIGHT)
        self.pip_sprite = pyglet.sprite.Sprite(self.pip_texture)
        self.pip_background = shapes.Rectangle(0, 0, PIP_WIDTH + 8, PIP_HEIGHT + 8, color=(0, 0, 0))
        self.pip_background.opacity = 170
        black_frame = np.zeros((PIP_HEIGHT, PIP_WIDTH, 3), dtype=np.uint8)
        self.pip_texture.blit_into(
            pyglet.image.ImageData(PIP_WIDTH, PIP_HEIGHT, "BGR", black_frame.tobytes(), pitch=-PIP_WIDTH * 3),
            0,
            0,
            0,
        )
        self.global_warp_points = self._default_global_warp_points()
        self.warp_points = self._default_warp_grid_points()
        self.active_global_corner: Optional[int] = None
        self.active_point: Optional[int] = None
        self.show_handles = True
        self.show_axes = False  # Toggle with 'A' key for camera axis visualization
        self.corner_pick_radius = 28.0
        self.handle_batch = pyglet.graphics.Batch()
        total_points = WARP_GRID_COLS * WARP_GRID_ROWS
        self.handle_points = [
            shapes.Circle(0, 0, 6, color=(255, 255, 255), batch=self.handle_batch)
            for _ in range(total_points)
        ]
        total_lines = WARP_GRID_ROWS * (WARP_GRID_COLS - 1) + WARP_GRID_COLS * (WARP_GRID_ROWS - 1)
        self.handle_lines = [
            shapes.Line(0, 0, 0, 0, thickness=2, color=(100, 100, 100), batch=self.handle_batch)
            for _ in range(total_lines)
        ]
        self.global_handle_points = [
            shapes.Circle(0, 0, 9, color=(255, 210, 90), batch=self.handle_batch)
            for _ in range(4)
        ]
        self.global_handle_lines = [
            shapes.Line(0, 0, 0, 0, thickness=3, color=(200, 140, 60), batch=self.handle_batch)
            for _ in range(4)
        ]
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    def _create_tracker(self) -> FaceExpressionTracker:
        MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        if not MODEL_CACHE.exists():
            print("Downloading MediaPipe face landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_CACHE)
        return FaceExpressionTracker(MODEL_CACHE)

    def update(self, dt: float) -> None:  # noqa: ARG002
        now = time.monotonic()
        frame = self.camera.latest_frame()
        if frame is not None:
            try:
                tracking = self.tracker.infer(frame)
            except Exception:
                tracking = TrackingResult(
                    weights={},
                    pose=(0.0, 0.0, 0.0),
                    pose_matrix=np.eye(4, dtype=np.float32),
                    face_landmarks=[],
                )
            self.raw_weights = tracking.weights
            self.current_pose = tracking.pose
            self.current_pose_matrix = self._pose_to_matrix(tracking.pose)
            self.current_face_landmarks = tracking.face_landmarks
            if now - self.last_pip_update >= self.pip_update_interval:
                self._update_pip_texture(frame, tracking.face_landmarks)
                self.last_pip_update = now
        else:
            self.raw_weights = {}
            self.current_pose = (0.0, 0.0, 0.0)
            self.current_pose_matrix = np.eye(4, dtype=np.float32)
            self.current_face_landmarks = []

        self._update_procedural_state(dt, now)
        self.smoothed_weights = self._filter_weights(self.raw_weights, now)

        if self.recording_active:
            self._append_recording_frame(now, self.smoothed_weights, self.current_pose)

    def _draw_debug_axes(self, projection: np.ndarray, view: np.ndarray) -> None:
        """Draw large XYZ world axes as a 2D overlay for camera/orbit debugging."""

        def project(world_pos: np.ndarray) -> tuple[float, float] | None:
            clip = projection @ (view @ np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float32))
            w = float(clip[3])
            if abs(w) < 1e-6:
                return None
            ndc = clip[:3] / w
            # Allow slight overflow so labels near borders still show.
            if ndc[2] < -1.2 or ndc[2] > 1.2:
                return None
            sx = float((ndc[0] + 1.0) * 0.5 * self.width)
            sy = float((ndc[1] + 1.0) * 0.5 * self.height)
            return sx, sy

        # Move the gizmo in front of the face and make it long enough to be obvious.
        axis_origin = np.array([0.0, 0.28, 0.85], dtype=np.float32)
        axis_len = 1.4
        axes = [
            ("X", np.array([axis_len, 0.0, 0.0], dtype=np.float32), (255, 70, 70)),
            ("Y", np.array([0.0, axis_len, 0.0], dtype=np.float32), (90, 255, 90)),
            ("Z", np.array([0.0, 0.0, axis_len], dtype=np.float32), (100, 150, 255)),
        ]

        start = project(axis_origin)
        if start is None:
            return

        glDisable(GL_DEPTH_TEST)
        try:
            origin_dot = shapes.Circle(start[0], start[1], 6.0, color=(240, 240, 240))
            origin_dot.opacity = 230
            origin_dot.draw()

            for label, offset, color in axes:
                end = project(axis_origin + offset)
                if end is None:
                    continue
                axis_line = shapes.Line(start[0], start[1], end[0], end[1], thickness=4.0, color=color)
                axis_line.opacity = 230
                axis_line.draw()

                tip = shapes.Circle(end[0], end[1], 7.0, color=color)
                tip.opacity = 240
                tip.draw()

                text = pyglet.text.Label(
                    label,
                    x=end[0] + 10,
                    y=end[1] + 8,
                    color=(color[0], color[1], color[2], 255),
                    font_size=14,
                    anchor_x="left",
                    anchor_y="bottom",
                )
                text.draw()
        finally:
            glEnable(GL_DEPTH_TEST)

    def on_draw(self) -> None:
        self.clear()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self._constrain_warp_points()
        projection = self._make_projection_matrix()
        view = self._make_view_matrix()
        global_warp = self._make_global_warp_matrix()
        warp_src_min, warp_src_max, warp_cols, warp_rows, warp_points = self._make_screen_warp_uniforms()
        self.avatar.draw(
            self.smoothed_weights,
            view,
            projection,
            self.current_pose_matrix,
            global_warp,
            warp_src_min,
            warp_src_max,
            warp_cols,
            warp_rows,
            warp_points,
        )
        if self.show_handles:
            self._update_warp_handle_visuals()
            self.handle_batch.draw()
        if self.show_axes:
            self._draw_debug_axes(projection, view)
        self._draw_camera_pip()

    def on_resize(self, width: int, height: int) -> None:
        """Resize viewport and keep warp controls aligned with window size changes."""
        old_w, old_h = self._last_viewport_size
        new_w = max(float(width), 1.0)
        new_h = max(float(height), 1.0)

        if not hasattr(self, "warp_points") or not hasattr(self, "global_warp_points"):
            self._last_viewport_size = (new_w, new_h)
            glViewport(0, 0, int(new_w), int(new_h))
            return

        if old_w > 1.0 and old_h > 1.0 and (abs(new_w - old_w) > 1.0 or abs(new_h - old_h) > 1.0):
            sx = new_w / old_w
            sy = new_h / old_h
            for point in self.warp_points:
                point[0] *= sx
                point[1] *= sy
            for point in self.global_warp_points:
                point[0] *= sx
                point[1] *= sy
            self._constrain_warp_points()

        self._last_viewport_size = (new_w, new_h)
        if not self.fullscreen:
            self._windowed_size = (int(new_w), int(new_h))
        glViewport(0, 0, int(new_w), int(new_h))

    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen on the monitor containing the current window."""
        if self.fullscreen:
            self.set_fullscreen(False, width=self._windowed_size[0], height=self._windowed_size[1])
            try:
                self.set_location(int(self._windowed_location[0]), int(self._windowed_location[1]))
            except Exception:
                pass
            return

        self._windowed_size = (int(self.width), int(self.height))
        try:
            self._windowed_location = tuple(self.get_location())
        except Exception:
            self._windowed_location = (0, 0)

        target_screen = None
        try:
            display = get_display()
            screens = display.get_screens()
            window_x, window_y = self._windowed_location
            window_center_x = window_x + self._windowed_size[0] // 2
            window_center_y = window_y + self._windowed_size[1] // 2
            for screen in screens:
                if (
                    screen.x <= window_center_x < screen.x + screen.width
                    and screen.y <= window_center_y < screen.y + screen.height
                ):
                    target_screen = screen
                    break
            if target_screen is None and screens:
                target_screen = screens[0]
        except Exception:
            target_screen = None

        if target_screen is not None:
            self.set_fullscreen(True, screen=target_screen)
        else:
            self.set_fullscreen(True)

    def on_key_press(self, symbol: int, modifiers: int) -> None:  # noqa: ARG002
        if symbol == pyglet.window.key.ESCAPE:
            self.close()
        if symbol == pyglet.window.key.F:
            self._toggle_fullscreen()
        if symbol == pyglet.window.key.R and (modifiers & pyglet.window.key.MOD_SHIFT):
            self._reset_warp_grid()
        elif symbol == pyglet.window.key.R:
            self._toggle_recording()
        if symbol == pyglet.window.key.C:
            self._calibrate_neutral_face()
        if symbol == pyglet.window.key.V:
            self.view_rotation_index = (self.view_rotation_index + 1) % 4
        if symbol == pyglet.window.key.E:
            self.show_handles = not self.show_handles
        if symbol == pyglet.window.key.A:
            self.show_axes = not self.show_axes

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:  # noqa: ARG002
        if button != pyglet.window.mouse.LEFT:
            return
        self.active_global_corner = self._pick_global_corner(x, y)
        if self.active_global_corner is None:
            self.active_point = self._pick_corner(x, y)
        else:
            self.active_point = None

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:  # noqa: ARG002
        clamped = [
            float(np.clip(x, 0, max(self.width - 1, 0))),
            float(np.clip(y, 0, max(self.height - 1, 0))),
        ]
        if self.active_global_corner is not None:
            self.global_warp_points[self.active_global_corner] = clamped
            return
        if self.active_point is not None:
            self.warp_points[self.active_point] = clamped

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:  # noqa: ARG002
        if button == pyglet.window.mouse.LEFT:
            self.active_global_corner = None
            self.active_point = None

    def on_close(self) -> None:
        if self.recording_active:
            self.recording_active = False
            frames = list(self.recording_frames)
            self.recording_frames.clear()
            threading.Thread(target=self._prompt_and_save_recording, args=(frames,), daemon=True).start()
        self.tracker.close()
        self.camera.close()
        super().on_close()

    def _default_warp_points(self) -> list[list[float]]:
        return self._default_warp_grid_points()

    def _default_global_warp_points(self) -> list[list[float]]:
        face_w = float(self.width) * 0.46
        face_h = float(self.height) * 0.72
        base_left = (float(self.width) - face_w) * 0.5
        base_top = (float(self.height) - face_h) * 0.5
        margin_x = face_w * 0.22
        margin_y = face_h * 0.22
        left = base_left - margin_x
        top = base_top - margin_y
        w = face_w + margin_x * 2.0
        h = face_h + margin_y * 2.0
        viewport_w = max(float(self.width), 1.0)
        viewport_h = max(float(self.height), 1.0)
        safe_pad = min(40.0, viewport_w * 0.05, viewport_h * 0.05)

        if w > viewport_w - safe_pad * 2.0:
            w = max(viewport_w - safe_pad * 2.0, 2.0)
        if h > viewport_h - safe_pad * 2.0:
            h = max(viewport_h - safe_pad * 2.0, 2.0)

        left = float(np.clip(left, safe_pad, max(viewport_w - safe_pad - w, safe_pad)))
        top = float(np.clip(top, safe_pad, max(viewport_h - safe_pad - h, safe_pad)))
        return [
            [left, top],
            [left + w, top],
            [left + w, top + h],
            [left, top + h],
        ]

    def _default_warp_grid_points(self) -> list[list[float]]:
        face_w = float(self.width) * 0.46
        face_h = float(self.height) * 0.72
        left = (float(self.width) - face_w) * 0.5
        top = (float(self.height) - face_h) * 0.5
        points: list[list[float]] = []
        for row in range(WARP_GRID_ROWS):
            fy = row / max(WARP_GRID_ROWS - 1, 1)
            y = top + fy * face_h
            for col in range(WARP_GRID_COLS):
                fx = col / max(WARP_GRID_COLS - 1, 1)
                x = left + fx * face_w
                points.append([x, y])
        return points

    def _pick_corner(self, x: int, y: int) -> Optional[int]:
        points = np.array(self.warp_points, dtype=np.float32)
        cursor = np.array([x, y], dtype=np.float32)
        distances = np.linalg.norm(points - cursor, axis=1)
        index = int(np.argmin(distances))
        if distances[index] <= self.corner_pick_radius:
            return index
        return None

    def _pick_global_corner(self, x: int, y: int) -> Optional[int]:
        points = np.array(self.global_warp_points, dtype=np.float32)
        cursor = np.array([x, y], dtype=np.float32)
        distances = np.linalg.norm(points - cursor, axis=1)
        index = int(np.argmin(distances))
        if distances[index] <= self.corner_pick_radius * 1.4:
            return index
        return None

    def _make_global_warp_matrix(self) -> np.ndarray:
        src = np.array(self._default_global_warp_points(), dtype=np.float32)
        dst = np.array(self.global_warp_points, dtype=np.float32)
        src_ndc = self._window_to_ndc_points(src)
        dst_ndc = self._window_to_ndc_points(dst)
        return cv2.getPerspectiveTransform(src_ndc, dst_ndc).astype(np.float32)

    def _make_screen_warp_uniforms(self) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
        src = np.array(self._default_warp_grid_points(), dtype=np.float32)
        src_ndc = self._window_to_ndc_points(src)
        dst_ndc = self._window_to_ndc_points(np.array(self.warp_points, dtype=np.float32))

        src_min = np.array([np.min(src_ndc[:, 0]), np.min(src_ndc[:, 1])], dtype=np.float32)
        src_max = np.array([np.max(src_ndc[:, 0]), np.max(src_ndc[:, 1])], dtype=np.float32)

        padded = np.zeros((MAX_WARP_POINTS, 2), dtype=np.float32)
        point_count = dst_ndc.shape[0]
        if point_count > MAX_WARP_POINTS:
            raise ValueError(f"Warp points exceed MAX_WARP_POINTS={MAX_WARP_POINTS}")
        padded[:point_count] = dst_ndc
        return src_min, src_max, WARP_GRID_COLS, WARP_GRID_ROWS, padded

    def _window_to_ndc_points(self, points: np.ndarray) -> np.ndarray:
        w = max(float(self.width), 1.0)
        h = max(float(self.height), 1.0)
        ndc = np.empty((points.shape[0], 2), dtype=np.float32)
        ndc[:, 0] = (points[:, 0] / w) * 2.0 - 1.0
        ndc[:, 1] = (points[:, 1] / h) * 2.0 - 1.0
        return ndc

    def _constrain_warp_points(self) -> None:
        max_x = max(self.width - 1, 0)
        max_y = max(self.height - 1, 0)
        for point in self.warp_points:
            point[0] = float(np.clip(point[0], 0, max_x))
            point[1] = float(np.clip(point[1], 0, max_y))
        for point in self.global_warp_points:
            point[0] = float(np.clip(point[0], 0, max_x))
            point[1] = float(np.clip(point[1], 0, max_y))

    def _update_warp_handle_visuals(self) -> None:
        points = self.warp_points
        for index, (x, y) in enumerate(points):
            point = self.handle_points[index]
            point.x = float(x)
            point.y = float(y)
            point.color = (0, 255, 255) if index == self.active_point else (255, 255, 255)
            point.opacity = 220

        for index, (x, y) in enumerate(self.global_warp_points):
            point = self.global_handle_points[index]
            point.x = float(x)
            point.y = float(y)
            point.color = (255, 170, 60) if index == self.active_global_corner else (255, 210, 90)
            point.opacity = 220

        for index in range(4):
            line = self.global_handle_lines[index]
            x1, y1 = self.global_warp_points[index]
            x2, y2 = self.global_warp_points[(index + 1) % 4]
            line.x = float(x1)
            line.y = float(y1)
            line.x2 = float(x2)
            line.y2 = float(y2)
            line.color = (255, 170, 60) if self.active_global_corner == index else (200, 140, 60)

        line_index = 0
        for row in range(WARP_GRID_ROWS):
            for col in range(WARP_GRID_COLS - 1):
                i0 = row * WARP_GRID_COLS + col
                i1 = i0 + 1
                line = self.handle_lines[line_index]
                line_index += 1
                x1, y1 = points[i0]
                x2, y2 = points[i1]
                line.x = float(x1)
                line.y = float(y1)
                line.x2 = float(x2)
                line.y2 = float(y2)
                line.color = (120, 120, 120)

        for col in range(WARP_GRID_COLS):
            for row in range(WARP_GRID_ROWS - 1):
                i0 = row * WARP_GRID_COLS + col
                i1 = i0 + WARP_GRID_COLS
                line = self.handle_lines[line_index]
                line_index += 1
                x1, y1 = points[i0]
                x2, y2 = points[i1]
                line.x = float(x1)
                line.y = float(y1)
                line.x2 = float(x2)
                line.y2 = float(y2)
                line.color = (120, 120, 120)

    def _calibrate_neutral_face(self) -> None:
        self.neutral_weights = dict(self.raw_weights)
        self.weight_filters.clear()
        self.smoothed_weights = {}
        print(f"[DANYA] Neutral face calibrated with {len(self.neutral_weights)} weights.")

    @staticmethod
    def _nonlinear_weight(weight: float, deadzone: float = WEIGHT_DEADZONE, power: float = WEIGHT_CURVE_POWER) -> float:
        if weight <= deadzone:
            return 0.0
        span = max(1.0 - deadzone, 1e-6)
        normalized = float(np.clip((weight - deadzone) / span, 0.0, 1.0))
        return float(np.clip(normalized**power, 0.0, 1.0))

    def _apply_calibration(self, raw_weights: dict[str, float]) -> dict[str, float]:
        keys = set(raw_weights) | set(self.neutral_weights)
        adjusted: dict[str, float] = {}
        for key in keys:
            raw = float(raw_weights.get(key, 0.0))
            neutral = float(self.neutral_weights.get(key, 0.0))
            value = max(raw - neutral, 0.0)
            if value > 0.0:
                adjusted[key] = value
        return adjusted

    def _filter_weights(self, raw_weights: dict[str, float], timestamp: float) -> dict[str, float]:
        calibrated = self._apply_calibration(raw_weights)
        keys = set(calibrated) | set(self.weight_filters)
        filtered: dict[str, float] = {}
        for key in keys:
            raw_value = calibrated.get(key, 0.0)
            if key in EYE_DIRECT_KEYS:
                value = float(np.clip(raw_value, 0.0, 1.0))
                if value > 0.001:
                    filtered[key] = value
                continue
            if key in EYE_GAZE_KEYS:
                value = float(np.clip((raw_value ** 0.72) * 1.3, 0.0, 1.0))
                if value > 0.001:
                    filtered[key] = value
                continue
            # Separate left/right eye tracking with independent filtering
            if key in FAST_EYE_LEFT_KEYS:
                value = self._nonlinear_weight(raw_value, deadzone=0.008, power=1.08)
                params = (5.2, 0.032, 2.0)  # Faster, more responsive to left eye
            elif key in FAST_EYE_RIGHT_KEYS:
                value = self._nonlinear_weight(raw_value, deadzone=0.008, power=1.08)
                params = (10.0, 0.03, 1.0)  # Faster, more responsive to right eye
            elif key in MOUTH_DIRECT_KEYS:
                value = float(np.clip(raw_value, 0.0, 1.0))
                if value > 0.01:
                    filtered[key] = value
                continue
            elif key in FAST_VISEME_KEYS:
                value = float(np.clip(raw_value, 0.0, 1.0))
                if value > 0.01:
                    filtered[key] = value
                continue
            elif key in FAST_MOUTH_KEYS:
                value = float(np.clip(raw_value, 0.0, 1.0))
                if value > 0.01:
                    filtered[key] = value
                continue
            else:
                value = self._nonlinear_weight(raw_value)
                params = (ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
            filt = self.weight_filters.get(key)
            if filt is None:
                filt = OneEuroFilter(*params)
                self.weight_filters[key] = filt
            elif (abs(filt.min_cutoff - params[0]) > 1e-6 or abs(filt.beta - params[1]) > 1e-6 or abs(filt.d_cutoff - params[2]) > 1e-6):
                filt = OneEuroFilter(*params)
                self.weight_filters[key] = filt
            smoothed = filt.filter(value, timestamp)
            if smoothed > 0.01:
                filtered[key] = float(np.clip(smoothed, 0.0, 1.0))
        return filtered

    def _update_procedural_state(self, dt: float, now: float) -> None:
        self.breath_phase += float(dt) * (2.0 * math.pi * 0.085)
        self.breath_offset = math.sin(self.breath_phase) * 0.012

        if now >= self.next_saccade_time:
            self.saccade_offset = np.array(
                [random.uniform(-0.012, 0.012), random.uniform(-0.008, 0.008)],
                dtype=np.float32,
            )
            self.next_saccade_time = now + random.uniform(1.0, 3.0)
        else:
            self.saccade_offset *= float(np.exp(-dt * 9.0))
            if float(np.linalg.norm(self.saccade_offset)) < 1e-4:
                self.saccade_offset[:] = 0.0

    def _make_view_matrix(self) -> np.ndarray:
        base_eye = np.array([0.0, 0.10, 3.45], dtype=np.float32)
        base_target = np.array([0.0, 0.05, 0.0], dtype=np.float32)
        base_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        orbit_rad = math.radians(90.0 * float(self.view_rotation_index))
        # Rotate around Z axis and rotate camera orientation with the same transform.
        rotation = _build_rotation_matrix_z(orbit_rad)
        eye_offset = np.array([base_eye[0] - base_target[0], base_eye[1] - base_target[1], base_eye[2] - base_target[2], 1.0], dtype=np.float32)
        rotated_offset = (rotation @ eye_offset)[:3]
        rotated_up = (rotation @ np.array([base_up[0], base_up[1], base_up[2], 0.0], dtype=np.float32))[:3]

        eye = base_target + rotated_offset
        eye[1] += self.breath_offset
        target = base_target.copy()
        target[0] += float(self.saccade_offset[0])
        target[1] += float(self.saccade_offset[1]) + self.breath_offset * 0.35

        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        up = rotated_up / max(np.linalg.norm(rotated_up), 1e-6)
        side = np.cross(forward, up)
        side = side / np.linalg.norm(side)
        up = np.cross(side, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = side
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ eye
        return view

    @staticmethod
    def _pose_to_matrix(pose: tuple[float, float, float]) -> np.ndarray:
        yaw, pitch, roll = pose
        # X-axis rotation for yaw (horizontal head turn)
        # Y-axis rotation for pitch (vertical head tilt)
        yaw_m = _build_rotation_matrix_x(math.radians(yaw * 0.6))
        pitch_m = _build_rotation_matrix_y(math.radians(pitch * 0.6))
        roll_m = _build_rotation_matrix_z(math.radians(roll * 0.4))
        return pitch_m @ yaw_m @ roll_m

    def _update_pip_texture(self, frame_bgr: np.ndarray, landmarks: list[tuple[float, float]]) -> None:
        pip_frame = cv2.resize(frame_bgr, self.pip_size, interpolation=cv2.INTER_AREA)

        if landmarks:
            pip_w, pip_h = self.pip_size

            def _to_px(index: int) -> tuple[int, int] | None:
                if index < 0 or index >= len(landmarks):
                    return None
                lx, ly = landmarks[index]
                x = int(np.clip(lx * (pip_w - 1), 0, pip_w - 1))
                y = int(np.clip(ly * (pip_h - 1), 0, pip_h - 1))
                return (x, y)

            for a, b in FACEMESH_TESSELATION:
                pa = _to_px(int(a))
                pb = _to_px(int(b))
                if pa is not None and pb is not None:
                    cv2.line(pip_frame, pa, pb, (70, 255, 170), 1, lineType=cv2.LINE_AA)

            for a, b in FACEMESH_LIPS:
                pa = _to_px(int(a))
                pb = _to_px(int(b))
                if pa is not None and pb is not None:
                    cv2.line(pip_frame, pa, pb, (80, 80, 255), 2, lineType=cv2.LINE_AA)

            # Fallback: if connection tables are unavailable, still show face shape points.
            if not FACEMESH_TESSELATION:
                for i in range(0, len(landmarks), 2):
                    p = _to_px(i)
                    if p is not None:
                        cv2.circle(pip_frame, p, 1, (70, 255, 170), -1, lineType=cv2.LINE_AA)

        image = pyglet.image.ImageData(
            self.pip_size[0],
            self.pip_size[1],
            "BGR",
            pip_frame.tobytes(),
            pitch=-self.pip_size[0] * 3,
        )
        self.pip_texture.blit_into(image, 0, 0, 0)

    def _draw_camera_pip(self) -> None:
        x = max(self.width - PIP_WIDTH - PIP_PADDING, PIP_PADDING)
        y = PIP_PADDING
        self.pip_background.x = x - 4
        self.pip_background.y = y - 4
        self.pip_sprite.x = x
        self.pip_sprite.y = y
        glDisable(GL_DEPTH_TEST)
        self.pip_background.draw()
        self.pip_sprite.draw()
        glEnable(GL_DEPTH_TEST)

    def _toggle_recording(self) -> None:
        if self.recording_active:
            self.recording_active = False
            frames = list(self.recording_frames)
            self.recording_frames.clear()
            if not self.recording_save_prompt_active:
                self.recording_save_prompt_active = True
                threading.Thread(target=self._prompt_and_save_recording, args=(frames,), daemon=True).start()
            return

        self.recording_active = True
        self.recording_started_at = time.monotonic()
        self.recording_frames.clear()
        print("[DANYA] Recording started.")

    def _append_recording_frame(self, now: float, weights: dict[str, float], pose: tuple[float, float, float]) -> None:
        frame = {
            "time": float(now - self.recording_started_at),
            "weights": {key: float(value) for key, value in weights.items()},
            "pose": {
                "yaw": float(pose[0]),
                "pitch": float(pose[1]),
                "roll": float(pose[2]),
            },
        }
        with self.recording_lock:
            self.recording_frames.append(frame)

    def _prompt_and_save_recording(self, frames: list[dict[str, Any]]) -> None:
        try:
            print("[DANYA] Recording stopped. Enter a file name to save the JSON:", flush=True)
            filename = input().strip()
        except EOFError:
            filename = ""
        finally:
            self.recording_save_prompt_active = False

        safe_name = Path(filename).stem if filename else "motion_recording"
        save_dir = BASE_DIR / "runtime" / "motion_records"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{safe_name}.json"
        payload = {"frames": frames}
        try:
            with save_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            print(f"[DANYA] Saved recording to {save_path}")
        except Exception as exc:
            print(f"[DANYA] Failed to save recording: {exc}")

    def _reset_warp_grid(self) -> None:
        self.global_warp_points = self._default_global_warp_points()
        self.warp_points = self._default_warp_grid_points()

    def _make_projection_matrix(self) -> np.ndarray:
        aspect = max(self.width, 1.0) / max(self.height, 1.0)
        fov = math.radians(26.0)
        near = 0.01
        far = 100.0
        f = 1.0 / math.tan(fov / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj


def main() -> None:
    app = AvatarApp()
    pyglet.app.run()
    pyglet.app.run()


if __name__ == "__main__":
    main()
