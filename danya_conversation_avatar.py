from __future__ import annotations

import argparse
import base64
import bisect
import concurrent.futures
import io
import json
import math
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import os
import queue
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyglet
import soundfile as sf
from PIL import Image
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
    glActiveTexture,
    glBlendFunc,
    glBindTexture,
    glClearColor,
    glDisable,
    glEnable,
    glTexParameteri,
)
from pyglet.graphics.shader import Shader, ShaderProgram

from gpt_sovits_tts_client import play_audio, synthesize_audio

WINDOW_TITLE = "DANYA Avatar"
DISPLAY_ROTATION_DEG = 90.0
WARP_GRID_COLS = 5
WARP_GRID_ROWS = 5
MAX_WARP_POINTS = 64
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "assets" / "models" / "avatar.glb"
# アニメーション保存/再生の既定パス（runtime配下）
ANIMATION_DATA_PATH = BASE_DIR / "runtime" / "motion_records" / "animation_data.json"
# 旧パス互換: 既存データがある場合はこちらも読み込み対象にする
LEGACY_ANIMATION_DATA_PATH = BASE_DIR / "data" / "animation_data.json"
MOTION_RECORD_DIR = BASE_DIR / "runtime" / "motion_records"
EMOTION_MOTION_ALIASES = {
    "happy": "smile",
}
DEFAULT_TTS_SERVER = os.environ.get("DANYA_TTS_SERVER", "http://192.168.73.239:8000")
DEFAULT_TTS_OUTPUT = BASE_DIR / "runtime" / "output.wav"
TTS_SEGMENT_DIR = BASE_DIR / "runtime" / "tts_segments"
DEFAULT_AUDIO_DEVICE = os.environ.get("DANYA_AUDIO_DEVICE", "").strip() or None
DEFAULT_TTS_REF_ID = os.environ.get("DANYA_TTS_REF_ID", "").strip() or None
DEFAULT_LLM_OUTPUT_SERVER = os.environ.get("DANYA_LLM_OUTPUT_SERVER", "http://127.0.0.1:8767").strip()
DEFAULT_LLM_OUTPUT_INTERVAL = float(os.environ.get("DANYA_LLM_OUTPUT_INTERVAL", "20"))
DEFAULT_LLM_OUTPUT_DEBUG = os.environ.get("DANYA_LLM_OUTPUT_DEBUG", "1").strip().lower() not in {"0", "false", "off", "no"}
YOLO_TRACKING_ENABLED = os.environ.get("DANYA_YOLO_TRACKING", "1").strip().lower() not in {"0", "false", "off", "no"}
YOLO_MODEL_PATH = os.environ.get("DANYA_YOLO_MODEL", "yolov8n.pt")
YOLO_CAMERA_INDEX = int(os.environ.get("DANYA_YOLO_CAMERA", "0"))
YOLO_CONFIDENCE = float(os.environ.get("DANYA_YOLO_CONF", "0.45"))
YOLO_FRAME_SKIP = max(1, int(os.environ.get("DANYA_YOLO_FRAME_SKIP", "2")))
YOLO_PREVIEW_ENABLED = os.environ.get("DANYA_YOLO_PREVIEW", "0").strip().lower() not in {"0", "false", "off", "no"}
YOLO_PREVIEW_WIDTH = int(os.environ.get("DANYA_YOLO_PREVIEW_WIDTH", "960"))
YOLO_PREVIEW_HEIGHT = int(os.environ.get("DANYA_YOLO_PREVIEW_HEIGHT", "540"))
YOLO_TARGET_MAX_AGE_SEC = 1.2
PERSON_LOOK_SIDE_YAW_DEG = 16.0
PERSON_LOOK_CENTER_DEADZONE = 0.18
PERSON_LOOK_BLEND_WHEN_SPEAKING = 0.75
DEFAULT_EMOTION_INTENSITY = "normal"
EMOTION_REF_PREFIXES = {"happy", "angry", "sad", "surprised", "fear"}
EMOTION_REF_ALIASES = {
    "smile": "happy",
    "surprise": "surprised",
    "scared": "fear",
    "afraid": "fear",
}
EMOTION_LEVEL_ALIASES = {
    "mid": "normal",
    "low": "normal",
}
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 8765
WINDOW_STATE_PATH = BASE_DIR / ".cache" / "window_state.json"
LIPSYNC_FRAME_SEC = 0.032
LIPSYNC_HOP_SEC = 0.010
SPEECH_FACE_BLEND_IN_SEC = 0.55
SPEECH_FACE_BLEND_OUT_SEC = 0.45
SPEECH_BASE_FACE_KEEP = 0.28
SPEECH_BASE_MOUTH_KEEP = 0.12

SPEECH_LIPSYNC_KEYS = {
    "jawopen",
    "mouthopen",
    "mouthpucker",
}

EXPRESSION_LIPSYNC_OVERRIDE_KEYS = {
    "jawopen",
    "mouthopen",
}

FACE_MESHES = {
    "Head_Mesh",
    "Eye_Mesh",
    "Teeth_Mesh",
    "Tongue_Mesh",
    "avaturn_hair_0",
    "avaturn_hair_1",
}

ALIAS_WEIGHTS = {
    "jawopen": ["mouthopen"],
    "mouthopen": ["jawopen"],
    "mouthsmile": ["mouthsmileleft", "mouthsmileright"],
    "mouthsmileleft": ["mouthsmile"],
    "mouthsmileright": ["mouthsmile"],
    "eyesclosed": ["eyeblinkleft", "eyeblinkright"],
    "eyeblinkleft": ["eyesclosed"],
    "eyeblinkright": ["eyesclosed"],
    "browinnerup": ["browraise"],
    "browraise": ["browinnerup"],
}
class ExternalControlServer(threading.Thread):
    def __init__(self, inbox: queue.Queue[str], host: str = CONTROL_HOST, port: int = CONTROL_PORT) -> None:
        super().__init__(daemon=True)
        self.inbox = inbox
        self.host = host
        self.port = port
        self.running = True
        self.sock: Optional[socket.socket] = None

    def run(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
            server.listen(5)
            server.settimeout(0.5)
            self.sock = server
        except OSError as exc:
            print(f"[CONTROL ERROR] Could not bind {self.host}:{self.port}: {exc}")
            try:
                server.close()
            except OSError:
                pass
            return
        while self.running:
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                chunks: list[bytes] = []
                while True:
                    part = conn.recv(4096)
                    if not part:
                        break
                    chunks.append(part)

                payload = b"".join(chunks).decode("utf-8", errors="ignore")
                for raw_line in payload.splitlines():
                    line = raw_line.strip()
                    if line:
                        self.inbox.put(line)

    def stop(self) -> None:
        self.running = False
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass


class LLMOutputReceiver(threading.Thread):
    TAG_PATTERN = re.compile(r"^\s*<([^>]+)>\s*(.*)$", re.DOTALL)

    def __init__(
        self,
        inbox: queue.Queue[str],
        server_url: str,
        interval_sec: float = DEFAULT_LLM_OUTPUT_INTERVAL,
    ) -> None:
        super().__init__(daemon=True)
        self.inbox = inbox
        self.server_url = server_url.rstrip("/")
        self.interval_sec = max(1.0, float(interval_sec))
        self.seq = 0
        self.running = True
        self.debug = DEFAULT_LLM_OUTPUT_DEBUG

    def run(self) -> None:
        try:
            import requests
        except Exception as exc:
            print(f"[LLM OUTPUT WARN] requests is not available: {exc}")
            return

        print(f"[LLM OUTPUT] polling {self.server_url}/api/output every {self.interval_sec:.0f}s")
        while self.running:
            try:
                response = requests.get(
                    f"{self.server_url}/api/output",
                    params={"since": self.seq},
                    timeout=min(25.0, max(2.0, self.interval_sec * 0.8)),
                )
                response.raise_for_status()
                data = response.json()
                outputs = data.get("outputs", [])
                latest_seq = int(data.get("latest_seq", self.seq))
                if self.debug:
                    print(
                        f"[LLM OUTPUT] poll ok since={self.seq} count={len(outputs)} latest_seq={latest_seq}"
                    )
                for text in outputs:
                    if self.debug:
                        preview = str(text).replace("\n", " ")[:80]
                        print(f"[LLM OUTPUT] received: {preview}")
                    payload = self._payload_from_output(str(text))
                    if payload:
                        self.inbox.put(payload)
                self.seq = max(self.seq, latest_seq)
            except requests.exceptions.Timeout:
                print(f"[LLM OUTPUT WARN] timeout polling {self.server_url}/api/output")
            except Exception as exc:
                print(f"[LLM OUTPUT WARN] {exc}")

            slept = 0.0
            while self.running and slept < self.interval_sec:
                step = min(0.5, self.interval_sec - slept)
                time.sleep(step)
                slept += step

    def stop(self) -> None:
        self.running = False

    @classmethod
    def _payload_from_output(cls, output: str) -> str:
        text = output.strip()
        if not text:
            return ""

        match = cls.TAG_PATTERN.match(text)
        if not match:
            return json.dumps({"source": "llm_output", "segments": [{"text": text}]}, ensure_ascii=False)

        tag = match.group(1).strip()
        body = match.group(2).strip()
        if not body:
            return ""

        emotion, intensity = cls._split_emotion_tag(tag)
        segment: dict[str, str] = {"text": body}
        if emotion:
            segment["emotion"] = emotion
        if intensity:
            segment["intensity"] = intensity
        return json.dumps({"source": "llm_output", "segments": [segment]}, ensure_ascii=False)

    @staticmethod
    def _split_emotion_tag(tag: str) -> tuple[str, str]:
        normalized = tag.replace("-", "_").strip().lower()
        if "|" in normalized:
            emotion, intensity = normalized.split("|", 1)
            return emotion.strip(), intensity.strip()

        parts = [part for part in normalized.split("_") if part]
        if len(parts) >= 2 and parts[-1] in {"high", "normal", "mid", "low"}:
            return "_".join(parts[:-1]), parts[-1]
        return normalized, ""


class PersonTracker(threading.Thread):
    def __init__(
        self,
        enabled: bool = YOLO_TRACKING_ENABLED,
        camera_index: int = YOLO_CAMERA_INDEX,
        model_path: str = YOLO_MODEL_PATH,
        confidence: float = YOLO_CONFIDENCE,
        frame_skip: int = YOLO_FRAME_SKIP,
        preview: bool = YOLO_PREVIEW_ENABLED,
    ) -> None:
        super().__init__(daemon=True)
        self.enabled = enabled
        self.camera_index = camera_index
        self.model_path = model_path
        self.confidence = confidence
        self.frame_skip = frame_skip
        self.preview = preview
        self.running = True
        self.lock = threading.Lock()
        self.target: Optional[dict[str, float]] = None

    def run(self) -> None:
        if not self.enabled:
            return
        try:
            import cv2
            from ultralytics import YOLO
        except Exception as exc:
            print(f"[YOLO WARN] Person tracking disabled: {exc}")
            return

        try:
            model = YOLO(self.model_path)
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                print(f"[YOLO WARN] Could not open camera index {self.camera_index}")
                return
        except Exception as exc:
            print(f"[YOLO WARN] Could not start person tracker: {exc}")
            return

        print(f"[YOLO] Person tracking started: camera={self.camera_index}, model={self.model_path}")
        frame_index = 0
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frame_index += 1
                if frame_index % self.frame_skip != 0:
                    if self.preview:
                        self._show_preview(cv2, frame, None)
                    continue
                best = self._detect_person(model, frame)
                if self.preview:
                    self._show_preview(cv2, frame, best)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False
        finally:
            try:
                cap.release()
            except Exception:
                pass
            if self.preview:
                try:
                    cv2.destroyWindow("DANYA YOLO Person View")
                except Exception:
                    pass

    def stop(self) -> None:
        self.running = False

    def get_target(self) -> Optional[dict[str, float]]:
        with self.lock:
            if not self.target:
                return None
            age = time.time() - self.target.get("seen_at", 0.0)
            if age > YOLO_TARGET_MAX_AGE_SEC:
                return None
            return dict(self.target)

    def _detect_person(self, model: Any, frame: Any) -> Optional[tuple[float, float, float, float]]:
        height, width = frame.shape[:2]
        try:
            results = model.predict(frame, classes=[0], conf=self.confidence, verbose=False)
        except Exception:
            return None

        best = None
        best_area = 0.0
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy()
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2)

        if best is None:
            return None

        x1, y1, x2, y2 = best
        center_x = ((x1 + x2) * 0.5) / max(width, 1)
        center_y = ((y1 + y2) * 0.5) / max(height, 1)
        with self.lock:
            self.target = {
                "x": float(np.clip(center_x, 0.0, 1.0)),
                "y": float(np.clip(center_y, 0.0, 1.0)),
                "area": float(best_area / max(width * height, 1)),
                "seen_at": time.time(),
            }
        return best

    def _show_preview(self, cv2: Any, frame: Any, box: Optional[tuple[float, float, float, float]]) -> None:
        height, width = frame.shape[:2]
        left_line = int(width * (0.5 - PERSON_LOOK_CENTER_DEADZONE))
        right_line = int(width * (0.5 + PERSON_LOOK_CENTER_DEADZONE))
        cv2.line(frame, (left_line, 0), (left_line, height), (80, 220, 255), 2)
        cv2.line(frame, (right_line, 0), (right_line, height), (80, 220, 255), 2)

        label = "no person"
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cx = (x1 + x2) // 2
            if cx < left_line:
                label = "person: LEFT"
            elif cx > right_line:
                label = "person: RIGHT"
            else:
                label = "person: CENTER"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 3)
            cv2.circle(frame, (cx, (y1 + y2) // 2), 7, (0, 255, 120), -1)

        cv2.putText(
            frame,
            label,
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 120) if box is not None else (80, 80, 255),
            3,
            cv2.LINE_AA,
        )
        preview = cv2.resize(frame, (YOLO_PREVIEW_WIDTH, YOLO_PREVIEW_HEIGHT))
        cv2.imshow("DANYA YOLO Person View", preview)

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
class SpeechSegment:
    text: str
    ref_id: Optional[str] = None

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
            out vec3 v_normal;
            out vec2 v_texcoord;

            void main() {
                vec4 world_pos = model * vec4(position, 1.0);
                v_normal = mat3(transpose(inverse(model))) * normal;
                v_texcoord = vec2(texcoord.x, 1.0 - texcoord.y);
                gl_Position = projection * view * world_pos;
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

    @staticmethod
    def _update_vertex_list(part: MeshPart, positions: np.ndarray, normals: np.ndarray) -> None:
        part.vertex_list.set_attribute_data("position", positions.astype(np.float32).reshape(-1))
        part.vertex_list.set_attribute_data("normal", normals.astype(np.float32).reshape(-1))

    def draw(
        self,
        weights: dict[str, float],
        view: np.ndarray,
        projection: np.ndarray,
    ) -> None:
        self.program.use()
        self.program["model"] = tuple(self.model_matrix.T.reshape(-1))
        self.program["view"] = tuple(view.T.reshape(-1))
        self.program["projection"] = tuple(projection.T.reshape(-1))
        self.program["light_dir"] = (0.3, 0.7, 0.8)
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
    def __init__(
        self,
        launch_terminal: bool = True,
        llm_output_server: Optional[str] = DEFAULT_LLM_OUTPUT_SERVER,
        llm_output_interval: float = DEFAULT_LLM_OUTPUT_INTERVAL,
    ) -> None:
        state = self._load_window_state()
        config = pyglet.gl.Config(double_buffer=True, depth_size=24)
        super().__init__(
            caption=WINDOW_TITLE,
            width=int(state.get("width", 1280)),
            height=int(state.get("height", 720)),
            resizable=True,
            fullscreen=bool(state.get("fullscreen", False)),
            config=config,
            vsync=True,
        )
        self.set_mouse_visible(True)
        if "x" in state and "y" in state and not self.fullscreen:
            try:
                self.set_location(int(state["x"]), int(state["y"]))
            except Exception:
                pass

        self.avatar = GLBAvatar(MODEL_PATH)
        self.avatar_base_model_matrix = self.avatar.model_matrix.copy()
        
        # モードと記録データの追加 ("IDLE", "TRACK", "RECORD", "PLAY")
        self.mode = "IDLE" 
        self.record_data = []
        self.record_start_time = 0.0
        self.play_data = []
        self.play_start_time = 0.0
        self.play_pause_elapsed = 0.0
        self.play_paused_for_speech = False
        self.speech_face_reset = 0.0
        self.speech_emotion_blend = 0.0
        self.speech_emotion_started_at = 0.0
        self.speech_emotion: Optional[str] = None
        self.animation_duration = 0.0  # アニメーション全体の長さ
        self.loop_transition_duration = 0.3  # ループ時の遷移時間（秒）
        self.last_frame_data: dict[str, Any] = {}  # ループ時のブレンド用
        self.expression_data = self._load_expression_recordings()

        self.smoothed_weights: dict[str, float] = {}
        self.tts_server = DEFAULT_TTS_SERVER
        self.tts_output = DEFAULT_TTS_OUTPUT
        self.audio_device = DEFAULT_AUDIO_DEVICE
        self.tts_ref_id = DEFAULT_TTS_REF_ID
        self.direct_tts_mode = False
        self.state_lock = threading.Lock()
        self.speech_queue: queue.Queue[Optional[list[SpeechSegment]]] = queue.Queue()
        self.control_inbox: queue.Queue[str] = queue.Queue()
        
        self.is_speaking = False
        self.speech_motion = 0.0
        
        # 姿勢情報
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        self.head_roll = 0.0
        self.person_look_yaw = 0.0
        self.person_look_strength = 0.0
        self.view_rotation_index = 0

        self.lipsync_active = False
        self.lipsync_start_time = 0.0
        self.lipsync_index = 0
        self.lipsync_timeline: list[tuple[float, dict[str, float]]] = []
        self.app_start = time.time()
        
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        self.control_server = ExternalControlServer(self.control_inbox)
        self.control_server.start()
        self.llm_output_receiver: Optional[LLMOutputReceiver] = None
        if llm_output_server:
            self.llm_output_receiver = LLMOutputReceiver(
                self.control_inbox,
                llm_output_server,
                interval_sec=llm_output_interval,
            )
            self.llm_output_receiver.start()
        self.person_tracker = PersonTracker()
        self.person_tracker.start()

        if self._load_recording():
            self.mode = "PLAY"
            self.play_start_time = time.time()
            print(f"[MODE] PLAY auto-started from {ANIMATION_DATA_PATH.name}.")
        
        if launch_terminal:
            launch_control_terminal()
            
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

        print("\n--- 操作方法 ---")
        print("T キー: トラッキング機能は無効")
        print("R キー: 録画機能は無効")
        print("P キー: 再生 (Play) モードの 開始/停止")
        print("control terminal: 'tts on' で入力文をそのまま読み上げ、'tts off' で戻る")
        print("control terminal: 'ref happy_normal' や 'ref happy_high' のように参照音声を切り替え")
        print("YOLO: 人物検出が有効なら、発話中に相手の方向を向く")
        print("----------------\n")

    def update(self, dt: float) -> None:
        self.smoothed_weights = self._decay_weights(self.smoothed_weights)
        now = time.time()

        with self.state_lock:
            speaking_now = self.is_speaking
            speech_emotion = self.speech_emotion
            speech_emotion_started_at = self.speech_emotion_started_at
        target_motion = 1.0 if speaking_now else 0.0
        self.speech_motion += (target_motion - self.speech_motion) * 0.12
        if speaking_now:
            face_target = min(1.0, max(0.0, (now - speech_emotion_started_at) / SPEECH_FACE_BLEND_IN_SEC))
            face_gain = 0.16
        else:
            face_target = 0.0
            face_gain = 0.10
        self.speech_face_reset += (face_target - self.speech_face_reset) * face_gain

        has_expression = speech_emotion in self.expression_data
        target_expression_blend = 1.0 if speaking_now and has_expression else 0.0
        expression_gain = 0.11 if target_expression_blend > self.speech_emotion_blend else 0.08
        self.speech_emotion_blend += (target_expression_blend - self.speech_emotion_blend) * expression_gain
        if not speaking_now and self.speech_emotion_blend < 0.01:
            self.speech_emotion_blend = 0.0
            with self.state_lock:
                if not self.is_speaking:
                    self.speech_emotion = None

        target_yaw, target_pitch, target_roll = 0.0, 0.0, 0.0

        # 再生モード
        if self.mode == "PLAY":
            if not self.play_data:
                self.mode = "IDLE"
            else:
                if speaking_now:
                    if not self.play_paused_for_speech:
                        self.play_pause_elapsed = max(0.0, now - self.play_start_time)
                        self.play_paused_for_speech = True
                    elapsed = self.play_pause_elapsed
                else:
                    if self.play_paused_for_speech:
                        self.play_start_time = now - self.play_pause_elapsed
                        self.play_paused_for_speech = False
                    elapsed = now - self.play_start_time
                
                # ループ処理（自然な遷移付き）
                if self.animation_duration > 0 and elapsed > self.animation_duration:
                    # ループ時の遷移ゾーン内かチェック
                    transition_zone_start = self.animation_duration - self.loop_transition_duration
                    
                    if elapsed - self.animation_duration < self.loop_transition_duration:
                        # 遷移ゾーン内：最後のフレームから最初のフレームへ自然にブレンド
                        blend_progress = (elapsed - self.animation_duration) / self.loop_transition_duration
                        frame_data = self._blend_frame_data(
                            self.last_frame_data,
                            self.play_data[0],
                            blend_progress
                        )
                    else:
                        # 遷移ゾーン外：ループをリセット
                        self.play_start_time = now
                        elapsed = 0.0
                        frame_data = self._get_interpolated_frame(elapsed)
                else:
                    frame_data = self._get_interpolated_frame(elapsed)
                
                for k, v in frame_data["weights"].items():
                    if speaking_now and self._is_base_speech_mouth_key(k):
                        continue
                    self.smoothed_weights[k] = max(self.smoothed_weights.get(k, 0.0), v)

                if speaking_now:
                    self._suppress_base_mouth_motion(self.speech_face_reset)
                    self._soft_reset_speech_face(self.speech_face_reset)

                target_yaw = frame_data["pose"]["yaw"]
                target_pitch = frame_data["pose"]["pitch"]
                target_roll = frame_data["pose"]["roll"]

                if speaking_now:
                    pose_release = float(np.clip(self.speech_face_reset, 0.0, 1.0))
                    target_yaw *= 1.0 - pose_release
                    target_pitch *= 1.0 - pose_release
                    target_roll *= 1.0 - pose_release

        expression_pose = None
        if speech_emotion and self.speech_emotion_blend > 0.0:
            expression_elapsed = max(0.0, now - speech_emotion_started_at)
            expression_pose = self._apply_emotion_expression(
                speech_emotion,
                expression_elapsed,
                self.speech_emotion_blend,
            )
        if expression_pose is not None:
            pose_blend = float(np.clip(self.speech_emotion_blend, 0.0, 1.0))
            target_yaw += (float(expression_pose.get("yaw", 0.0)) - target_yaw) * pose_blend
            target_pitch += (float(expression_pose.get("pitch", 0.0)) - target_pitch) * pose_blend
            target_roll += (float(expression_pose.get("roll", 0.0)) - target_roll) * pose_blend

        if speaking_now:
            if not self._apply_lipsync_weights_from_timeline():
                talk_wave = 0.30 + 0.55 * (0.5 + 0.5 * math.sin((now - self.app_start) * 12.0))
                self.smoothed_weights["jawopen"] = max(self.smoothed_weights.get("jawopen", 0.0), talk_wave)
                self.smoothed_weights["mouthopen"] = max(self.smoothed_weights.get("mouthopen", 0.0), talk_wave * 0.88)

        person_pose = self._get_person_look_pose(speaking_now)
        if person_pose is not None:
            person_yaw, person_strength = person_pose
            target_yaw += (person_yaw - target_yaw) * person_strength

        # 姿勢をスムージングしながら適用
        head_gain = 0.2
        self.head_yaw += (target_yaw - self.head_yaw) * head_gain
        self.head_pitch += (target_pitch - self.head_pitch) * head_gain
        self.head_roll += (target_roll - self.head_roll) * head_gain

        self.avatar.model_matrix = self._make_head_neck_pose_matrix() @ self.avatar_base_model_matrix
        self._drain_external_commands()

    def _soft_reset_speech_face(self, strength: float) -> None:
        """話している間、ループ中断時の顔を自然な正面に戻す。"""
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return

        keep = SPEECH_LIPSYNC_KEYS
        keep_factor = 1.0 - (1.0 - SPEECH_BASE_FACE_KEEP) * strength
        for key in list(self.smoothed_weights.keys()):
            if key in keep:
                continue
            self.smoothed_weights[key] *= keep_factor
            if self.smoothed_weights[key] < 0.01:
                self.smoothed_weights.pop(key, None)

    def _get_person_look_pose(self, speaking_now: bool) -> Optional[tuple[float, float]]:
        target = self.person_tracker.get_target() if hasattr(self, "person_tracker") else None
        desired_strength = PERSON_LOOK_BLEND_WHEN_SPEAKING if speaking_now and target else 0.0
        if target:
            x = float(target.get("x", 0.5))
            if x < 0.5 - PERSON_LOOK_CENTER_DEADZONE:
                desired_yaw = PERSON_LOOK_SIDE_YAW_DEG
            elif x > 0.5 + PERSON_LOOK_CENTER_DEADZONE:
                desired_yaw = -PERSON_LOOK_SIDE_YAW_DEG
            else:
                desired_yaw = 0.0
        else:
            desired_yaw = 0.0

        gain = 0.08 if desired_strength > self.person_look_strength else 0.05
        self.person_look_yaw += (desired_yaw - self.person_look_yaw) * gain
        self.person_look_strength += (desired_strength - self.person_look_strength) * gain

        if self.person_look_strength < 0.01:
            return None
        return self.person_look_yaw, self.person_look_strength

    @staticmethod
    def _is_base_speech_mouth_key(key: str) -> bool:
        return key == "jawopen" or key.startswith("mouth") or key.startswith("tongue")

    def _suppress_base_mouth_motion(self, strength: float) -> None:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return
        keep_factor = 1.0 - (1.0 - SPEECH_BASE_MOUTH_KEEP) * strength
        for key in list(self.smoothed_weights.keys()):
            if not self._is_base_speech_mouth_key(key):
                continue
            self.smoothed_weights[key] *= keep_factor
            if self.smoothed_weights[key] < 0.01:
                self.smoothed_weights.pop(key, None)

    def _get_interpolated_frame(self, elapsed: float) -> dict[str, Any]:
        """再生データから経過時間に応じたフレームを補間して取得"""
        if not self.play_data:
            return {"weights": {}, "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}}
        if len(self.play_data) == 1:
            return self.play_data[0]
            
        times = [d["time"] for d in self.play_data]
        idx = bisect.bisect_right(times, elapsed)
        
        if idx == 0:
            return self.play_data[0]
        if idx >= len(self.play_data):
            return self.play_data[-1]
            
        f0 = self.play_data[idx - 1]
        f1 = self.play_data[idx]
        
        t0, t1 = f0["time"], f1["time"]
        ratio = (elapsed - t0) / max((t1 - t0), 1e-6)
        
        interp_weights = {}
        for k in set(f0["weights"].keys()).union(f1["weights"].keys()):
            v0 = f0["weights"].get(k, 0.0)
            v1 = f1["weights"].get(k, 0.0)
            interp_weights[k] = v0 + (v1 - v0) * ratio
            
        interp_pose = {}
        for k in ["yaw", "pitch", "roll"]:
            v0 = f0["pose"][k]
            v1 = f1["pose"][k]
            interp_pose[k] = v0 + (v1 - v0) * ratio
            
        return {"weights": interp_weights, "pose": interp_pose}

    def _blend_frame_data(self, frame1: dict[str, Any], frame2: dict[str, Any], blend: float) -> dict[str, Any]:
        """２つのフレームデータをブレンド（blend=0で frame1、blend=1で frame2）"""
        blend = max(0.0, min(1.0, blend))
        
        # ウェイトをブレンド
        blended_weights = {}
        all_keys = set(frame1.get("weights", {}).keys()).union(frame2.get("weights", {}).keys())
        for k in all_keys:
            v1 = frame1.get("weights", {}).get(k, 0.0)
            v2 = frame2.get("weights", {}).get(k, 0.0)
            blended_weights[k] = v1 + (v2 - v1) * blend
        
        # ポーズをブレンド
        blended_pose = {}
        for k in ["yaw", "pitch", "roll"]:
            v1 = frame1.get("pose", {}).get(k, 0.0)
            v2 = frame2.get("pose", {}).get(k, 0.0)
            blended_pose[k] = v1 + (v2 - v1) * blend
        
        return {"weights": blended_weights, "pose": blended_pose}

    def _load_expression_recordings(self) -> dict[str, dict[str, Any]]:
        expression_data: dict[str, dict[str, Any]] = {}
        legacy_dir = BASE_DIR / "runtime" / "motion_record"
        candidate_dirs = [MOTION_RECORD_DIR, legacy_dir]
        for motion_dir in candidate_dirs:
            if not motion_dir.exists():
                continue
            for target_path in sorted(motion_dir.glob("*.json")):
                emotion = target_path.stem.lower()
                if emotion == "animation_data" or emotion in expression_data:
                    continue
                self._load_one_expression_recording(expression_data, emotion, target_path)

        for alias, target in EMOTION_MOTION_ALIASES.items():
            if alias not in expression_data and target in expression_data:
                expression_data[alias] = expression_data[target]
        return expression_data

    @staticmethod
    def _load_one_expression_recording(
        expression_data: dict[str, dict[str, Any]],
        emotion: str,
        target_path: Path,
    ) -> None:
        try:
            data = json.loads(target_path.read_text(encoding="utf-8"))
            frames = data.get("frames", [])
            if not frames:
                print(f"[MOTION WARN] Empty expression motion: {target_path}")
                return
            expression_data[emotion] = {
                "frames": frames,
                "duration": float(frames[-1].get("time", 0.0)),
            }
            print(f"[MOTION] Loaded {emotion} expression: {target_path.name}")
        except Exception as exc:
            print(f"[MOTION WARN] Failed to load {target_path}: {exc}")

    @staticmethod
    def _motion_key_from_ref_id(ref_id: Optional[str]) -> Optional[str]:
        ref = (ref_id or DEFAULT_TTS_REF_ID or "happy_high").strip().lower()
        if not ref:
            return None
        if ref.endswith(".wav"):
            ref = ref[:-4]
        emotion = ref.split("_", 1)[0]
        return EMOTION_MOTION_ALIASES.get(emotion, emotion)

    def _get_interpolated_motion_frame(
        self,
        frames: list[dict[str, Any]],
        elapsed: float,
        duration: float,
        loop: bool = True,
    ) -> dict[str, Any]:
        if not frames:
            return {"weights": {}, "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}}
        if len(frames) == 1 or duration <= 0.0:
            return frames[0]

        if loop:
            transition = min(self.loop_transition_duration, max(duration * 0.2, 0.05))
            phase = elapsed % (duration + transition)
            if phase > duration:
                return self._blend_frame_data(
                    frames[-1],
                    frames[0],
                    (phase - duration) / max(transition, 1e-6),
                )
            elapsed = phase
        else:
            elapsed = float(np.clip(elapsed, 0.0, duration))

        times = [float(d.get("time", 0.0)) for d in frames]
        idx = bisect.bisect_right(times, elapsed)
        if idx == 0:
            return frames[0]
        if idx >= len(frames):
            return frames[-1]

        f0 = frames[idx - 1]
        f1 = frames[idx]
        t0, t1 = float(f0.get("time", 0.0)), float(f1.get("time", 0.0))
        ratio = (elapsed - t0) / max((t1 - t0), 1e-6)
        return self._blend_frame_data(f0, f1, ratio)

    def _apply_emotion_expression(self, emotion: str, elapsed: float, blend: float) -> Optional[dict[str, float]]:
        data = self.expression_data.get(emotion)
        if not data:
            return None
        frame = self._get_interpolated_motion_frame(
            data["frames"],
            elapsed,
            float(data.get("duration", 0.0)),
            loop=True,
        )
        for key, value in frame.get("weights", {}).items():
            if key in EXPRESSION_LIPSYNC_OVERRIDE_KEYS:
                continue
            expression_value = float(value) * float(np.clip(blend, 0.0, 1.0))
            self.smoothed_weights[key] = max(self.smoothed_weights.get(key, 0.0), expression_value)
        pose = frame.get("pose")
        if not isinstance(pose, dict):
            return None
        return {
            "yaw": float(pose.get("yaw", 0.0)),
            "pitch": float(pose.get("pitch", 0.0)),
            "roll": float(pose.get("roll", 0.0)),
        }

    def _emotion_from_ref_id(self, ref_id: Optional[str]) -> Optional[str]:
        return self._motion_key_from_ref_id(ref_id)

    def _apply_lipsync_weights_from_timeline(self) -> bool:
        with self.state_lock:
            active = self.lipsync_active
            start_time = self.lipsync_start_time
            index = self.lipsync_index
            timeline = self.lipsync_timeline

        if not active or not timeline:
            return False

        elapsed = max(0.0, time.time() - start_time)
        while index + 1 < len(timeline) and timeline[index + 1][0] <= elapsed:
            index += 1
        index = min(index, len(timeline) - 1)
        current = timeline[index][1]

        for key, value in current.items():
            self.smoothed_weights[key] = max(self.smoothed_weights.get(key, 0.0), float(value))

        with self.state_lock:
            self.lipsync_index = index
        return True

    def on_draw(self) -> None:
        self.clear()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        projection = self._make_projection_matrix()
        view = self._make_view_matrix()
        self.avatar.draw(
            self.smoothed_weights,
            view,
            projection,
        )

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == pyglet.window.key.ESCAPE:
            self.close()
            return
        if symbol == pyglet.window.key.F11:
            self.set_fullscreen(not self.fullscreen)
            self._save_window_state()
            return
        if symbol == pyglet.window.key.V:
            self.view_rotation_index = (self.view_rotation_index + 1) % 4
            return
        
        if symbol == pyglet.window.key.T:
            print("[MODE] TRACK is disabled.")
            
        if symbol == pyglet.window.key.R:
            print("[MODE] RECORD is disabled.")
                
        if symbol == pyglet.window.key.P:
            if self.mode == "PLAY":
                self.mode = "IDLE"
                print("[MODE] PLAY stopped.")
            else:
                if self._load_recording():
                    self.mode = "PLAY"
                    self.play_start_time = time.time()
                    self.play_pause_elapsed = 0.0
                    self.play_paused_for_speech = False
                    print("[MODE] PLAY started (Looping).")
                else:
                    print(f"Failed to load {ANIMATION_DATA_PATH.name}. Please record (R key) first.")

    def _save_recording(self) -> None:
        """記録したデータをJSONファイルに保存"""
        if not self.record_data:
            return
        try:
            data_to_save = {"frames": self.record_data}
            ANIMATION_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            ANIMATION_DATA_PATH.write_text(json.dumps(data_to_save, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Error saving recording: {e}")

    def _load_recording(self) -> bool:
        """JSONファイルからアニメーションデータを読み込み"""
        candidate_paths = [ANIMATION_DATA_PATH, LEGACY_ANIMATION_DATA_PATH]
        target_path = next((p for p in candidate_paths if p.exists()), None)
        if target_path is None:
            return False
        try:
            data = json.loads(target_path.read_text(encoding="utf-8"))
            self.play_data = data.get("frames", [])
            # アニメーション全体の長さを計算
            if self.play_data:
                self.animation_duration = self.play_data[-1]["time"]
                self.last_frame_data = {
                    "weights": self.play_data[-1].get("weights", {}),
                    "pose": self.play_data[-1].get("pose", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
                }
            return len(self.play_data) > 0
        except Exception as e:
            print(f"Error loading recording: {e}")
            return False

    def on_close(self) -> None:
        self._save_window_state()
        if self.llm_output_receiver is not None:
            self.llm_output_receiver.stop()
            if self.llm_output_receiver.is_alive():
                self.llm_output_receiver.join(timeout=1.5)
        self.control_server.stop()
        if self.control_server.is_alive():
            self.control_server.join(timeout=1.0)
        self.person_tracker.stop()
        if self.person_tracker.is_alive():
            self.person_tracker.join(timeout=1.5)
        self.speech_queue.put(None)
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.5)
        super().on_close()

    def on_move(self, x: int, y: int) -> None:
        if not self.fullscreen:
            self._save_window_state()

    def on_resize(self, width: int, height: int) -> None:
        if not self.fullscreen:
            self._save_window_state()

    def _save_window_state(self) -> None:
        WINDOW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            x, y = self.get_location()
        except Exception:
            x, y = 0, 0
        state = {
            "x": int(x),
            "y": int(y),
            "width": int(self.width),
            "height": int(self.height),
            "fullscreen": bool(self.fullscreen),
        }
        WINDOW_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")

    @staticmethod
    def _load_window_state() -> dict[str, Any]:
        if not WINDOW_STATE_PATH.exists():
            return {}
        try:
            return json.loads(WINDOW_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _drain_external_commands(self) -> None:
        for _ in range(4):
            try:
                raw = self.control_inbox.get_nowait()
            except queue.Empty:
                return

            command, command_ref_id, command_segments, command_source = self._extract_command_payload(raw)
            if not command:
                continue
            if command.lower() in {"quit", "exit"}:
                pyglet.clock.schedule_once(lambda _dt: self.close(), 0.0)
                continue

            lower = command.lower()
            if lower in {"tts on", "tts start", "tts enable", "tts"}:
                self.direct_tts_mode = True
                print("[MODE] TextToSpeech ON")
                continue
            if lower in {"tts off", "tts stop", "tts disable"}:
                self.direct_tts_mode = False
                print("[MODE] TextToSpeech OFF")
                continue
            if lower.startswith("ref "):
                ref_id = command.split(maxsplit=1)[1].strip()
                self.tts_ref_id = None if ref_id.lower() in {"default", "none", "off"} else ref_id
                print(f"[TTS REF] {self.tts_ref_id or 'server default'}")
                continue

            label = "LLM OUTPUT" if command_source == "llm_output" else "You"
            print(f"[{label}] {command}")
            ref_id = command_ref_id or self.tts_ref_id
            if command_segments:
                segments = [
                    SpeechSegment(segment.text, segment.ref_id or ref_id)
                    for segment in command_segments
                    if segment.text.strip()
                ]
                print(f"[TTS] queued {len(segments)} segments")
                self.speech_queue.put(segments)
            elif self.direct_tts_mode:
                print(f"[TTS] {command}")
                self.speech_queue.put([SpeechSegment(command, ref_id)])
            else:
                reply = self._generate_danya_reply(command)
                print(f"[DANYA] {reply}")
                self.speech_queue.put([SpeechSegment(reply, ref_id)])

    @classmethod
    def _extract_command_payload(cls, raw: str) -> tuple[str, Optional[str], list[SpeechSegment], str]:
        text = raw.strip()
        if not text:
            return "", None, [], ""
        if text.startswith("{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return text, None, [], ""
            ref_id = cls._normalize_ref_id(obj.get("ref_id") or obj.get("ref") or obj.get("emotion"))
            segments = cls._segments_from_payload(obj, ref_id)
            command_text = str(obj.get("text") or obj.get("message") or "").strip()
            if not command_text and segments:
                command_text = " ".join(segment.text for segment in segments)
            source = str(obj.get("source") or "").strip().lower()
            return command_text, ref_id, segments, source
        return text, None, [], ""

    @classmethod
    def _segments_from_payload(cls, obj: dict[str, Any], fallback_ref_id: Optional[str]) -> list[SpeechSegment]:
        raw_segments = None
        for key in ("utterances", "sentences", "segments"):
            if isinstance(obj.get(key), list):
                raw_segments = obj[key]
                break

        if raw_segments is None:
            text = str(obj.get("text") or obj.get("message") or "").strip()
            if not text:
                return []
            ref_id = cls._normalize_ref_id(
                obj.get("ref_id") or obj.get("ref") or obj.get("emotion"),
                fallback_ref_id,
                obj.get("intensity") or obj.get("level"),
            )
            return [SpeechSegment(text, ref_id)]

        segments: list[SpeechSegment] = []
        for item in raw_segments:
            if isinstance(item, str):
                segment_text = item.strip()
                ref_id = fallback_ref_id
            elif isinstance(item, dict):
                segment_text = str(item.get("text") or item.get("message") or "").strip()
                ref_id = cls._normalize_ref_id(
                    item.get("ref_id") or item.get("ref") or item.get("emotion"),
                    fallback_ref_id,
                    item.get("intensity") or item.get("level"),
                )
            else:
                continue
            if segment_text:
                segments.append(SpeechSegment(segment_text, ref_id))
        return segments

    @staticmethod
    def _normalize_ref_id(
        value: Any,
        fallback: Optional[str] = None,
        intensity: Any = None,
    ) -> Optional[str]:
        raw = str(value or "").strip().lower()
        if not raw:
            return fallback
        if raw.endswith(".wav"):
            raw = raw[:-4]
        if raw.endswith("_01"):
            raw = raw[:-3]
        parts = raw.replace("-", "_").split("_")
        emotion = EMOTION_REF_ALIASES.get(parts[0], parts[0])
        if emotion not in EMOTION_REF_PREFIXES:
            return raw
        explicit_intensity = next((part for part in parts[1:] if part in {"high", "normal", "mid", "low"}), None)
        level = str(intensity or explicit_intensity or DEFAULT_EMOTION_INTENSITY).strip().lower()
        level = EMOTION_LEVEL_ALIASES.get(level, level)
        if level not in {"high", "normal"}:
            level = DEFAULT_EMOTION_INTENSITY
        return f"{emotion}_{level}"

    def _build_lipsync_timeline(self, wav_path: Path) -> list[tuple[float, dict[str, float]]]:
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception:
            return []

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size < 128:
            return []

        frame_len = max(int(sr * LIPSYNC_FRAME_SEC), 256)
        hop_len = max(int(sr * LIPSYNC_HOP_SEC), 80)
        if audio.size <= frame_len:
            return []

        win = np.hanning(frame_len).astype(np.float32)
        amp_ref = float(np.percentile(np.abs(audio), 95)) + 1e-6
        timeline: list[tuple[float, dict[str, float]]] = []

        for start in range(0, audio.size - frame_len, hop_len):
            frame = audio[start : start + frame_len] * win
            rms = float(np.sqrt(np.mean(np.square(frame))) + 1e-9)
            energy = float(np.clip(rms / amp_ref, 0.0, 1.0))

            if energy < 0.03:
                weights = {
                    "jawopen": 0.0,
                    "mouthopen": 0.0,
                    "mouthsmileleft": 0.0,
                    "mouthsmileright": 0.0,
                    "mouthpucker": 0.0,
                }
                timeline.append((start / sr, weights))
                continue

            spec = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
            band = (freqs >= 200.0) & (freqs <= 3000.0)
            mag = spec[band]
            f = freqs[band]
            if mag.size < 8:
                continue

            mag_smooth = np.convolve(mag, np.ones(5, dtype=np.float32) / 5.0, mode="same")
            top_idx = np.argpartition(mag_smooth, -6)[-6:]
            ranked = top_idx[np.argsort(mag_smooth[top_idx])[::-1]]
            cand = [float(f[i]) for i in ranked]
            if not cand:
                continue

            f1 = cand[0]
            f2 = cand[0]
            for val in cand[1:]:
                if abs(val - f1) >= 120.0:
                    f2 = val
                    break
            if f2 < f1:
                f1, f2 = f2, f1

            probs = self._estimate_vowel_probabilities(f1, f2)
            weights = self._vowel_probs_to_weights(probs, energy)
            timeline.append((start / sr, weights))

        return timeline

    @staticmethod
    def _estimate_vowel_probabilities(f1: float, f2: float) -> dict[str, float]:
        prototypes = {
            "a": (800.0, 1200.0),
            "i": (320.0, 2300.0),
            "u": (350.0, 1300.0),
            "e": (500.0, 2000.0),
            "o": (520.0, 900.0),
        }
        sigma_f1 = 220.0
        sigma_f2 = 420.0
        scores: dict[str, float] = {}
        for key, (pf1, pf2) in prototypes.items():
            d1 = (f1 - pf1) / sigma_f1
            d2 = (f2 - pf2) / sigma_f2
            scores[key] = math.exp(-0.5 * (d1 * d1 + d2 * d2))

        total = sum(scores.values()) + 1e-9
        return {k: v / total for k, v in scores.items()}

    @staticmethod
    def _vowel_probs_to_weights(probs: dict[str, float], energy: float) -> dict[str, float]:
        a = probs.get("a", 0.0)
        i = probs.get("i", 0.0)
        u = probs.get("u", 0.0)
        e = probs.get("e", 0.0)
        o = probs.get("o", 0.0)

        jaw = energy * (0.18 + 0.78 * a + 0.32 * o + 0.28 * e)
        mouth_open = energy * (0.12 + 0.84 * a + 0.42 * o + 0.34 * e)
        smile = energy * (0.60 * i + 0.38 * e)
        pucker = energy * (0.76 * u + 0.56 * o)

        return {
            "jawopen": float(np.clip(jaw, 0.0, 1.0)),
            "mouthopen": float(np.clip(mouth_open, 0.0, 1.0)),
            "mouthsmileleft": float(np.clip(smile, 0.0, 1.0)),
            "mouthsmileright": float(np.clip(smile, 0.0, 1.0)),
            "mouthpucker": float(np.clip(pucker, 0.0, 1.0)),
        }

    def _speech_worker(self) -> None:
        while True:
            segments = self.speech_queue.get()
            if segments is None:
                return
            segments = [segment for segment in segments if segment.text.strip()]
            if not segments:
                continue

            batch_id = int(time.time() * 1000)
            TTS_SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
            max_workers = min(3, max(1, len(segments)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for index, segment in enumerate(segments):
                    out_path = TTS_SEGMENT_DIR / f"tts_{batch_id}_{index:02d}.wav"
                    futures.append(executor.submit(self._synthesize_speech_segment, segment, out_path))

                for future in futures:
                    ok, reason, segment, out_path = future.result()
                    if not ok:
                        print(f"[TTS ERROR] {reason[:110]}")
                        continue
                    self._play_speech_segment(segment, out_path)

            try:
                shutil.copyfile(str(TTS_SEGMENT_DIR / f"tts_{batch_id}_00.wav"), str(self.tts_output))
            except Exception:
                pass

    def _synthesize_speech_segment(
        self,
        segment: SpeechSegment,
        out_path: Path,
    ) -> tuple[bool, str, SpeechSegment, Path]:
        ok, reason = synthesize_audio(
            server_url=self.tts_server,
            text=segment.text,
            out_path=out_path,
            ref_id=segment.ref_id,
        )
        return ok, reason, segment, out_path

    def _play_speech_segment(self, segment: SpeechSegment, out_path: Path) -> None:
        timeline = self._build_lipsync_timeline(out_path)
        speech_emotion = self._emotion_from_ref_id(segment.ref_id)

        with self.state_lock:
            self.is_speaking = True
            self.speech_emotion = speech_emotion
            self.speech_emotion_started_at = time.time()
            self.lipsync_timeline = timeline
            self.lipsync_index = 0
            self.lipsync_start_time = self.speech_emotion_started_at
            self.lipsync_active = len(timeline) > 0

        ok_play, reason_play = play_audio(out_path, audio_device=self.audio_device)

        with self.state_lock:
            self.is_speaking = False
            self.lipsync_active = False
            self.lipsync_index = 0
            self.lipsync_timeline = []
        if not ok_play:
            print(f"[AUDIO ERROR] {reason_play[:110]}")

    @staticmethod
    def _generate_danya_reply(user_text: str) -> str:
        text = user_text.strip()
        lower = text.lower()
        if not text:
            return "もう一度、聞かせてください。"
        if "こんにちは" in text or "hello" in lower or "hi" in lower:
            return "こんにちは。DANYAです。今日はどんなことを話しますか？"
        if "名前" in text:
            return "私はDANYAです。あなたと自然に会話できるよう練習中です。"
        if "ありがとう" in text or "thanks" in lower:
            return "どういたしまして。続けて話してみましょう。"
        if "元気" in text:
            return "元気です。あなたは今日どんな気分ですか？"
        starts = ["なるほど", "いいですね", "わかりました", "面白いですね"]
        follow = [
            "もう少し詳しく教えてください。",
            "そのとき、あなたはどう感じましたか？",
            "次にやりたいことは何ですか？",
            "一緒に整理してみましょう。",
        ]
        return f"{random.choice(starts)}。{random.choice(follow)}"

    def _make_head_neck_pose_matrix(self) -> np.ndarray:
        speak = float(self.speech_motion)

        mouth_open = float(np.clip(self.smoothed_weights.get("mouthopen", 0.0), 0.0, 1.0))
        speaking_nod = mouth_open * 0.06 * speak

        yaw = self.head_yaw
        pitch = self.head_pitch + speaking_nod
        roll = self.head_roll

        # Keep pose behavior consistent with mediapipe_face_avatar.py.
        yaw_m = self._rotation_x_matrix(math.radians(-yaw * 0.6))
        pitch_m = self._rotation_y_matrix(math.radians(pitch * 0.6))
        roll_m = self._rotation_z_matrix(math.radians(roll * 0.4))
        head_rot = pitch_m @ yaw_m @ roll_m

        pivot = np.array([0.0, -0.06, 0.0], dtype=np.float32)
        return (
            self._translation_matrix(pivot)
            @ head_rot
            @ self._translation_matrix(-pivot)
        )

    @staticmethod
    def _translation_matrix(offset: np.ndarray) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = offset[:3]
        return m

    @staticmethod
    def _rotation_x_matrix(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, -s, 0.0],
                [0.0, s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _rotation_y_matrix(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

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
    def _decay_weights(current: dict[str, float]) -> dict[str, float]:
        next_weights: dict[str, float] = {}
        for key, value in current.items():
            value *= 0.93
            if value > 0.01:
                next_weights[key] = value
        return next_weights

    def _make_projection_matrix(self) -> np.ndarray:
        aspect = max(self.width, 1) / max(self.height, 1)
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

    def _make_view_matrix(self) -> np.ndarray:
        base_eye = np.array([0.0, 0.10, 3.45], dtype=np.float32)
        target = np.array([0.0, 0.05, 0.0], dtype=np.float32)
        base_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        orbit_rad = math.radians(90.0 * float(self.view_rotation_index))
        rotation = self._rotation_z_matrix(orbit_rad)

        eye_offset = np.array(
            [
                base_eye[0] - target[0],
                base_eye[1] - target[1],
                base_eye[2] - target[2],
                1.0,
            ],
            dtype=np.float32,
        )

        # Portrait rotations need a little more distance so the full head stays in frame.
        if self.view_rotation_index % 2 == 1:
            eye_offset[:3] *= 1.22

        rotated_offset = (rotation @ eye_offset)[:3]
        rotated_up = (rotation @ np.array([base_up[0], base_up[1], base_up[2], 0.0], dtype=np.float32))[:3]
        eye = target + rotated_offset
        up = rotated_up / max(np.linalg.norm(rotated_up), 1e-6)
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        side = np.cross(forward, up)
        side = side / np.linalg.norm(side)
        up = np.cross(side, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = side
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ eye
        return view

def send_to_control_server(text: str, host: str = CONTROL_HOST, port: int = CONTROL_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2.0) as sock:
            sock.sendall((text.strip() + "\n").encode("utf-8"))
        return True
    except OSError:
        return False

def run_control_terminal_loop(host: str = CONTROL_HOST, port: int = CONTROL_PORT) -> None:
    print("DANYA control terminal")
    print(f"target: {host}:{port}")
    print("Type message and press Enter. 'tts on' to speak typed text as-is, 'tts off' to return, 'exit' to quit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            send_to_control_server("quit", host=host, port=port)
            return
        if not send_to_control_server(text, host=host, port=port):
            print("Failed to send: avatar app is not running yet.")

def launch_control_terminal() -> None:
    script = shlex.quote(str(Path(__file__).resolve()))
    python_exec = shlex.quote(sys.executable)
    command = f"{python_exec} {script} --control"
    shell_cmd = f"{command}; echo; echo '[Control terminal closed]'; exec bash"

    candidates = [
        ("x-terminal-emulator", ["x-terminal-emulator", "-e", "bash", "-lc", shell_cmd]),
        ("gnome-terminal", ["gnome-terminal", "--", "bash", "-lc", shell_cmd]),
        ("konsole", ["konsole", "-e", "bash", "-lc", shell_cmd]),
        ("xfce4-terminal", ["xfce4-terminal", "--command", f"bash -lc \"{shell_cmd}\""]),
        ("xterm", ["xterm", "-e", "bash", "-lc", shell_cmd]),
    ]

    for binary, argv in candidates:
        if shutil.which(binary):
            try:
                subprocess.Popen(argv)
                return
            except Exception:
                continue

    print("Could not auto-launch terminal. Run this manually:")
    print(f"bash -lc \"{shell_cmd}\"")

def main() -> None:
    global YOLO_TRACKING_ENABLED, YOLO_CAMERA_INDEX, YOLO_MODEL_PATH, YOLO_PREVIEW_ENABLED

    parser = argparse.ArgumentParser(description="DANYA avatar viewer and control endpoint")
    parser.add_argument("--control", action="store_true", help="Run control terminal loop")
    parser.add_argument(
        "--no-control-terminal",
        action="store_true",
        help="Do not auto-open control terminal window",
    )
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO person tracking")
    parser.add_argument("--no-yolo-preview", action="store_true", help="Disable YOLO preview window")
    parser.add_argument("--yolo-camera", type=int, default=YOLO_CAMERA_INDEX, help="Camera index for YOLO tracking")
    parser.add_argument("--yolo-model", default=YOLO_MODEL_PATH, help="Ultralytics YOLO model path/name")
    parser.add_argument(
        "--llm-output-server",
        default=DEFAULT_LLM_OUTPUT_SERVER,
        help="LLM output server URL for /api/output polling",
    )
    parser.add_argument(
        "--llm-output-interval",
        type=float,
        default=DEFAULT_LLM_OUTPUT_INTERVAL,
        help="Seconds between LLM output polls",
    )
    parser.add_argument("--no-llm-output", action="store_true", help="Disable LLM output polling")
    args = parser.parse_args()

    if args.control:
        run_control_terminal_loop()
        return

    YOLO_TRACKING_ENABLED = YOLO_TRACKING_ENABLED and not args.no_yolo
    YOLO_PREVIEW_ENABLED = YOLO_PREVIEW_ENABLED and not args.no_yolo_preview
    YOLO_CAMERA_INDEX = args.yolo_camera
    YOLO_MODEL_PATH = args.yolo_model

    llm_output_server = None if args.no_llm_output else args.llm_output_server
    AvatarApp(
        launch_terminal=not args.no_control_terminal,
        llm_output_server=llm_output_server,
        llm_output_interval=args.llm_output_interval,
    )
    pyglet.app.run()

if __name__ == "__main__":
    main()
