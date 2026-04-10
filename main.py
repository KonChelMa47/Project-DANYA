from __future__ import annotations

import base64
import io
import json
import math
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
    glActiveTexture,
    glBlendFunc,
    glBindTexture,
    glClearColor,
    glEnable,
    glTexParameteri,
)
from pyglet.graphics.shader import Shader, ShaderProgram


WINDOW_TITLE = "DANYA Avatar"
DISPLAY_ROTATION_DEG = 90.0
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "assets" / "models" / "avatar.glb"
MODEL_CACHE = BASE_DIR / ".cache" / "mediapipe_face_landmarker.task"
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
    "mouthsmileleft": ["mouthsmile"],
    "mouthsmileright": ["mouthsmile"],
    "eyesclosed": ["eyeblinkleft", "eyeblinkright"],
    "eyeblinkleft": ["eyesclosed"],
    "eyeblinkright": ["eyesclosed"],
    "browinnerup": ["browraise"],
    "browraise": ["browinnerup"],
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
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = face_landmarker.FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def infer(self, frame_bgr: np.ndarray) -> dict[str, float]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.timestamp_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        weights: dict[str, float] = {}

        if result.face_blendshapes:
            for item in result.face_blendshapes[0]:
                if item.category_name:
                    weights[item.category_name.lower()] = float(item.score or 0.0)

        self._apply_aliases(weights)
        return self._shape_weights(weights)

    @staticmethod
    def _apply_aliases(weights: dict[str, float]) -> None:
        for source, targets in ALIAS_WEIGHTS.items():
            if source in weights:
                for target in targets:
                    weights[target] = max(weights.get(target, 0.0), weights[source])

    @staticmethod
    def _shape_weights(weights: dict[str, float]) -> dict[str, float]:
        shaped: dict[str, float] = {}
        for key, value in weights.items():
            if value <= 0.01:
                continue
            shaped[key] = float(np.clip((value - 0.03) / 0.77, 0.0, 1.0))
        return shaped


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
            uniform mat3 screen_warp;
            out vec3 v_normal;
            out vec2 v_texcoord;
            void main() {
                vec4 world_pos = model * vec4(position, 1.0);
                v_normal = mat3(transpose(inverse(model))) * normal;
                v_texcoord = vec2(texcoord.x, 1.0 - texcoord.y);
                vec4 clip = projection * view * world_pos;
                vec3 ndc = clip.xyz / clip.w;
                vec3 warped = screen_warp * vec3(ndc.xy, 1.0);
                gl_Position = vec4(warped.xy * clip.w, clip.z, clip.w);
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
        self.program["screen_warp"] = tuple(np.eye(3, dtype=np.float32).T.reshape(-1))

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

    def draw(self, weights: dict[str, float], view: np.ndarray, projection: np.ndarray, screen_warp: np.ndarray) -> None:
        self.program.use()
        self.program["model"] = tuple(self.model_matrix.T.reshape(-1))
        self.program["view"] = tuple(view.T.reshape(-1))
        self.program["projection"] = tuple(projection.T.reshape(-1))
        self.program["light_dir"] = (0.3, 0.7, 0.8)
        self.program["screen_warp"] = tuple(np.array(screen_warp, dtype=np.float32).T.reshape(-1))
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
        super().__init__(caption=WINDOW_TITLE, fullscreen=True, config=config, vsync=True)
        self.set_mouse_visible(True)
        self.camera = CameraStream()
        self.tracker = self._create_tracker()
        self.avatar = GLBAvatar(MODEL_PATH)
        self.smoothed_weights: dict[str, float] = {}
        self.warp_points = self._default_warp_points()
        self.active_corner: Optional[int] = None
        self.show_handles = True
        self.corner_pick_radius = 64.0
        self.handle_batch = pyglet.graphics.Batch()
        self.handle_points = [
            shapes.Circle(0, 0, 8, color=(255, 255, 255), batch=self.handle_batch),
            shapes.Circle(0, 0, 8, color=(255, 255, 255), batch=self.handle_batch),
            shapes.Circle(0, 0, 8, color=(255, 255, 255), batch=self.handle_batch),
            shapes.Circle(0, 0, 8, color=(255, 255, 255), batch=self.handle_batch),
        ]
        self.handle_lines = [
            shapes.Line(0, 0, 0, 0, thickness=2, color=(140, 140, 140), batch=self.handle_batch),
            shapes.Line(0, 0, 0, 0, thickness=2, color=(140, 140, 140), batch=self.handle_batch),
            shapes.Line(0, 0, 0, 0, thickness=2, color=(140, 140, 140), batch=self.handle_batch),
            shapes.Line(0, 0, 0, 0, thickness=2, color=(140, 140, 140), batch=self.handle_batch),
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
        frame = self.camera.latest_frame()
        if frame is None:
            self.smoothed_weights = self._decay_weights(self.smoothed_weights)
            return

        try:
            new_weights = self.tracker.infer(frame)
        except Exception:
            new_weights = {}
        self.smoothed_weights = self._smooth_to_target(self.smoothed_weights, new_weights)

    def on_draw(self) -> None:
        self.clear()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        projection = self._make_projection_matrix()
        view = self._make_view_matrix()
        self.avatar.draw(self.smoothed_weights, view, projection, self._make_screen_warp_matrix())
        if self.show_handles:
            self._update_warp_handle_visuals()
            self.handle_batch.draw()

    def on_key_press(self, symbol: int, modifiers: int) -> None:  # noqa: ARG002
        if symbol == pyglet.window.key.ESCAPE:
            self.close()
        if symbol == pyglet.window.key.R:
            self.warp_points = self._default_warp_points()
        if symbol == pyglet.window.key.E:
            self.show_handles = not self.show_handles

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:  # noqa: ARG002
        if button != pyglet.window.mouse.LEFT:
            return
        self.active_corner = self._pick_corner(x, y)

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:  # noqa: ARG002
        if self.active_corner is None:
            return
        self.warp_points[self.active_corner] = [
            float(np.clip(x, 0, max(self.width - 1, 0))),
            float(np.clip(y, 0, max(self.height - 1, 0))),
        ]

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:  # noqa: ARG002
        if button == pyglet.window.mouse.LEFT:
            self.active_corner = None

    def on_close(self) -> None:
        self.tracker.close()
        self.camera.close()
        super().on_close()

    def _default_warp_points(self) -> list[list[float]]:
        face_w = float(self.width) * 0.46
        face_h = float(self.height) * 0.72
        left = (float(self.width) - face_w) * 0.5
        top = (float(self.height) - face_h) * 0.5
        return [
            [left, top],
            [left + face_w, top],
            [left + face_w, top + face_h],
            [left, top + face_h],
        ]

    def _pick_corner(self, x: int, y: int) -> Optional[int]:
        points = np.array(self.warp_points, dtype=np.float32)
        cursor = np.array([x, y], dtype=np.float32)
        distances = np.linalg.norm(points - cursor, axis=1)
        index = int(np.argmin(distances))
        if distances[index] <= self.corner_pick_radius:
            return index
        return None

    def _make_screen_warp_matrix(self) -> np.ndarray:
        src = np.array(self._default_warp_points(), dtype=np.float32)
        dst = np.array(self.warp_points, dtype=np.float32)
        src_ndc = self._window_to_ndc_points(src)
        dst_ndc = self._window_to_ndc_points(dst)
        warp = cv2.getPerspectiveTransform(src_ndc, dst_ndc)
        return warp.astype(np.float32)

    def _window_to_ndc_points(self, points: np.ndarray) -> np.ndarray:
        w = max(float(self.width), 1.0)
        h = max(float(self.height), 1.0)
        ndc = np.empty((points.shape[0], 2), dtype=np.float32)
        ndc[:, 0] = (points[:, 0] / w) * 2.0 - 1.0
        ndc[:, 1] = (points[:, 1] / h) * 2.0 - 1.0
        return ndc

    def _update_warp_handle_visuals(self) -> None:
        points = self.warp_points
        for index, (x, y) in enumerate(points):
            point = self.handle_points[index]
            point.x = float(x)
            point.y = float(y)
            point.color = (0, 255, 255) if index == self.active_corner else (255, 255, 255)
            point.opacity = 220

        for index in range(4):
            line = self.handle_lines[index]
            x1, y1 = points[index]
            x2, y2 = points[(index + 1) % 4]
            line.x = float(x1)
            line.y = float(y1)
            line.x2 = float(x2)
            line.y2 = float(y2)
            line.color = (0, 200, 255) if self.active_corner == index else (140, 140, 140)

    @staticmethod
    def _smooth_to_target(current: dict[str, float], target: dict[str, float]) -> dict[str, float]:
        keys = set(current) | set(target)
        next_weights: dict[str, float] = {}
        for key in keys:
            cur = current.get(key, 0.0)
            tar = target.get(key, 0.0)
            value = cur + (tar - cur) * 0.35
            if value > 0.01:
                next_weights[key] = float(np.clip(value, 0.0, 1.0))
        return next_weights

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

    @staticmethod
    def _make_view_matrix() -> np.ndarray:
        eye = np.array([0.0, 0.10, 3.45], dtype=np.float32)
        target = np.array([0.0, 0.05, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
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


def main() -> None:
    AvatarApp()
    pyglet.app.run()


if __name__ == "__main__":
    main()
