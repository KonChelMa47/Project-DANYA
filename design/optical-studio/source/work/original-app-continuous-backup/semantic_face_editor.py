from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np
import pyglet
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from pyglet import shapes
from pyglet.gl import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_COMPONENT24,
    GL_DEPTH_TEST,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    glActiveTexture,
    glBindTexture,
    glBlendFunc,
    glClear,
    glDisable,
    glEnable,
    glViewport,
)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.image.buffer import Framebuffer, Renderbuffer

from face_editor_ui import EditorUiState, FaceEditorPanel


FEATURE_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
)

EYE_NAMES = ("left_eye", "right_eye")
SELECTABLE_FEATURES = ("left_eye", "nose", "mouth")

# Applied back-to-front. The final visible stack is eyes, nose, mouth.
COMPOSITE_ORDER = ("mouth", "nose", "left_eye", "right_eye")
MAX_CUSTOM_PARTS = 4

# MediaPipe Face Landmarker indices ordered around each visible feature contour.
FEATURE_POLYGONS = {
    "left_eye": (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7),
    "right_eye": (362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382),
    "nose": (4, 275, 440, 344, 278, 294, 327, 326, 2, 97, 98, 64, 48, 115, 220, 45),
    "mouth": (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146),
}

FEATURE_PADDING = {
    "left_eye": 0.02,
    "right_eye": 0.02,
    "nose": 0.03,
    "mouth": 0.03,
}

SYMMETRIC_FEATURE_PAIRS = (
    ("left_eye", "right_eye"),
)

FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
)


@dataclass
class FeatureGeometry:
    center: tuple[float, float]
    size: tuple[float, float]
    outline: tuple[tuple[float, float], ...]


@dataclass
class FeatureTransform:
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale: float = 1.0
    rotation_deg: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "scale": self.scale,
            "rotation_deg": self.rotation_deg,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "FeatureTransform":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            offset_x=float(np.clip(float(raw.get("offset_x", 0.0)), -0.5, 0.5)),
            offset_y=float(np.clip(float(raw.get("offset_y", 0.0)), -0.5, 0.5)),
            scale=float(np.clip(float(raw.get("scale", 1.0)), 0.45, 2.2)),
            rotation_deg=float(np.clip(float(raw.get("rotation_deg", 0.0)), -45.0, 45.0)),
        )


@dataclass
class CustomPart:
    name: str
    polygon: tuple[tuple[float, float], ...]

    def as_dict(self, transform: FeatureTransform) -> dict[str, object]:
        return {
            "name": self.name,
            "polygon": [[x, y] for x, y in self.polygon],
            "transform": transform.as_dict(),
        }


@dataclass
class FaceAnalysis:
    base_rgba: np.ndarray
    mask_a_rgba: np.ndarray
    mask_b_rgba: np.ndarray
    geometry: dict[str, FeatureGeometry]
    rotation_quadrants: int


class FacePartAnalyzer:
    def __init__(self, model_path: Path) -> None:
        options = face_landmarker.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
        )
        self.landmarker = face_landmarker.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:
            pass

    def analyze(self, rgba_top_down: np.ndarray) -> FaceAnalysis:
        rgb = np.ascontiguousarray(rgba_top_down[:, :, :3])
        landmarks: Optional[np.ndarray] = None
        detected_rotation = 0

        # The projection is often physically rotated. Trying all right-angle
        # orientations is cheap because this runs only during calibration.
        for quadrants in (0, 1, 3, 2):
            candidate = np.ascontiguousarray(np.rot90(rgb, quadrants))
            result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=candidate))
            if not result.face_landmarks:
                continue
            h_rot, w_rot = candidate.shape[:2]
            points_rot = np.array(
                [(lm.x * (w_rot - 1), lm.y * (h_rot - 1)) for lm in result.face_landmarks[0]],
                dtype=np.float32,
            )
            landmarks = self._unrotate_points(points_rot, quadrants, rgb.shape[1], rgb.shape[0])
            detected_rotation = quadrants
            break

        if landmarks is None or landmarks.shape[0] < 468:
            raise RuntimeError("Face Landmarker could not identify the rendered avatar")

        height, width = rgb.shape[:2]
        eye_centers = {
            name: landmarks[np.array(FEATURE_POLYGONS[name], dtype=np.int32)].mean(axis=0)
            for name in ("left_eye", "right_eye")
        }
        masks: dict[str, np.ndarray] = {}
        for name in FEATURE_NAMES:
            polygon = landmarks[np.array(FEATURE_POLYGONS[name], dtype=np.int32)]
            masks[name] = self._soft_polygon_mask(polygon, width, height, FEATURE_PADDING[name])

        reflection = self._face_reflection_matrix(
            eye_centers["left_eye"],
            eye_centers["right_eye"],
            landmarks[10],
            landmarks[152],
        )
        for left_name, right_name in SYMMETRIC_FEATURE_PAIRS:
            left_mask, right_mask = self._make_symmetric_pair(
                masks[left_name],
                masks[right_name],
                reflection,
                width,
                height,
                use_union=True,
            )
            masks[left_name] = left_mask
            masks[right_name] = right_mask

        geometry = {
            name: self._geometry_from_mask(masks[name], width, height)
            for name in FEATURE_NAMES
        }

        face_width = float(np.ptp(landmarks[:, 0]))
        feature_union = np.maximum.reduce([masks[name] for name in FEATURE_NAMES])
        base_rgb = self._make_skin_base(rgb, landmarks, feature_union, face_width)
        base_rgba = np.dstack((base_rgb, rgba_top_down[:, :, 3])).astype(np.uint8)

        packed_a = np.zeros((height, width, 4), dtype=np.uint8)
        packed_b = np.zeros((height, width, 4), dtype=np.uint8)
        for channel, name in enumerate(FEATURE_NAMES):
            packed_a[:, :, channel] = np.round(masks[name] * 255.0).astype(np.uint8)

        return FaceAnalysis(base_rgba, packed_a, packed_b, geometry, detected_rotation)

    @staticmethod
    def _make_skin_base(
        rgb: np.ndarray,
        landmarks: np.ndarray,
        feature_union: np.ndarray,
        face_width: float,
    ) -> np.ndarray:
        height, width = rgb.shape[:2]
        face_hard = np.zeros((height, width), dtype=np.uint8)
        oval = np.round(landmarks[np.array(FACE_OVAL, dtype=np.int32)]).astype(np.int32)
        cv2.fillPoly(face_hard, [oval], 255, lineType=cv2.LINE_AA)

        feature_hard = np.where(feature_union > 0.02, 255, 0).astype(np.uint8)
        detail_guard = max(12, int(face_width * 0.040))
        guard_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (detail_guard * 2 + 1, detail_guard * 2 + 1),
        )
        feature_guard = cv2.dilate(feature_hard, guard_kernel)
        valid_skin = (face_hard > 220) & (feature_guard == 0)
        pixels = rgb[valid_skin].astype(np.float32)
        if pixels.shape[0] < 100:
            valid_skin = face_hard > 220
            pixels = rgb[face_hard > 220].astype(np.float32)
        median = np.median(pixels, axis=0)
        distances = np.linalg.norm(pixels - median, axis=1)
        keep = distances <= np.percentile(distances, 60.0)
        skin_color = np.median(pixels[keep], axis=0) if np.any(keep) else median

        luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        clean_luminance = luminance.copy()
        clean_luminance[feature_guard > 0] = float(np.median(luminance[valid_skin]))
        shading_sigma = max(10.0, face_width * 0.055)
        shading = cv2.GaussianBlur(clean_luminance, (0, 0), shading_sigma)
        reference = max(float(np.median(shading[valid_skin])), 1.0)
        shading_ratio = np.clip(shading / reference, 0.86, 1.14)[:, :, None]

        skin_field = np.clip(skin_color[None, None, :] * shading_ratio, 0.0, 255.0)
        face_alpha = cv2.GaussianBlur(
            face_hard.astype(np.float32) / 255.0,
            (0, 0),
            2.0,
        )[:, :, None]
        base = rgb.astype(np.float32) * (1.0 - face_alpha) + skin_field * face_alpha
        return np.clip(base, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _unrotate_points(points: np.ndarray, quadrants: int, width: int, height: int) -> np.ndarray:
        x = points[:, 0]
        y = points[:, 1]
        if quadrants == 0:
            return points
        if quadrants == 1:
            return np.column_stack((width - 1 - y, x)).astype(np.float32)
        if quadrants == 2:
            return np.column_stack((width - 1 - x, height - 1 - y)).astype(np.float32)
        return np.column_stack((y, height - 1 - x)).astype(np.float32)

    @staticmethod
    def _soft_polygon_mask(
        polygon: np.ndarray,
        width: int,
        height: int,
        padding_ratio: float,
    ) -> np.ndarray:
        contour = np.round(polygon).astype(np.int32)
        x, y, w, h = cv2.boundingRect(contour)
        pad_x = max(0, int(round(w * padding_ratio)))
        pad_y = max(0, int(round(h * padding_ratio)))
        center = polygon.mean(axis=0)
        expanded = center + (polygon - center) * np.array(
            [1.0 + 2.0 * pad_x / max(w, 1), 1.0 + 2.0 * pad_y / max(h, 1)],
            dtype=np.float32,
        )
        expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
        expanded_contour = np.round(expanded).astype(np.int32)
        hard = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(hard, [expanded_contour], 255, lineType=cv2.LINE_AA)
        return cv2.GaussianBlur(hard, (0, 0), 1.0).astype(np.float32) / 255.0

    @staticmethod
    def _face_reflection_matrix(
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        forehead: np.ndarray,
        chin: np.ndarray,
    ) -> np.ndarray:
        midpoint = (left_eye + right_eye) * 0.5
        face_axis = chin - forehead
        length = float(np.linalg.norm(face_axis))
        if length < 1e-6:
            raise RuntimeError("Face Landmarker returned an invalid facial midline")
        symmetry_axis = face_axis.astype(np.float32) / length
        linear = 2.0 * np.outer(symmetry_axis, symmetry_axis) - np.eye(2, dtype=np.float32)
        translation = midpoint - linear @ midpoint
        return np.column_stack((linear, translation)).astype(np.float32)

    @staticmethod
    def _reflect_mask(
        mask: np.ndarray,
        reflection: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        return cv2.warpAffine(
            mask,
            reflection,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    @classmethod
    def _make_symmetric_pair(
        cls,
        left: np.ndarray,
        right: np.ndarray,
        reflection: np.ndarray,
        width: int,
        height: int,
        use_union: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        right_on_left = cls._reflect_mask(right, reflection, width, height)
        # A complete shared eye contour is mirrored to both sides.
        consensus = (
            np.maximum(left, right_on_left)
            if use_union
            else np.minimum(left, right_on_left)
        )
        hard = np.where(consensus >= 0.20, 255, 0).astype(np.uint8)
        hard = cv2.morphologyEx(
            hard,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        if cv2.countNonZero(hard) < 12:
            hard = np.where((left + right_on_left) * 0.5 >= 0.50, 255, 0).astype(np.uint8)
        canonical = cv2.GaussianBlur(hard, (0, 0), 1.0).astype(np.float32) / 255.0
        mirrored = cls._reflect_mask(canonical, reflection, width, height)
        return canonical, mirrored

    @staticmethod
    def _geometry_from_mask(mask: np.ndarray, width: int, height: int) -> FeatureGeometry:
        hard = np.where(mask >= 0.35, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("An editable face-part mask was empty")
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        outline = cv2.approxPolyDP(contour, max(0.75, perimeter * 0.006), True).reshape(-1, 2)
        if len(outline) > 64:
            indices = np.linspace(0, len(outline) - 1, 64, dtype=np.int32)
            outline = outline[indices]
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-6:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
        else:
            center_x, center_y = x + w * 0.5, y + h * 0.5
        return FeatureGeometry(
            center=(float(center_x / width), float(1.0 - center_y / height)),
            size=(float(w / width), float(h / height)),
            outline=tuple((float(px / width), float(1.0 - py / height)) for px, py in outline),
        )


class SemanticFaceEditor:
    def __init__(self, model_path: Path, state_path: Path, landmarker_model_path: Path) -> None:
        self.model_path = model_path
        self.state_path = state_path
        self.trim_mask_path = state_path.with_name("semantic_brow_trim.png")
        self.landmarker_model_path = landmarker_model_path
        self.transforms = {name: FeatureTransform() for name in FEATURE_NAMES}
        self.custom_parts: list[CustomPart] = []
        self._load_state()

        self.width = 0
        self.height = 0
        self.source_texture: Optional[pyglet.image.Texture] = None
        self.framebuffer: Optional[Framebuffer] = None
        self.depth_buffer: Optional[Renderbuffer] = None
        self.base_texture: Optional[pyglet.image.Texture] = None
        self.mask_a_texture: Optional[pyglet.image.Texture] = None
        self.mask_b_texture: Optional[pyglet.image.Texture] = None
        self.custom_mask_texture: Optional[pyglet.image.Texture] = None
        self.geometry: dict[str, FeatureGeometry] = {}
        self.ready = False
        self.analysis_pending = True
        self.analysis_error: Optional[str] = None
        self.edit_visible = True
        self.selected: Optional[str] = None
        self.drag_start: Optional[tuple[float, float]] = None
        self.drag_origin: Optional[tuple[float, float]] = None
        self.rotation_drag_start_x: Optional[float] = None
        self.rotation_drag_origin: Optional[float] = None
        self.trim_mode = False
        self.trim_mask: Optional[np.ndarray] = None
        self.trim_brush_radius = 18.0
        self.trim_last_point: Optional[tuple[float, float]] = None
        self.trim_cursor_position: Optional[tuple[float, float]] = None
        self.tool_mode = "select"
        self.gui_visible = True
        self.cut_points: list[tuple[float, float]] = []
        self.undo_stack: list[
            tuple[
                dict[str, FeatureTransform],
                Optional[np.ndarray],
                list[CustomPart],
            ]
        ] = []

        self.program = self._make_program()
        self.quad = self.program.vertex_list(
            6,
            pyglet.gl.GL_TRIANGLES,
            position=("f", (-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1)),
            texcoord=("f", (0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1)),
        )
        self.overlay_batch = pyglet.graphics.Batch()
        self.overlay_lines = [
            shapes.Line(0, 0, 0, 0, thickness=2, color=(30, 255, 170), batch=self.overlay_batch)
            for _ in range(128)
        ]
        self.overlay_centers = [
            shapes.Circle(0, 0, 5, color=(255, 255, 255), batch=self.overlay_batch)
            for _ in range(2)
        ]
        self.trim_overlay_batch = pyglet.graphics.Batch()
        self.trim_cursor = shapes.Circle(
            0,
            0,
            self.trim_brush_radius,
            color=(255, 220, 40),
            batch=self.trim_overlay_batch,
        )
        self.trim_cursor.opacity = 70
        self.cut_overlay_batch = pyglet.graphics.Batch()
        self.cut_overlay_lines = [
            shapes.Line(0, 0, 0, 0, thickness=2, color=(255, 210, 55), batch=self.cut_overlay_batch)
            for _ in range(64)
        ]
        self.cut_overlay_points = [
            shapes.Circle(0, 0, 4, color=(255, 240, 150), batch=self.cut_overlay_batch)
            for _ in range(64)
        ]
        self.editor_panel = FaceEditorPanel(self._handle_gui_action)

    def ensure_size(self, width: int, height: int) -> None:
        width = max(1, int(width))
        height = max(1, int(height))
        if (width, height) == (self.width, self.height) and self.framebuffer is not None:
            return
        self.width, self.height = width, height
        self.source_texture = pyglet.image.Texture.create(width, height, min_filter=GL_LINEAR, mag_filter=GL_LINEAR)
        self.depth_buffer = Renderbuffer(width, height, GL_DEPTH_COMPONENT24)
        self.framebuffer = Framebuffer()
        self.framebuffer.attach_texture(self.source_texture)
        self.framebuffer.attach_renderbuffer(self.depth_buffer, attachment=GL_DEPTH_ATTACHMENT)
        self.ready = False
        self.analysis_pending = True
        self.analysis_error = None

    def render(
        self,
        draw_source: Callable[[], None],
        render_width: int,
        render_height: int,
        viewport_width: int,
        viewport_height: int,
        overlay_width: int,
        overlay_height: int,
    ) -> None:
        self.ensure_size(render_width, render_height)
        assert self.framebuffer is not None and self.source_texture is not None

        self.framebuffer.bind()
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_source()
        self.framebuffer.unbind()
        glViewport(0, 0, viewport_width, viewport_height)

        if self.analysis_pending:
            self._analyze_current_frame()
        self._draw_composite()
        if (
            self.edit_visible
            and self.ready
            and self.selected is not None
            and self.tool_mode == "select"
        ):
            self._draw_selection_overlay(overlay_width, overlay_height)
        if self.trim_mode and self.ready and self.trim_cursor_position is not None:
            self._draw_trim_overlay(overlay_width, overlay_height)
        if self.tool_mode == "cut" and self.cut_points:
            self._draw_cut_overlay(overlay_width, overlay_height)
        self.editor_panel.draw(
            overlay_width,
            overlay_height,
            EditorUiState(
                visible=self.gui_visible,
                tool=self.tool_mode,
                selected=self.selected,
                custom_names=tuple(part.name for part in self.custom_parts),
                cut_point_count=len(self.cut_points),
                selected_scale=(
                    self.transforms[self.selected].scale
                    if self.selected in self.transforms
                    else 1.0
                ),
                selected_rotation=(
                    self.transforms[self.selected].rotation_deg
                    if self.selected in self.transforms
                    else 0.0
                ),
            ),
        )

    def request_analysis(self) -> None:
        self.ready = False
        self.analysis_pending = True
        self.analysis_error = None

    def toggle_edit(self) -> bool:
        self.edit_visible = not self.edit_visible
        if not self.edit_visible:
            self.trim_mode = False
            self.trim_last_point = None
        if self.edit_visible and self.selected is None and self.geometry:
            self.selected = FEATURE_NAMES[0]
        return self.edit_visible

    def toggle_trim_mode(self) -> bool:
        self.trim_mode = not self.trim_mode
        if self.trim_mode:
            self.edit_visible = True
            self.tool_mode = "brow"
        else:
            self.trim_last_point = None
            if self.tool_mode == "brow":
                self.tool_mode = "select"
        return self.trim_mode

    def set_tool(self, tool: str) -> None:
        if tool not in {"select", "cut", "brow"}:
            return
        self.tool_mode = tool
        self.edit_visible = True
        self.trim_mode = tool == "brow"
        self.trim_last_point = None
        if tool != "cut":
            self.cut_points.clear()

    def toggle_gui(self) -> bool:
        self.gui_visible = not self.gui_visible
        self.edit_visible = self.gui_visible
        if not self.gui_visible:
            self.trim_mode = False
            self.trim_last_point = None
            self.cut_points.clear()
            self.tool_mode = "select"
        self.editor_panel.cache_key = None
        return self.gui_visible

    def handle_gui_click(self, x: float, y: float) -> bool:
        return self.editor_panel.handle_click(x, y)

    def cycle_selection(self) -> None:
        names = list(SELECTABLE_FEATURES) + [part.name for part in self.custom_parts]
        if not names:
            self.selected = None
            return
        current = "left_eye" if self.selected == "right_eye" else self.selected
        index = names.index(current) if current in names else -1
        self.selected = names[(index + 1) % len(names)]

    def on_cut_press(self, x: float, y: float, remove_last: bool = False) -> bool:
        if self.tool_mode != "cut" or not self.ready:
            return False
        if remove_last:
            if self.cut_points:
                self.cut_points.pop()
            return True
        point = (x / max(self.width, 1), y / max(self.height, 1))
        if len(self.cut_points) >= 3:
            first = self.cut_points[0]
            distance_px = math.hypot(
                (point[0] - first[0]) * self.width,
                (point[1] - first[1]) * self.height,
            )
            if distance_px <= 14.0:
                return self.finish_cut()
        if len(self.cut_points) < 64:
            self.cut_points.append(point)
        return True

    def finish_cut(self) -> bool:
        if len(self.cut_points) < 3 or len(self.custom_parts) >= MAX_CUSTOM_PARTS:
            return False
        self._push_undo()
        used = {part.name for part in self.custom_parts}
        number = 1
        while f"Cut {number}" in used:
            number += 1
        name = f"Cut {number}"
        part = CustomPart(name=name, polygon=tuple(self.cut_points))
        self.custom_parts.append(part)
        self.transforms[name] = FeatureTransform()
        self.geometry[name] = self._geometry_from_normalized_polygon(part.polygon)
        self.cut_points.clear()
        self.selected = name
        self.set_tool("select")
        self._upload_custom_masks()
        return True

    def cancel_cut(self) -> None:
        self.cut_points.clear()
        self.set_tool("select")

    def delete_selected_custom(self) -> bool:
        if self.selected is None:
            return False
        index = next(
            (i for i, part in enumerate(self.custom_parts) if part.name == self.selected),
            None,
        )
        if index is None:
            return False
        self._push_undo()
        name = self.custom_parts[index].name
        self.custom_parts.pop(index)
        self.transforms.pop(name, None)
        self.geometry.pop(name, None)
        self.selected = "left_eye"
        self._upload_custom_masks()
        return True

    def _handle_gui_action(self, action: str) -> None:
        if action == "show":
            self.gui_visible = True
            self.edit_visible = True
        elif action == "hide":
            self.gui_visible = False
            self.edit_visible = False
            self.trim_mode = False
            self.trim_last_point = None
            self.cut_points.clear()
            self.tool_mode = "select"
        elif action.startswith("tool:"):
            self.set_tool(action.split(":", 1)[1])
        elif action == "cut:finish":
            self.finish_cut()
        elif action == "cut:cancel":
            self.cancel_cut()
        elif action.startswith("part:"):
            name = action.split(":", 1)[1]
            if name in self.transforms:
                self.selected = name
                self.set_tool("select")
        elif action.startswith("rotate:"):
            self.rotate_selected(float(action.split(":", 1)[1]))
        elif action.startswith("scale:"):
            self.scale_selected(float(action.split(":", 1)[1]))
        elif action == "delete":
            self.delete_selected_custom()
        elif action == "undo":
            self.undo()
        elif action == "reset":
            self.reset()
        elif action == "save":
            self.save()
        self.editor_panel.cache_key = None

    def on_mouse_press(self, x: float, y: float) -> bool:
        if not self.edit_visible or not self.ready:
            return False
        picked = self._pick_feature(x / max(self.width, 1), y / max(self.height, 1))
        if picked is None:
            self.selected = None
            return True
        self.selected = picked
        self._push_undo()
        transform = self.transforms[picked]
        self.drag_start = (x / self.width, y / self.height)
        self.drag_origin = (transform.offset_x, transform.offset_y)
        return True

    def on_mouse_drag(self, x: float, y: float) -> bool:
        if self.selected is None or self.drag_start is None or self.drag_origin is None:
            return False
        nx, ny = x / self.width, y / self.height
        transform = self.transforms[self.selected]
        transform.offset_x = float(np.clip(self.drag_origin[0] + nx - self.drag_start[0], -0.5, 0.5))
        transform.offset_y = float(np.clip(self.drag_origin[1] + ny - self.drag_start[1], -0.5, 0.5))
        return True

    def on_rotation_press(self, x: float, y: float) -> bool:
        if not self.edit_visible or not self.ready:
            return False
        picked = self._pick_feature(x / max(self.width, 1), y / max(self.height, 1))
        if picked is None:
            self.selected = None
            return True
        self.selected = picked
        self._push_undo()
        self.rotation_drag_start_x = x
        self.rotation_drag_origin = self.transforms[picked].rotation_deg
        return True

    def on_rotation_drag(self, x: float) -> bool:
        if (
            self.selected is None
            or self.rotation_drag_start_x is None
            or self.rotation_drag_origin is None
        ):
            return False
        delta_degrees = (x - self.rotation_drag_start_x) * 0.35
        transform = self.transforms[self.selected]
        transform.rotation_deg = float(
            np.clip(self.rotation_drag_origin + delta_degrees, -45.0, 45.0)
        )
        self._sync_eye_pair(self.selected)
        return True

    def on_mouse_release(self) -> None:
        self.drag_start = None
        self.drag_origin = None
        self.rotation_drag_start_x = None
        self.rotation_drag_origin = None
        self.trim_last_point = None

    def on_trim_press(self, x: float, y: float, erase: bool) -> bool:
        if not self.trim_mode or not self.ready or self.trim_mask is None:
            return False
        self._push_undo()
        self.trim_last_point = (x, y)
        self.trim_cursor_position = (x / self.width, y / self.height)
        self._paint_trim_segment((x, y), (x, y), erase)
        return True

    def on_trim_drag(self, x: float, y: float, erase: bool) -> bool:
        if not self.trim_mode or self.trim_mask is None or self.trim_last_point is None:
            return False
        current = (x, y)
        self._paint_trim_segment(self.trim_last_point, current, erase)
        self.trim_last_point = current
        self.trim_cursor_position = (x / self.width, y / self.height)
        return True

    def update_trim_cursor(self, x: float, y: float) -> None:
        if self.trim_mode:
            self.trim_cursor_position = (x / max(self.width, 1), y / max(self.height, 1))

    def adjust_trim_brush(self, steps: float) -> bool:
        if not self.trim_mode:
            return False
        self.trim_brush_radius = float(
            np.clip(self.trim_brush_radius * (1.12 ** steps), 4.0, 80.0)
        )
        return True

    def scale_selected(self, steps: float) -> bool:
        if not self.edit_visible or self.selected is None:
            return False
        self._push_undo()
        transform = self.transforms[self.selected]
        transform.scale = float(np.clip(transform.scale * (1.04 ** steps), 0.45, 2.2))
        self._sync_eye_pair(self.selected)
        return True

    def rotate_selected(self, degrees: float) -> bool:
        if not self.edit_visible or self.selected is None:
            return False
        self._push_undo()
        transform = self.transforms[self.selected]
        transform.rotation_deg = float(np.clip(transform.rotation_deg + degrees, -45.0, 45.0))
        self._sync_eye_pair(self.selected)
        return True

    def reset(self) -> None:
        self._push_undo()
        self.transforms = {name: FeatureTransform() for name in FEATURE_NAMES}
        for part in self.custom_parts:
            self.transforms[part.name] = FeatureTransform()
        if self.trim_mask is not None:
            self.trim_mask.fill(0)
            self._upload_trim_mask()

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.transforms, trim_mask, custom_parts = self.undo_stack.pop()
        self.custom_parts = list(custom_parts)
        self.trim_mask = None if trim_mask is None else trim_mask.copy()
        for name in list(self.geometry):
            if name not in FEATURE_NAMES:
                self.geometry.pop(name)
        for part in self.custom_parts:
            self.geometry[part.name] = self._geometry_from_normalized_polygon(part.polygon)
        self._upload_trim_mask()
        self._upload_custom_masks()
        return True

    def save(self) -> None:
        payload = {
            "version": 3,
            "model": self.model_path.name,
            "features": {name: self.transforms[name].as_dict() for name in FEATURE_NAMES},
            "custom_parts": [
                part.as_dict(self.transforms[part.name]) for part in self.custom_parts
            ],
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self.trim_mask is not None:
            cv2.imwrite(str(self.trim_mask_path), self.trim_mask)
        print(f"[FACE EDITOR] saved to {self.state_path}")

    def close(self) -> None:
        analyzer = getattr(self, "_analyzer", None)
        if analyzer is not None:
            analyzer.close()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if raw.get("model") not in {None, self.model_path.name}:
                return
            features = raw.get("features", {})
            self.transforms = {
                name: FeatureTransform.from_dict(features.get(name)) for name in FEATURE_NAMES
            }
            self.custom_parts = []
            used_names: set[str] = set(FEATURE_NAMES)
            custom_raw = raw.get("custom_parts", [])
            if isinstance(custom_raw, list):
                for index, item in enumerate(custom_raw[:MAX_CUSTOM_PARTS], start=1):
                    if not isinstance(item, dict):
                        continue
                    polygon_raw = item.get("polygon", [])
                    if not isinstance(polygon_raw, list) or len(polygon_raw) < 3:
                        continue
                    polygon: list[tuple[float, float]] = []
                    for point in polygon_raw[:64]:
                        if not isinstance(point, (list, tuple)) or len(point) != 2:
                            continue
                        polygon.append(
                            (
                                float(np.clip(float(point[0]), 0.0, 1.0)),
                                float(np.clip(float(point[1]), 0.0, 1.0)),
                            )
                        )
                    if len(polygon) < 3:
                        continue
                    name = str(item.get("name", f"Cut {index}"))[:24]
                    if not name or name in used_names:
                        name = f"Cut {index}"
                    used_names.add(name)
                    part = CustomPart(name=name, polygon=tuple(polygon))
                    self.custom_parts.append(part)
                    self.transforms[name] = FeatureTransform.from_dict(item.get("transform"))
            print(f"[FACE EDITOR] loaded {self.state_path}")
        except Exception as exc:
            print(f"[FACE EDITOR] ignored invalid state: {exc}")

    def _analyze_current_frame(self) -> None:
        assert self.source_texture is not None
        self.analysis_pending = False
        try:
            raw = self.source_texture.get_image_data().get_data("RGBA", self.width * 4)
            rgba_bottom_up = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
            rgba_top_down = np.ascontiguousarray(np.flipud(rgba_bottom_up))
            if not hasattr(self, "_analyzer"):
                self._analyzer = FacePartAnalyzer(self.landmarker_model_path)
            analysis = self._analyzer.analyze(rgba_top_down)
            self.base_texture = self._texture_from_rgba(analysis.base_rgba)
            self.mask_a_texture = self._texture_from_rgba(analysis.mask_a_rgba)
            self.geometry = analysis.geometry
            for part in self.custom_parts:
                self.geometry[part.name] = self._geometry_from_normalized_polygon(part.polygon)
            self._ensure_trim_mask()
            self._upload_custom_masks()
            self._sync_eye_pair("left_eye")
            self.ready = True
            self.selected = self.selected or FEATURE_NAMES[0]
            print(
                f"[FACE EDITOR] detected face and {len(FEATURE_NAMES)} editable parts "
                f"(rotation={analysis.rotation_quadrants * 90} deg)"
            )
        except Exception as exc:
            self.analysis_error = str(exc)
            self.ready = False
            print(f"[FACE EDITOR] analysis failed; using normal rendering: {exc}")

    @staticmethod
    def _texture_from_rgba(rgba_top_down: np.ndarray) -> pyglet.image.Texture:
        height, width = rgba_top_down.shape[:2]
        return pyglet.image.ImageData(
            width,
            height,
            "RGBA",
            np.ascontiguousarray(rgba_top_down).tobytes(),
            pitch=-width * 4,
        ).get_texture()

    def _ensure_trim_mask(self) -> None:
        if self.trim_mask is not None and self.trim_mask.shape == (self.height, self.width):
            self._upload_trim_mask()
            return
        loaded = (
            cv2.imread(str(self.trim_mask_path), cv2.IMREAD_GRAYSCALE)
            if self.trim_mask_path.exists()
            else None
        )
        if loaded is None:
            self.trim_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        elif loaded.shape != (self.height, self.width):
            self.trim_mask = cv2.resize(
                loaded,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            self.trim_mask = loaded
        self._upload_trim_mask()

    def _upload_trim_mask(self) -> None:
        if self.width <= 0 or self.height <= 0:
            return
        mask = self.trim_mask
        if mask is None or mask.shape != (self.height, self.width):
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
        packed = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        packed[:, :, 0] = mask
        self.mask_b_texture = self._texture_from_rgba(packed)

    def _upload_custom_masks(self) -> None:
        if self.width <= 0 or self.height <= 0:
            return
        packed = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        for channel, part in enumerate(self.custom_parts[:MAX_CUSTOM_PARTS]):
            contour = np.array(
                [
                    (
                        int(np.clip(round(x * (self.width - 1)), 0, self.width - 1)),
                        int(np.clip(round((1.0 - y) * (self.height - 1)), 0, self.height - 1)),
                    )
                    for x, y in part.polygon
                ],
                dtype=np.int32,
            )
            hard = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(hard, [contour], 255, lineType=cv2.LINE_AA)
            packed[:, :, channel] = cv2.GaussianBlur(hard, (0, 0), 1.0)
        self.custom_mask_texture = self._texture_from_rgba(packed)

    @staticmethod
    def _geometry_from_normalized_polygon(
        polygon: tuple[tuple[float, float], ...],
    ) -> FeatureGeometry:
        points = np.asarray(polygon, dtype=np.float32)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = points.mean(axis=0)
        return FeatureGeometry(
            center=(float(center[0]), float(center[1])),
            size=(float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1])),
            outline=polygon,
        )

    def _paint_trim_segment(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        erase: bool,
    ) -> None:
        if self.trim_mask is None:
            return
        to_top_down = lambda point: (
            int(np.clip(round(point[0]), 0, self.width - 1)),
            int(np.clip(round(self.height - 1 - point[1]), 0, self.height - 1)),
        )
        value = 0 if erase else 255
        thickness = max(1, int(round(self.trim_brush_radius * 2.0)))
        cv2.line(
            self.trim_mask,
            to_top_down(start),
            to_top_down(end),
            value,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        self._upload_trim_mask()

    def _draw_composite(self) -> None:
        assert self.source_texture is not None
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.program.use()
        self.program["editor_ready"] = int(self.ready)
        enabled = {
            name: float(
                abs(self.transforms[name].offset_x) > 1e-7
                or abs(self.transforms[name].offset_y) > 1e-7
                or abs(self.transforms[name].scale - 1.0) > 1e-7
                or abs(self.transforms[name].rotation_deg) > 1e-7
            )
            for name in FEATURE_NAMES
        }
        textures = (
            self.source_texture,
            self.base_texture or self.source_texture,
            self.mask_a_texture or self.source_texture,
            self.mask_b_texture or self.source_texture,
            self.custom_mask_texture or self.source_texture,
        )
        for unit, texture in enumerate(textures):
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, texture.id)
        self.program["source_tex"] = 0
        self.program["base_tex"] = 1
        self.program["mask_a_tex"] = 2
        self.program["mask_b_tex"] = 3
        self.program["custom_mask_tex"] = 4
        for index, name in enumerate(FEATURE_NAMES):
            geometry = self.geometry.get(name, FeatureGeometry((0.5, 0.5), (0.1, 0.1), ()))
            transform = self.transforms[name]
            self.program[f"center_{index}"] = geometry.center
            self.program[f"offset_{index}"] = (transform.offset_x, transform.offset_y)
            self.program[f"scale_{index}"] = transform.scale
            self.program[f"rotation_{index}"] = math.radians(transform.rotation_deg)
            self.program[f"enabled_{index}"] = enabled[name]
        for index in range(MAX_CUSTOM_PARTS):
            if index < len(self.custom_parts):
                name = self.custom_parts[index].name
                geometry = self.geometry[name]
                transform = self.transforms[name]
                custom_enabled = float(
                    abs(transform.offset_x) > 1e-7
                    or abs(transform.offset_y) > 1e-7
                    or abs(transform.scale - 1.0) > 1e-7
                    or abs(transform.rotation_deg) > 1e-7
                )
            else:
                geometry = FeatureGeometry((0.5, 0.5), (0.1, 0.1), ())
                transform = FeatureTransform()
                custom_enabled = 0.0
            self.program[f"custom_center_{index}"] = geometry.center
            self.program[f"custom_offset_{index}"] = (transform.offset_x, transform.offset_y)
            self.program[f"custom_scale_{index}"] = transform.scale
            self.program[f"custom_rotation_{index}"] = math.radians(transform.rotation_deg)
            self.program[f"custom_enabled_{index}"] = custom_enabled
        self.quad.draw(pyglet.gl.GL_TRIANGLES)

    def _pick_feature(self, x: float, y: float) -> Optional[str]:
        best: Optional[str] = None
        best_distance = float("inf")
        names = [part.name for part in reversed(self.custom_parts)] + list(FEATURE_NAMES)
        for name in names:
            geometry = self.geometry[name]
            transform = self.transforms[name]
            cx = geometry.center[0] + transform.offset_x
            cy = geometry.center[1] + transform.offset_y
            angle = math.radians(-transform.rotation_deg)
            c, s = math.cos(angle), math.sin(angle)
            qx = (x - cx) / max(transform.scale, 1e-6)
            qy = (y - cy) / max(transform.scale, 1e-6)
            source_point = (
                geometry.center[0] + qx * c - qy * s,
                geometry.center[1] + qx * s + qy * c,
            )
            if self._point_in_polygon(source_point, geometry.outline):
                distance = (x - cx) ** 2 + (y - cy) ** 2
                if distance < best_distance:
                    best, best_distance = name, distance
        return best

    def _draw_selection_overlay(self, width: int, height: int) -> None:
        assert self.selected is not None
        selected_names = EYE_NAMES if self.selected in EYE_NAMES else (self.selected,)
        line_index = 0
        for center_index, name in enumerate(selected_names):
            geometry = self.geometry[name]
            transform = self.transforms[name]
            cx = (geometry.center[0] + transform.offset_x) * width
            cy = (geometry.center[1] + transform.offset_y) * height
            angle = math.radians(transform.rotation_deg)
            c, s = math.cos(angle), math.sin(angle)
            outline = []
            for x, y in geometry.outline:
                dx = (x - geometry.center[0]) * transform.scale * width
                dy = (y - geometry.center[1]) * transform.scale * height
                outline.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
            for point_index, point in enumerate(outline):
                line = self.overlay_lines[line_index]
                line.x, line.y = point
                line.x2, line.y2 = outline[(point_index + 1) % len(outline)]
                line.opacity = 255
                line_index += 1
            center = self.overlay_centers[center_index]
            center.x, center.y, center.opacity = cx, cy, 255
        for line in self.overlay_lines[line_index:]:
            line.opacity = 0
        for center in self.overlay_centers[len(selected_names):]:
            center.opacity = 0
        self.overlay_batch.draw()

    def _draw_trim_overlay(self, width: int, height: int) -> None:
        assert self.trim_cursor_position is not None
        self.trim_cursor.x = self.trim_cursor_position[0] * width
        self.trim_cursor.y = self.trim_cursor_position[1] * height
        self.trim_cursor.radius = self.trim_brush_radius * width / max(self.width, 1)
        self.trim_overlay_batch.draw()

    def _draw_cut_overlay(self, width: int, height: int) -> None:
        points = [(x * width, y * height) for x, y in self.cut_points]
        for index, line in enumerate(self.cut_overlay_lines):
            if index < max(0, len(points) - 1):
                line.x, line.y = points[index]
                line.x2, line.y2 = points[index + 1]
                line.opacity = 255
            else:
                line.opacity = 0
        for index, point in enumerate(points):
            marker = self.cut_overlay_points[index]
            marker.x, marker.y, marker.opacity = point[0], point[1], 255
        for marker in self.cut_overlay_points[len(points):]:
            marker.opacity = 0
        self.cut_overlay_batch.draw()

    def _sync_eye_pair(self, source_name: str) -> None:
        if source_name not in EYE_NAMES:
            return
        target_name = "right_eye" if source_name == "left_eye" else "left_eye"
        source = self.transforms[source_name]
        target = self.transforms[target_name]
        target.scale = source.scale
        target.rotation_deg = source.rotation_deg

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float],
        polygon: tuple[tuple[float, float], ...],
    ) -> bool:
        if len(polygon) < 3:
            return False
        x, y = point
        inside = False
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = previous
            x2, y2 = current
            if (y1 > y) != (y2 > y):
                denominator = y2 - y1
                if abs(denominator) < 1e-12:
                    previous = current
                    continue
                crossing = (x2 - x1) * (y - y1) / denominator + x1
                if x < crossing:
                    inside = not inside
            previous = current
        return inside

    def _push_undo(self) -> None:
        snapshot = {
            name: FeatureTransform.from_dict(transform.as_dict())
            for name, transform in self.transforms.items()
        }
        trim_snapshot = None if self.trim_mask is None else self.trim_mask.copy()
        self.undo_stack.append((snapshot, trim_snapshot, list(self.custom_parts)))
        if len(self.undo_stack) > 40:
            self.undo_stack.pop(0)

    @staticmethod
    def _make_program() -> ShaderProgram:
        vertex = """
            #version 330 core
            in vec2 position;
            in vec2 texcoord;
            out vec2 uv;
            void main() {
                uv = texcoord;
                gl_Position = vec4(position, 0.0, 1.0);
            }
        """
        feature_uniforms = "\n".join(
            f"uniform vec2 center_{i}; uniform vec2 offset_{i}; uniform float scale_{i}; "
            f"uniform float rotation_{i}; uniform float enabled_{i};"
            for i in range(len(FEATURE_NAMES))
        )
        custom_uniforms = "\n".join(
            f"uniform vec2 custom_center_{i}; uniform vec2 custom_offset_{i}; "
            f"uniform float custom_scale_{i}; uniform float custom_rotation_{i}; "
            f"uniform float custom_enabled_{i};"
            for i in range(MAX_CUSTOM_PARTS)
        )
        removal_calls = "\n".join(
            f"color = remove_part(color, {i}, enabled_{i});"
            for i in range(len(FEATURE_NAMES))
        )
        feature_calls = "\n".join(
            f"color = apply_part(color, {i}, center_{i}, offset_{i}, scale_{i}, rotation_{i});"
            for i in (FEATURE_NAMES.index(name) for name in COMPOSITE_ORDER)
        )
        custom_removal_calls = "\n".join(
            f"color = remove_custom(color, {i}, custom_enabled_{i});"
            for i in range(MAX_CUSTOM_PARTS)
        )
        custom_feature_calls = "\n".join(
            f"color = apply_custom(color, {i}, custom_center_{i}, custom_offset_{i}, "
            f"custom_scale_{i}, custom_rotation_{i});"
            for i in range(MAX_CUSTOM_PARTS)
        )
        fragment = f"""
            #version 330 core
            in vec2 uv;
            uniform sampler2D source_tex;
            uniform sampler2D base_tex;
            uniform sampler2D mask_a_tex;
            uniform sampler2D mask_b_tex;
            uniform sampler2D custom_mask_tex;
            uniform int editor_ready;
            {feature_uniforms}
            {custom_uniforms}
            out vec4 fragColor;

            float feature_mask(int index, vec2 p) {{
                vec4 a = texture(mask_a_tex, p);
                if (index == 0) return a.r;
                if (index == 1) return a.g;
                if (index == 2) return a.b;
                return a.a;
            }}

            vec4 remove_part(vec4 under, int index, float part_enabled) {{
                float alpha = smoothstep(0.03, 0.92, feature_mask(index, uv)) * part_enabled;
                return mix(under, texture(base_tex, uv), alpha);
            }}

            float custom_mask(int index, vec2 p) {{
                vec4 mask = texture(custom_mask_tex, p);
                if (index == 0) return mask.r;
                if (index == 1) return mask.g;
                if (index == 2) return mask.b;
                return mask.a;
            }}

            vec4 remove_custom(vec4 under, int index, float part_enabled) {{
                float alpha = smoothstep(0.03, 0.92, custom_mask(index, uv)) * part_enabled;
                return mix(under, texture(base_tex, uv), alpha);
            }}

            vec4 apply_part(
                vec4 under,
                int index,
                vec2 center,
                vec2 offset,
                float part_scale,
                float angle
            ) {{
                float c = cos(-angle);
                float s = sin(-angle);
                vec2 q = (uv - center - offset) / max(part_scale, 0.001);
                vec2 source_uv = center + vec2(c * q.x - s * q.y, s * q.x + c * q.y);
                if (source_uv.x < 0.0 || source_uv.x > 1.0 || source_uv.y < 0.0 || source_uv.y > 1.0) return under;
                float alpha = smoothstep(0.03, 0.92, feature_mask(index, source_uv));
                vec4 feature = texture(source_tex, source_uv);
                return mix(under, feature, alpha * feature.a);
            }}

            vec4 apply_custom(
                vec4 under,
                int index,
                vec2 center,
                vec2 offset,
                float part_scale,
                float angle
            ) {{
                float c = cos(-angle);
                float s = sin(-angle);
                vec2 q = (uv - center - offset) / max(part_scale, 0.001);
                vec2 source_uv = center + vec2(c * q.x - s * q.y, s * q.x + c * q.y);
                if (source_uv.x < 0.0 || source_uv.x > 1.0 || source_uv.y < 0.0 || source_uv.y > 1.0) return under;
                float alpha = smoothstep(0.03, 0.92, custom_mask(index, source_uv));
                vec4 feature = texture(source_tex, source_uv);
                return mix(under, feature, alpha * feature.a);
            }}

            void main() {{
                if (editor_ready == 0) {{
                    fragColor = texture(source_tex, uv);
                    return;
                }}
                float trim_alpha = smoothstep(0.03, 0.92, texture(mask_b_tex, uv).r);
                vec4 color = mix(texture(source_tex, uv), texture(base_tex, uv), trim_alpha);
                {removal_calls}
                {custom_removal_calls}
                {feature_calls}
                {custom_feature_calls}
                fragColor = color;
            }}
        """
        return ShaderProgram(Shader(vertex, "vertex"), Shader(fragment, "fragment"))
