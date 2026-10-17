from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyglet
from pyglet.gl import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    glBlendFunc,
    glClear,
    glClearColor,
    glEnable,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AVATAR_APP_DIR = PROJECT_ROOT / "apps" / "avatar"
if str(AVATAR_APP_DIR) not in sys.path:
    sys.path.insert(0, str(AVATAR_APP_DIR))

from conversation_avatar import GLBAvatar  # noqa: E402
from semantic_face_editor import SemanticFaceEditor  # noqa: E402


WINDOW_TITLE = "DANYA Robot Projection"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "avatar_nohair.glb"
DEFAULT_BACKGROUND = (185 / 255.0, 134 / 255.0, 118 / 255.0, 1.0)
CAMERA_STATE_PATH = Path(__file__).with_name("camera_state.json")
FACE_EDITOR_STATE_PATH = Path(__file__).with_name("semantic_face_state.json")
FACE_LANDMARKER_MODEL_PATH = PROJECT_ROOT / ".cache" / "mediapipe_face_landmarker.task"
FACE_RENDER_MAX_WIDTH = 1024
DEFAULT_CAMERA_STATE = {
    "eye": [-0.18, 0.08, 2.55],
    "target": [0.16, 0.04, 0.0],
    "fov_deg": 22.0,
}
CAMERA_STATE_KEYS = ("mirror_off", "mirror_on")


class RobotProjectionWindow(pyglet.window.Window):
    def __init__(
        self,
        width: int,
        height: int,
        fullscreen: bool,
        model_path: Path,
        background: tuple[float, float, float, float],
    ) -> None:
        config = pyglet.gl.Config(double_buffer=True, depth_size=24)
        super().__init__(
            width=width,
            height=height,
            caption=WINDOW_TITLE,
            fullscreen=fullscreen,
            resizable=True,
            config=config,
            vsync=True,
        )
        self.avatar = GLBAvatar(model_path)
        self.avatar.parts = [
            part for part in self.avatar.parts
            if part.name in {
                "Head_Mesh",
                "Eye_Mesh",
                "Teeth_Mesh",
                "Tongue_Mesh",
            }
        ]
        self.weights: dict[str, float] = {}
        self.start_time = pyglet.clock.get_default().time()
        self.background = background
        self.camera_states = self._load_camera_states()
        self.camera_controls_enabled = False
        self.mirror_display = False
        self.camera_state = self.camera_states[self._camera_state_key()]
        self.pressed_keys: set[int] = set()
        self.camera_dirty = False
        self.face_editor = None
        if FACE_LANDMARKER_MODEL_PATH.exists():
            self.face_editor = SemanticFaceEditor(
                model_path=model_path,
                state_path=FACE_EDITOR_STATE_PATH,
                landmarker_model_path=FACE_LANDMARKER_MODEL_PATH,
            )
        else:
            print(
                "[FACE EDITOR] disabled: missing MediaPipe model at "
                f"{FACE_LANDMARKER_MODEL_PATH}"
            )
        pyglet.clock.schedule_interval(self._update_camera, 1 / 60.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def on_draw(self) -> None:
        now = pyglet.clock.get_default().time()
        elapsed = now - self.start_time
        jaw = (math.sin(elapsed * 4.5) + 1.0) * 0.18
        self.weights = {
            "jawopen": jaw,
            "mouthopen": jaw * 0.75,
        }

        if self.face_editor is None:
            self._draw_avatar_scene()
            return

        framebuffer_width, framebuffer_height = self.get_framebuffer_size()
        render_width = min(framebuffer_width, FACE_RENDER_MAX_WIDTH)
        render_height = max(1, round(render_width * framebuffer_height / max(framebuffer_width, 1)))
        self.face_editor.render(
            self._draw_avatar_scene,
            render_width,
            render_height,
            framebuffer_width,
            framebuffer_height,
            self.width,
            self.height,
        )

    def _draw_avatar_scene(self) -> None:
        glClearColor(*self.background)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        original_model_matrix = self.avatar.model_matrix
        if self.mirror_display:
            mirror = np.eye(4, dtype=np.float32)
            mirror[0, 0] = -1.0
            self.avatar.model_matrix = mirror @ original_model_matrix
        try:
            self.avatar.draw(self.weights, self._make_view_matrix(), self._make_projection_matrix())
        finally:
            self.avatar.model_matrix = original_model_matrix

    def _make_projection_matrix(self) -> np.ndarray:
        aspect = max(self.width, 1) / max(self.height, 1)
        fov = math.radians(float(self.camera_state["fov_deg"]))
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
        eye = np.array(self.camera_state["eye"], dtype=np.float32)
        target = np.array(self.camera_state["target"], dtype=np.float32)
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

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        key = pyglet.window.key
        if self.face_editor is not None:
            command_modifier = modifiers & (key.MOD_CTRL | key.MOD_COMMAND)
            if symbol == key.B:
                enabled = self.face_editor.toggle_edit()
                print(f"[FACE EDITOR] controls {'on' if enabled else 'off'}")
                return
            if symbol == key.M:
                enabled = self.face_editor.toggle_trim_mode()
                print(f"[BROW TRIM] {'on' if enabled else 'off'}")
                return
            if symbol == key.C:
                self.face_editor.set_tool("cut")
                print("[FACE EDITOR] cut tool")
                return
            if symbol == key.G:
                visible = self.face_editor.toggle_gui()
                print(f"[FACE EDITOR GUI] {'on' if visible else 'off'}")
                return
            if symbol in {key.ENTER, key.RETURN} and self.face_editor.tool_mode == "cut":
                self.face_editor.finish_cut()
                return
            if symbol == key.ESCAPE and self.face_editor.tool_mode == "cut":
                self.face_editor.cancel_cut()
                return
            if symbol in {key.DELETE, key.BACKSPACE}:
                if self.face_editor.delete_selected_custom():
                    return
            if symbol == key.L:
                self.face_editor.request_analysis()
                print("[FACE EDITOR] rescanning face")
                return
            if symbol == key.R and self.face_editor.edit_visible:
                self.face_editor.reset()
                print("[FACE EDITOR] reset all feature transforms")
                return
            if symbol == key.Z and command_modifier:
                if self.face_editor.undo():
                    print("[FACE EDITOR] undo")
                return
            if symbol == key.BRACKETLEFT and self.face_editor.rotate_selected(-1.0):
                return
            if symbol == key.BRACKETRIGHT and self.face_editor.rotate_selected(1.0):
                return
            if symbol == key.TAB and self.face_editor.edit_visible and self.face_editor.geometry:
                self.face_editor.cycle_selection()
                return
        if symbol == pyglet.window.key.K:
            self.camera_controls_enabled = not self.camera_controls_enabled
            self.pressed_keys.clear()
            print(f"[CAMERA] controls {'on' if self.camera_controls_enabled else 'off'}")
            return
        if symbol == pyglet.window.key.F:
            self._save_camera_state()
            if self.face_editor is not None:
                self.face_editor.save()
            return
        if symbol == pyglet.window.key.T:
            self.camera_states[self._camera_state_key()] = self._copy_camera_state(self.camera_state)
            self.mirror_display = not self.mirror_display
            self.camera_state = self.camera_states[self._camera_state_key()]
            self.pressed_keys.clear()
            if self.face_editor is not None:
                self.face_editor.request_analysis()
            print(f"[DISPLAY] mirror {'on' if self.mirror_display else 'off'}; loaded {self._camera_state_key()}")
            return
        if not self.camera_controls_enabled:
            return
        self.pressed_keys.add(symbol)

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self.pressed_keys.discard(symbol)
        if self.camera_dirty and not self.pressed_keys:
            self.camera_dirty = False
            if self.face_editor is not None:
                self.face_editor.request_analysis()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if self.face_editor is None:
            return
        if button == pyglet.window.mouse.LEFT and self.face_editor.handle_gui_click(x, y):
            return
        editor_x = x * self.face_editor.width / max(self.width, 1)
        editor_y = y * self.face_editor.height / max(self.height, 1)
        self.face_editor.update_trim_cursor(editor_x, editor_y)
        if self.face_editor.tool_mode == "cut" and button in {
            pyglet.window.mouse.LEFT,
            pyglet.window.mouse.RIGHT,
        }:
            self.face_editor.on_cut_press(
                editor_x,
                editor_y,
                remove_last=button == pyglet.window.mouse.RIGHT,
            )
            return
        if self.face_editor.trim_mode and button in {
            pyglet.window.mouse.LEFT,
            pyglet.window.mouse.RIGHT,
        }:
            self.face_editor.on_trim_press(
                editor_x,
                editor_y,
                erase=button == pyglet.window.mouse.RIGHT,
            )
            return
        if button == pyglet.window.mouse.LEFT and self.face_editor.on_mouse_press(editor_x, editor_y):
            return
        if button == pyglet.window.mouse.RIGHT and self.face_editor.on_rotation_press(editor_x, editor_y):
            return

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if self.face_editor is not None and self.face_editor.trim_mode:
            editor_x = x * self.face_editor.width / max(self.width, 1)
            editor_y = y * self.face_editor.height / max(self.height, 1)
            self.face_editor.update_trim_cursor(editor_x, editor_y)
            if buttons & (pyglet.window.mouse.LEFT | pyglet.window.mouse.RIGHT):
                self.face_editor.on_trim_drag(
                    editor_x,
                    editor_y,
                    erase=bool(buttons & pyglet.window.mouse.RIGHT),
                )
            return
        if self.face_editor is not None and buttons & pyglet.window.mouse.LEFT:
            self.face_editor.on_mouse_drag(
                x * self.face_editor.width / max(self.width, 1),
                y * self.face_editor.height / max(self.height, 1),
            )
        if self.face_editor is not None and buttons & pyglet.window.mouse.RIGHT:
            self.face_editor.on_rotation_drag(
                x * self.face_editor.width / max(self.width, 1)
            )

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if self.face_editor is not None and button in {
            pyglet.window.mouse.LEFT,
            pyglet.window.mouse.RIGHT,
        }:
            self.face_editor.on_mouse_release()

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        if self.face_editor is not None and self.face_editor.trim_mode:
            self.face_editor.update_trim_cursor(
                x * self.face_editor.width / max(self.width, 1),
                y * self.face_editor.height / max(self.height, 1),
            )

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        if self.face_editor is not None:
            if self.face_editor.adjust_trim_brush(scroll_y):
                return
            self.face_editor.scale_selected(scroll_y)

    def on_close(self) -> None:
        if self.face_editor is not None:
            self.face_editor.close()
        super().on_close()

    def _update_camera(self, dt: float) -> None:
        if not self.camera_controls_enabled or not self.pressed_keys:
            return
        move = 0.85 * dt
        depth = 1.25 * dt
        eye = np.array(self.camera_state["eye"], dtype=np.float32)
        target = np.array(self.camera_state["target"], dtype=np.float32)

        forward = target - eye
        forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right = right / max(float(np.linalg.norm(right)), 1e-6)
        up = np.cross(right, forward)
        delta = np.zeros(3, dtype=np.float32)

        if pyglet.window.key.W in self.pressed_keys:
            delta += up * move
        if pyglet.window.key.S in self.pressed_keys:
            delta -= up * move
        if pyglet.window.key.A in self.pressed_keys:
            delta -= right * move
        if pyglet.window.key.D in self.pressed_keys:
            delta += right * move
        if pyglet.window.key.Q in self.pressed_keys:
            delta -= forward * depth
        if pyglet.window.key.E in self.pressed_keys:
            delta += forward * depth

        eye += delta
        target += delta
        self.camera_state["eye"] = [float(v) for v in eye]
        self.camera_state["target"] = [float(v) for v in target]
        if float(np.linalg.norm(delta)) > 0.0 and not self.camera_dirty:
            self.camera_dirty = True
            if self.face_editor is not None:
                self.face_editor.ready = False

    def _camera_state_key(self) -> str:
        return "mirror_on" if self.mirror_display else "mirror_off"

    @staticmethod
    def _copy_camera_state(state: dict[str, object]) -> dict[str, object]:
        return {
            "eye": [float(v) for v in state["eye"]],
            "target": [float(v) for v in state["target"]],
            "fov_deg": float(state["fov_deg"]),
        }

    @staticmethod
    def _is_camera_state(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        return (
            isinstance(value.get("eye"), list)
            and len(value["eye"]) == 3
            and isinstance(value.get("target"), list)
            and len(value["target"]) == 3
            and "fov_deg" in value
        )

    @classmethod
    def _load_camera_states(cls) -> dict[str, dict[str, object]]:
        default_state = cls._copy_camera_state(DEFAULT_CAMERA_STATE)
        states = {
            "mirror_off": cls._copy_camera_state(default_state),
            "mirror_on": cls._copy_camera_state(default_state),
        }
        if CAMERA_STATE_PATH.exists():
            try:
                state = json.loads(CAMERA_STATE_PATH.read_text(encoding="utf-8"))
                if all(cls._is_camera_state(state.get(key)) for key in CAMERA_STATE_KEYS):
                    return {
                        "mirror_off": cls._copy_camera_state(state["mirror_off"]),
                        "mirror_on": cls._copy_camera_state(state["mirror_on"]),
                    }
                if cls._is_camera_state(state):
                    legacy_state = cls._copy_camera_state(state)
                    return {
                        "mirror_off": cls._copy_camera_state(legacy_state),
                        "mirror_on": cls._copy_camera_state(legacy_state),
                    }
            except Exception:
                pass
        return states

    def _save_camera_state(self) -> None:
        self.camera_states[self._camera_state_key()] = self._copy_camera_state(self.camera_state)
        payload = {
            "active_mirror": self.mirror_display,
            "mirror_off": self.camera_states["mirror_off"],
            "mirror_on": self.camera_states["mirror_on"],
        }
        CAMERA_STATE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[CAMERA] saved {self._camera_state_key()} to {CAMERA_STATE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pyglet robot projection renderer")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="GLB model path. Defaults to assets/models/avatar_nohair.glb",
    )
    parser.add_argument(
        "--background-rgb",
        default="185,134,118",
        help="Background color as R,G,B values from 0 to 255",
    )
    args = parser.parse_args()
    try:
        r, g, b = (max(0, min(255, int(part.strip()))) for part in args.background_rgb.split(",", 2))
        background = (r / 255.0, g / 255.0, b / 255.0, 1.0)
    except Exception:
        parser.error("--background-rgb must be like 185,134,118")

    RobotProjectionWindow(
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
        model_path=args.model if args.model.is_absolute() else PROJECT_ROOT / args.model,
        background=background,
    )
    pyglet.app.run()


if __name__ == "__main__":
    main()



