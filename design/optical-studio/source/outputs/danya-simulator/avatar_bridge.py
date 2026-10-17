"""Render the user's actual Open Campus v2 program; publish its own framebuffer.

No screen capture, no network services from Project-DANYA, no edits to that repo.
The original camera / semantic calibration files are read, never saved here.
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.project / 'open-campus-demo-v2/robot_projection'))
    os.environ.setdefault('PYGLET_SHADOW_WINDOW', '0')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    import pyglet
    pyglet.options.dpi_scaling = 'real'
    from PIL import Image
    import robot_projection as original

    class SimulatorFeed(original.RobotProjectionWindow):
        last_capture = 0.0
        frames = 0
        def on_draw(self):
            start = time.monotonic()
            super().on_draw()
            now = time.monotonic()
            if now - self.last_capture < 1 / 15:
                return
            self.last_capture = now
            buffer = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            picture = Image.frombytes('RGB', (buffer.width, buffer.height),
                                      buffer.get_data('RGB', -buffer.width * 3))
            picture = picture.resize((1280, 720), Image.Resampling.BILINEAR)
            payload = io.BytesIO()
            picture.save(payload, format='JPEG', quality=88)
            temp = args.output / 'frame.tmp'
            temp.write_bytes(payload.getvalue())
            temp.replace(args.output / 'frame.jpg')
            self.frames += 1
            status = {'frame': self.frames, 'time': time.time(), 'source': 'open-campus-demo-v2',
                      'width': 1280, 'height': 720, 'semantic_ready': bool(self.face_editor and self.face_editor.ready),
                      'semantic_error': getattr(self.face_editor, 'analysis_error', None),
                      'render_ms': round((time.monotonic()-start)*1000, 1)}
            temp = args.output / 'feed.tmp'
            temp.write_text(json.dumps(status, ensure_ascii=False))
            temp.replace(args.output / 'feed.json')

        def on_key_press(self, symbol, modifiers):
            if symbol == pyglet.window.key.ESCAPE:
                self.close()

        def on_mouse_press(self, *args): pass
        def on_mouse_drag(self, *args): pass
        def on_mouse_scroll(self, *args): pass

    window = SimulatorFeed(width=1280, height=720, fullscreen=False,
                           model_path=original.DEFAULT_MODEL_PATH,
                           background=original.DEFAULT_BACKGROUND)
    window.set_caption('DANYA · Open Campus v2 source — Simulator feed')
    # Honor the saved active camera, something the original main currently omits.
    if original.CAMERA_STATE_PATH.exists():
        saved = json.loads(original.CAMERA_STATE_PATH.read_text())
        window.mirror_display = bool(saved.get('active_mirror', False))
        window.camera_state = window.camera_states[window._camera_state_key()]
    if window.face_editor:
        window.face_editor.edit_visible = False
        window.face_editor.gui_visible = False
        window.face_editor.editor_panel.draw = lambda *a, **k: None
        window.face_editor.request_analysis()
    window.set_vsync(False)
    window.set_visible(False)
    # Explicit draws continue when another app is foreground or our window is hidden.
    # macOS otherwise throttles occluded window redraw events.
    def tick(dt):
        window.switch_to()
        window.dispatch_event('on_draw')
        window.flip()
    pyglet.clock.schedule_interval(tick, 1/15)
    pyglet.app.run(interval=None)

if __name__ == '__main__': main()
