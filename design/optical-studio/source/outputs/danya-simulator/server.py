"""Local-only static server with an Open Campus v2 framebuffer feed."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import subprocess
import sys
import time
import webbrowser

ROOT = Path(__file__).resolve().parent
PROJECT = Path.home() / 'Documents/Project-DANYA'
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/javascript', '.mjs')

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if not self.path.startswith('/api/'):
            super().log_message(fmt, *args)

    def do_GET(self):
        route = self.path.split('?')[0]
        if route == '/api/status':
            try:
                data = json.loads((ROOT/'runtime/feed.json').read_text())
                data['live'] = time.time()-data['time'] < 3
            except (OSError, ValueError):
                data = {'live': False, 'source': 'open-campus-demo-v2', 'frame': 0}
            return self.payload(json.dumps(data).encode(), 'application/json')
        if route == '/api/frame':
            try: return self.payload((ROOT/'runtime/frame.jpg').read_bytes(), 'image/jpeg')
            except OSError: return self.send_error(503, 'Open Campus v2 source is starting or unavailable')
        # Expose only this app, not Project-DANYA or parent directories.
        if route.startswith('/runtime/') or route.endswith('.py') or route.endswith('.command'):
            return self.send_error(403)
        return super().do_GET()

    def payload(self, data, mime):
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

def main():
    parser = argparse.ArgumentParser(description='Project DANYA optical simulator')
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--no-avatar', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), functools.partial(Handler, directory=str(ROOT)))
    child = None
    if not args.no_avatar:
        python = args.project/'.venv/bin/python'
        if python.exists():
            (ROOT/'runtime').mkdir(exist_ok=True)
            log = open(ROOT/'runtime/avatar.log', 'a')
            child = subprocess.Popen([str(python), '-u', str(ROOT/'avatar_bridge.py'),
                '--project', str(args.project), '--output', str(ROOT/'runtime')], stdout=log, stderr=log)
        else: print('v2 Python environment missing. Calibration pattern and local video remain available.')
    url = f'http://127.0.0.1:{args.port}'
    print(f'Project DANYA simulator: {url}', flush=True)
    if not args.no_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        if child is not None:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait()

if __name__ == '__main__': main()
