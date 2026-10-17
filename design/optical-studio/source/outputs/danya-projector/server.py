"""DANYA projector-only server. No CAD or avatar renderer is launched."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import argparse, functools, json, time, mimetypes, webbrowser

ROOT = Path(__file__).resolve().parent
FEED = ROOT.parent / 'danya-simulator' / 'runtime'
mimetypes.add_type('text/javascript', '.js')
class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split('?')[0]
        if route == '/api/status':
            try:
                d = json.loads((FEED / 'feed.json').read_text())
                d['live'] = time.time() - d['time'] < 3
            except (OSError, ValueError, KeyError):
                d = {'live': False}
            return self.payload(json.dumps(d).encode(), 'application/json')
        if route == '/api/frame':
            try: return self.payload((FEED / 'frame.jpg').read_bytes(), 'image/jpeg')
            except OSError: return self.send_error(503, 'No live feed')
        if route.endswith(('.py', '.command')):return self.send_error(403)
        return super().do_GET()
    def payload(self, data, mime):
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff');super().end_headers()
    def log_message(self,fmt,*args):
        if not self.path.startswith('/api/'):super().log_message(fmt,*args)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8771);p.add_argument('--no-browser',action='store_true');a=p.parse_args();url=f'http://127.0.0.1:{a.port}/'
    try:s=ThreadingHTTPServer(('127.0.0.1',a.port),functools.partial(Handler,directory=str(ROOT)))
    except OSError as e:raise SystemExit(f'Port {a.port} unavailable: {e}')
    print(url,flush=True)
    if not a.no_browser:webbrowser.open(url)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()
