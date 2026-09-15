from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from cs336_alignment.vllm_utils import VLLMServer, _http_json, wait_for_server


def test_weight_sync_server_reserves_gpu_headroom():
    assert VLLMServer(model_id="test").gpu_memory_utilization <= 0.75


def test_local_vllm_requests_bypass_http_proxy(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"world_size": 1}' if self.path == "/get_world_size" else b""
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        base_url = f"http://127.0.0.1:{server.server_port}"
        wait_for_server(base_url, None, 1)
        assert _http_json("GET", f"{base_url}/get_world_size") == {"world_size": 1}
    finally:
        server.shutdown()
        thread.join()
