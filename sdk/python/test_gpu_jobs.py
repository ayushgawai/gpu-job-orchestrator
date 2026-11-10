import io, json, os, sys, unittest
from contextlib import redirect_stdout
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_jobs
from gpu_jobs import Client

class Resp:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def read(self): return json.dumps({"id": "job-1", "status": "running"}).encode()

class TestClient(unittest.TestCase):
    @patch("gpu_jobs.urllib.request.urlopen", return_value=Resp())
    def test_submit_sends_idempotency_key(self, mock_open):
        out = Client("http://x:8080").submit("train", "alpha", 8, idempotency_key="k1")
        req = mock_open.call_args[0][0]
        self.assertEqual(req.headers["Idempotency-key"], "k1")
        self.assertEqual(out["id"], "job-1")
        self.assertEqual(json.loads(req.data)["gpus"], 8)

    @patch("gpu_jobs.urllib.request.urlopen", return_value=Resp())
    def test_cli_submit_maps_flags(self, mock_open):
        with redirect_stdout(io.StringIO()):
            gpu_jobs.main(["--base", "http://x:8080", "submit", "ft", "--gpus", "4", "--gpu-type", "A100", "--no-spot"])
        req = mock_open.call_args[0][0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, "http://x:8080/v1/jobs")
        self.assertEqual((body["gpus"], body["gpu_type"], body["allow_spot"]), (4, "A100", False))
        self.assertTrue(req.headers["Idempotency-key"])  # auto-generated when not given

if __name__ == "__main__":
    unittest.main()
