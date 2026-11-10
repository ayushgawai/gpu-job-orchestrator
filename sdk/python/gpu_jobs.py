"""Tiny SDK + CLI for the gpu-job-orchestrator API (stdlib only)."""
import argparse, json, os, urllib.request, uuid

class Client:
    def __init__(self, base="http://localhost:8080"):
        self.base = base.rstrip("/")
    def _req(self, method, path, body=None, key=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data: req.add_header("Content-Type", "application/json")
        if key: req.add_header("Idempotency-Key", key)
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or b"{}")
    def health(self): return self._req("GET", "/healthz")
    def submit(self, name, tenant, gpus, duration_min=60, gpu_type=None, priority=5, allow_spot=True,
               idempotency_key=None, checkpoint_every_min=10):
        body = {"name": name, "tenant": tenant, "gpus": gpus, "duration_min": duration_min,
                "priority": priority, "allow_spot": allow_spot, "checkpoint_every_min": checkpoint_every_min}
        if gpu_type: body["gpu_type"] = gpu_type
        return self._req("POST", "/v1/jobs", body, key=idempotency_key or uuid.uuid4().hex)
    def get(self, job_id): return self._req("GET", f"/v1/jobs/{job_id}")
    def list(self): return self._req("GET", "/v1/jobs")
    def cancel(self, job_id): return self._req("DELETE", f"/v1/jobs/{job_id}")
    def pools(self): return self._req("GET", "/v1/pools")
    def preempt(self, node_id): return self._req("POST", f"/v1/nodes/{node_id}/preempt", {})
    def restore(self, node_id): return self._req("POST", f"/v1/nodes/{node_id}/restore", {})

def main(argv=None):
    p = argparse.ArgumentParser(prog="gpu-jobs")
    p.add_argument("--base", default=os.environ.get("GPU_ORCH_URL", "http://localhost:8080"))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("name"); s.add_argument("--tenant", default="alpha")
    s.add_argument("--gpus", type=int, default=1); s.add_argument("--duration-min", type=int, default=60)
    s.add_argument("--gpu-type", default=None); s.add_argument("--priority", type=int, default=5)
    s.add_argument("--no-spot", action="store_true"); s.add_argument("--key", default=None)
    s.add_argument("--checkpoint-every-min", type=int, default=10)
    for name in ("list", "pools", "health"):
        sub.add_parser(name)
    g = sub.add_parser("get"); g.add_argument("id")
    c = sub.add_parser("cancel"); c.add_argument("id")
    pr = sub.add_parser("preempt"); pr.add_argument("node")
    rs = sub.add_parser("restore"); rs.add_argument("node")
    a = p.parse_args(argv)
    cli = Client(a.base)
    if a.cmd == "submit":
        out = cli.submit(a.name, a.tenant, a.gpus, a.duration_min, a.gpu_type, a.priority, not a.no_spot, a.key,
                         a.checkpoint_every_min)
    elif a.cmd == "list": out = cli.list()
    elif a.cmd == "pools": out = cli.pools()
    elif a.cmd == "health": out = cli.health()
    elif a.cmd == "get": out = cli.get(a.id)
    elif a.cmd == "cancel": out = cli.cancel(a.id)
    elif a.cmd == "restore": out = cli.restore(a.node)
    else: out = cli.preempt(a.node)
    print(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
