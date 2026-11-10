# gpu-job-orchestrator

A small control plane that schedules AI training and inference jobs across
simulated GPU pools on AWS, GCP and on-prem. It is written in Go with no
dependencies outside the standard library. A Python SDK and CLI submit jobs.

It does gang scheduling, best-fit bin-packing, priorities, per-tenant quotas
and spot-first placement with on-demand fallback. When a spot node is
reclaimed, every job on it rolls back to its last checkpoint and resumes on
another pool.

## Design

```
   gpu-jobs CLI / Python SDK            Prometheus            loadtest
             |  REST + Idempotency-Key       | GET /metrics       |
             v                               v                    v
   +------------------------------------------------------------------+
   |  api.Server  (net/http, one mutex, in-memory job store)          |
   |   POST /v1/jobs   GET /v1/jobs[/{id}]   DELETE /v1/jobs/{id}     |
   |   GET /v1/pools   POST /v1/nodes/{id}/preempt|restore            |
   |   GET /healthz  /readyz  /metrics                                |
   |                                                                  |
   |   reconcile(): queued jobs -> SortQueue (priority, FIFO)          |
   |                -> Scheduler.Schedule (gang, all-or-nothing)       |
   |   Advance(dt): progress + checkpoint every N min -> complete      |
   +-------------------------------+----------------------------------+
                                   |
             +---------------------v-----------------------+
             |  scheduler (best-fit | first-fit baseline)  |
             |  quota check -> candidate pools:             |
             |    gpu type, spot allowed?, spot first,      |
             |    cheapest first, preempting pool last      |
             |  -> tightest-fit nodes inside ONE pool       |
             +---------------------+-----------------------+
                                   |
   +-----------+-----------+-------+-------+-----------+-------------+
   | aws H100  | aws A10G  | gcp A100      | gcp L4    | on-prem     |
   | od + spot | od + spot | od + spot     | spot      | A100, H100  |
   +-----------+-----------+---------------+-----------+-------------+
          cluster: 9 pools, 76 nodes, 512 simulated GPUs

   spot reclaim -> node down -> job.Preempt(): progress = last checkpoint,
   lost GPU-min counted -> Release whole gang -> requeue -> resume elsewhere
```

| Concern | Choice | Why / trade-off |
|---|---|---|
| Gang scheduling | All GPUs of a job are placed inside one pool or not at all | No partial gangs holding GPUs while they wait. A gang never spans pools, because cross-region NCCL is not realistic. |
| Bin-packing | Best-fit: fill the nodes with the fewest free GPUs first | Leaves whole nodes free for 8- and 16-GPU gangs. The baseline is first-fit in topology order. |
| Pool order | Spot before on-demand, then cheapest $/GPU-h | Uses cheap capacity first and falls back to on-demand when spot is full or the job disallows spot. |
| Priorities | Queue sorted by priority desc, then submit time | Simple and predictable. No aging yet, so starvation is possible (see limitations). |
| Quotas | Hard per-tenant GPU cap checked before placement | A rejected placement never consumes quota. |
| Preemption | Node down, so every job on it calls `Preempt`, the whole gang is released and the job is requeued | Work since the last checkpoint is counted as lost GPU-minutes. |
| Resume | Progress restored from `checkpoint_min`, and the pool that preempted the job is tried last | The job moves to another pool when one fits but never starves if that pool is the only fit. |
| Idempotency | `Idempotency-Key` header (or `idempotency_key` field) maps to the job ID | A replay returns `200` with the original job and creates no new work. |
| Metrics | Prometheus text format written by hand (`/metrics`) | Seven series plus per-pool gauges didn't justify the client library and its dependency tree. |
| State | In-memory behind one mutex | Easy to reason about. A restart loses all jobs (see limitations). |
| Time | `Advance(dt)` moves simulated job-minutes, and the server ticks 1 job-minute per second (`-minute`) | The same job model drives the API, the tests and the simulator. |

### Pools (512 GPUs)

| Pool | Provider | GPU | Capacity | $/GPU-h | Nodes x GPUs | Mean h to reclaim |
|---|---|---|---|---|---|---|
| aws-use1-h100-od | aws | H100 | on-demand | 6.90 | 4 x 8 | - |
| aws-use1-h100-spot | aws | H100 | spot | 2.75 | 6 x 8 | 18 |
| aws-usw2-a10g-od | aws | A10G | on-demand | 1.45 | 8 x 4 | - |
| aws-usw2-a10g-spot | aws | A10G | spot | 0.55 | 16 x 4 | 30 |
| gcp-usc1-a100-od | gcp | A100 | on-demand | 3.70 | 6 x 8 | - |
| gcp-usc1-a100-spot | gcp | A100 | spot | 1.45 | 8 x 8 | 12 |
| gcp-euw4-l4-spot | gcp | L4 | spot | 0.30 | 8 x 8 | 24 |
| onprem-dc1-a100 | onprem | A100 | on-demand | 1.10 | 12 x 8 | - |
| onprem-dc1-h100 | onprem | H100 | on-demand | 2.40 | 8 x 8 | - |

Prices are illustrative, not quotes.

## Layout

```
cmd/orchestrator   REST control plane (:8080)
cmd/simulate       replays 1,000 jobs, best-fit vs first-fit
cmd/loadtest       API latency (p50/p95/p99), in-process server by default
internal/cluster   pools, nodes, default 512-GPU topology
internal/job       job model, Advance (checkpointing), Preempt (rollback)
internal/scheduler gang + best-fit/first-fit, quotas, spot fallback
internal/sim       tick-based replay with random spot reclaims
internal/api       HTTP handlers, reconcile loop, /metrics
sdk/python         gpu_jobs.py: Client + `gpu-jobs` CLI (stdlib only)
```

## Run

```bash
go run ./cmd/orchestrator                 # :8080, 1 job-minute per second
go run ./cmd/orchestrator -minute 100ms   # faster clock

pip install -e sdk/python
gpu-jobs health
gpu-jobs submit llama-ft --tenant alpha --gpus 16 --gpu-type H100 --duration-min 90 --key ft-1
gpu-jobs submit embed-batch --tenant beta --gpus 2 --no-spot
gpu-jobs pools
gpu-jobs get job-000001
gpu-jobs preempt aws-use1-h100-spot-n00   # simulate a spot reclaim
gpu-jobs get job-000001                   # progress_min back at last checkpoint, new pool
gpu-jobs restore aws-use1-h100-spot-n00
curl -s localhost:8080/metrics
```

From Python:

```python
from gpu_jobs import Client
c = Client("http://localhost:8080")
job = c.submit("train-7b", tenant="alpha", gpus=8, gpu_type="A100", duration_min=120, idempotency_key="train-7b-v1")
print(c.get(job["id"])["status"])
```

## Test

```bash
go test -race ./...                                  # scheduler unit tests, API + sim e2e
python3 -m unittest discover -s sdk/python -v        # SDK / CLI
```

- `internal/scheduler`: gang all-or-nothing, best-fit packing, spot-first then
  on-demand fallback, quota enforcement and release, priority order, first-fit
  order, resume moves off the preempting pool, fallback to that pool when it is
  the only fit, restoring an up node is a no-op.
- `internal/sim`: **end-to-end preemption and resume**. A job runs 17 min on
  spot with 10-min checkpoints and is preempted, so it loses exactly 28 GPU-min.
  It resumes from minute 10 on the on-demand pool and needs exactly 20 more
  minutes. The test also checks that a 300-job replay completes with every
  preemption recovered.
- `internal/api`: health/ready, idempotent replay (one job, same ID), the same
  preempt→checkpoint→resume→finish flow over HTTP with metric assertions, queue
  draining by priority under quota, validation, cancel and 404/409.

## Bench

```bash
go run ./cmd/simulate                       # 1,000 jobs, seed 42
go run ./cmd/simulate -jobs 1000 -seed 7
go run ./cmd/loadtest -n 5000 -c 32         # in-process server
go run ./cmd/loadtest -base http://localhost:8080
```

RESULTS_PLACEHOLDER

## Limitations

- **Everything is simulated.** No cloud APIs, no real nodes, no containers.
  A job is a duration that ticks down. A "checkpoint" is just a recorded
  progress mark. Real checkpoint size and restore time are not modelled.
  Resume is assumed instant.
- **Preemption model is crude.** Each spot node fails independently with a
  fixed mean time to reclaim and comes back after 5 minutes. There is no
  correlated zone-wide reclaim and no 2-minute warning used to checkpoint early.
- **No persistence or HA.** State lives in one process behind one mutex. A
  restart loses every job. There is one replica and no leader election.
- **Scheduling is greedy, per tick.** There is no backfilling, no
  reservations for large gangs (a 16-GPU job can wait behind a stream of small
  ones) and no priority preemption of running low-priority jobs. There is no
  aging, so low-priority jobs can starve. Topology (NVLink, rack, network)
  inside a pool is ignored.
- **The utilization comparison is narrow.** Both schedulers see the same arrival
  trace. Utilization is total GPU-minutes used divided by 512 × makespan, so
  the gap mostly reflects makespan and packing on this one synthetic workload.
- **The load test is in-process loopback.** Its numbers measure handler and lock
  cost, not a network. Without the clock running, the queue keeps growing during
  the test (tenant quota fills), so submit latency includes reconcile scanning
  all jobs.
- **The SDK is minimal.** It is stdlib `urllib`, with no retries or async, and
  errors surface as `urllib.error.HTTPError`.
- No auth or TLS on the API.

UNFINISHED_PLACEHOLDER

## License

MIT, see [LICENSE](LICENSE).
