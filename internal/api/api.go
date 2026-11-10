// Package api is the REST control plane. In-memory state behind one mutex;
// ponytail: swap the store for SQLite/Postgres when jobs must survive restart.
package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"sync"

	"github.com/ayushgawai/gpu-job-orchestrator/internal/cluster"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/job"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/scheduler"
)

type Server struct {
	mu      sync.Mutex
	cluster *cluster.Cluster
	sched   *scheduler.Scheduler
	jobs    map[string]*job.Job
	byKey   map[string]string // idempotency key -> job ID
	seq     int
	now     float64 // simulated minutes since start

	// Prometheus counters
	submitted, scheduled, preempted, resumed, completed, lostGPUMin float64
}

func New(c *cluster.Cluster, quotas map[string]int) *Server {
	return &Server{
		cluster: c, sched: scheduler.New(c, scheduler.BestFit, quotas),
		jobs: map[string]*job.Job{}, byKey: map[string]string{},
	}
}

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { write(w, 200, map[string]string{"status": "ok"}) })
	mux.HandleFunc("GET /readyz", s.ready)
	mux.HandleFunc("POST /v1/jobs", s.submit)
	mux.HandleFunc("GET /v1/jobs", s.list)
	mux.HandleFunc("GET /v1/jobs/{id}", s.get)
	mux.HandleFunc("DELETE /v1/jobs/{id}", s.cancel)
	mux.HandleFunc("GET /v1/pools", s.pools)
	mux.HandleFunc("POST /v1/nodes/{id}/preempt", s.preempt)
	mux.HandleFunc("POST /v1/nodes/{id}/restore", s.restore)
	mux.HandleFunc("GET /metrics", s.metrics)
	return mux
}

func write(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func errMsg(w http.ResponseWriter, code int, msg string) {
	write(w, code, map[string]string{"error": msg})
}

// Advance moves simulated time forward: running jobs progress and
// checkpoint, finished jobs release GPUs, queued jobs get placed.
// ponytail: scans every job ever submitted; keep a running set if job
// history grows past ~100k.
func (s *Server) Advance(minutes float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now += minutes
	for _, j := range s.jobs {
		if j.Status == job.Running && j.Advance(minutes) {
			j.Status = job.Succeeded
			s.sched.Release(j)
			s.completed++
		}
	}
	s.reconcile()
}

// reconcile places queued jobs in priority order. Caller holds s.mu.
func (s *Server) reconcile() {
	var q []*job.Job
	for _, j := range s.jobs {
		if j.Status == job.Queued {
			q = append(q, j)
		}
	}
	sort.Slice(q, func(a, b int) bool { return q[a].ID < q[b].ID }) // stable tie-break
	scheduler.SortQueue(q)
	for _, j := range q {
		allocs, err := s.sched.Schedule(j)
		if err != nil {
			continue
		}
		j.Placement, j.Status = allocs, job.Running
		s.scheduled++
		if j.Preemptions > 0 {
			j.Resumes++
			s.resumed++
		}
	}
}

func (s *Server) submit(w http.ResponseWriter, r *http.Request) {
	var spec job.Spec
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&spec); err != nil {
		errMsg(w, 400, "invalid json")
		return
	}
	key := r.Header.Get("Idempotency-Key")
	if key == "" {
		key = spec.IdempotencyKey
	}
	if err := spec.Validate(); err != nil {
		errMsg(w, 400, err.Error())
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if id, ok := s.byKey[key]; ok && key != "" {
		write(w, 200, s.jobs[id]) // idempotent replay: same job, no new work
		return
	}
	s.seq++
	j := &job.Job{ID: fmt.Sprintf("job-%06d", s.seq), Spec: spec, Status: job.Queued, SubmittedAtMin: s.now}
	s.jobs[j.ID] = j
	if key != "" {
		s.byKey[key] = j.ID
	}
	s.submitted++
	s.reconcile()
	write(w, 201, j)
}

func (s *Server) list(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*job.Job, 0, len(s.jobs))
	for _, j := range s.jobs {
		if st := r.URL.Query().Get("status"); st == "" || string(j.Status) == st {
			out = append(out, j)
		}
	}
	sort.Slice(out, func(a, b int) bool { return out[a].ID < out[b].ID })
	write(w, 200, out)
}

func (s *Server) get(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[r.PathValue("id")]
	if !ok {
		errMsg(w, 404, "not found")
		return
	}
	write(w, 200, j)
}

func (s *Server) cancel(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[r.PathValue("id")]
	if !ok {
		errMsg(w, 404, "not found")
		return
	}
	if j.Status != job.Queued && j.Status != job.Running {
		errMsg(w, 409, "job already "+string(j.Status))
		return
	}
	s.sched.Release(j)
	j.Status = job.Cancelled
	s.reconcile()
	write(w, 200, j)
}

func (s *Server) ready(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	write(w, 200, map[string]any{"status": "ready", "gpus": s.cluster.TotalGPUs(), "free": s.cluster.FreeGPUs()})
}

func (s *Server) pools(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := []map[string]any{}
	for _, p := range s.cluster.Pools {
		total, alloc, up := p.PoolGPUs()
		out = append(out, map[string]any{
			"name": p.Name, "provider": p.Provider, "region": p.Region, "gpu_type": p.GPUType,
			"capacity": p.Capacity, "price_per_gpu_hour": p.PricePerGPUHour,
			"total_gpus": total, "allocated_gpus": alloc, "up_gpus": up,
		})
	}
	write(w, 200, out)
}

// preempt simulates a spot reclaim: the node goes down, every job on it rolls
// back to its last checkpoint and is rescheduled, preferring another pool.
func (s *Server) preempt(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	n, ok := s.cluster.Node(r.PathValue("id"))
	if !ok {
		errMsg(w, 404, "node not found")
		return
	}
	ids := s.sched.PreemptNode(n.ID)
	for _, id := range ids {
		j := s.jobs[id]
		s.lostGPUMin += j.Preempt(n.Pool.Name)
		s.sched.Release(j)
		s.preempted++
	}
	s.reconcile()
	write(w, 200, map[string]any{"node": n.ID, "preempted_jobs": ids})
}

func (s *Server) restore(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	n, ok := s.cluster.Node(r.PathValue("id"))
	if !ok {
		errMsg(w, 404, "node not found")
		return
	}
	s.sched.RestoreNode(n.ID)
	s.reconcile()
	write(w, 200, map[string]any{"node": n.ID, "up": n.Up})
}

// metrics writes the Prometheus text exposition format by hand; a handful of
// counters and gauges does not need the client library.
func (s *Server) metrics(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	queued := 0
	for _, j := range s.jobs {
		if j.Status == job.Queued {
			queued++
		}
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	m := func(name, typ, help string, v float64) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s %s\n%s %g\n", name, help, name, typ, name, v)
	}
	m("gpu_jobs_submitted_total", "counter", "Jobs accepted (idempotent replays excluded).", s.submitted)
	m("gpu_jobs_scheduled_total", "counter", "Job placements, including resumes.", s.scheduled)
	m("gpu_jobs_completed_total", "counter", "Jobs that ran to completion.", s.completed)
	m("gpu_job_preemptions_total", "counter", "Jobs interrupted by spot reclaims.", s.preempted)
	m("gpu_job_resumes_total", "counter", "Preempted jobs restarted from checkpoint.", s.resumed)
	m("gpu_lost_gpu_minutes_total", "counter", "GPU-minutes of work lost since last checkpoint.", s.lostGPUMin)
	m("gpu_queue_depth", "gauge", "Jobs waiting for capacity.", float64(queued))
	fmt.Fprintf(w, "# HELP gpu_pool_gpus GPUs per pool by state.\n# TYPE gpu_pool_gpus gauge\n")
	for _, p := range s.cluster.Pools {
		total, alloc, up := p.PoolGPUs()
		l := fmt.Sprintf("pool=%q,capacity=%q", p.Name, p.Capacity)
		fmt.Fprintf(w, "gpu_pool_gpus{%s,state=\"total\"} %d\n", l, total)
		fmt.Fprintf(w, "gpu_pool_gpus{%s,state=\"allocated\"} %d\n", l, alloc)
		fmt.Fprintf(w, "gpu_pool_gpus{%s,state=\"up\"} %d\n", l, up)
	}
}
