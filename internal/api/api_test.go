package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ayushgawai/gpu-job-orchestrator/internal/cluster"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/job"
)

func testServer(t *testing.T) *Server {
	t.Helper()
	c, err := cluster.New([]cluster.PoolSpec{
		{Name: "spot", Provider: "aws", GPUType: "H100", Capacity: cluster.Spot, PricePerGPUHour: 1, GPUsPerNode: 8, Nodes: 2, MeanHoursToPreempt: 1},
		{Name: "od", Provider: "onprem", GPUType: "H100", Capacity: cluster.OnDemand, PricePerGPUHour: 5, GPUsPerNode: 8, Nodes: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	return New(c, map[string]int{"t1": 16})
}

func do(t *testing.T, h http.Handler, method, path string, body any, hdr ...string) (int, []byte) {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		json.NewEncoder(&buf).Encode(body)
	}
	req := httptest.NewRequest(method, path, &buf)
	for i := 0; i+1 < len(hdr); i += 2 {
		req.Header.Set(hdr[i], hdr[i+1])
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec.Code, rec.Body.Bytes()
}

func decode(t *testing.T, b []byte) job.Job {
	t.Helper()
	var j job.Job
	if err := json.Unmarshal(b, &j); err != nil {
		t.Fatalf("decode %s: %v", b, err)
	}
	return j
}

var trainSpec = map[string]any{"name": "train", "tenant": "t1", "gpus": 4, "gpu_type": "H100", "duration_min": 30, "checkpoint_every_min": 10, "allow_spot": true}

func TestHealthAndIdempotentSubmit(t *testing.T) {
	h := testServer(t).Handler()
	if code, _ := do(t, h, "GET", "/healthz", nil); code != 200 {
		t.Fatalf("healthz = %d", code)
	}
	if code, _ := do(t, h, "GET", "/readyz", nil); code != 200 {
		t.Fatalf("readyz = %d", code)
	}
	code1, b1 := do(t, h, "POST", "/v1/jobs", trainSpec, "Idempotency-Key", "key-1")
	code2, b2 := do(t, h, "POST", "/v1/jobs", trainSpec, "Idempotency-Key", "key-1")
	j1, j2 := decode(t, b1), decode(t, b2)
	if code1 != 201 || j1.ID == "" {
		t.Fatalf("submit code=%d job=%s", code1, b1)
	}
	if code2 != 200 || j2.ID != j1.ID {
		t.Fatalf("idempotent replay code=%d id=%s want %s", code2, j2.ID, j1.ID)
	}
	if j1.Status != job.Running || !j1.Placement[0].Spot {
		t.Fatalf("job should run on spot: %s", b1)
	}
	_, list := do(t, h, "GET", "/v1/jobs", nil)
	var all []job.Job
	json.Unmarshal(list, &all)
	if len(all) != 1 {
		t.Fatalf("replay must not create a second job, have %d", len(all))
	}
}

// End-to-end over HTTP: run on spot, checkpoint, get preempted, resume from
// the checkpoint on the on-demand pool, finish, free GPUs.
func TestPreemptResumeFromCheckpointE2E(t *testing.T) {
	s := testServer(t)
	h := s.Handler()
	_, b := do(t, h, "POST", "/v1/jobs", trainSpec)
	j := decode(t, b)
	s.Advance(17) // checkpoint at 10, 7 minutes un-checkpointed

	if code, _ := do(t, h, "POST", "/v1/nodes/"+j.Placement[0].NodeID+"/preempt", nil); code != 200 {
		t.Fatalf("preempt = %d", code)
	}
	_, b = do(t, h, "GET", "/v1/jobs/"+j.ID, nil)
	got := decode(t, b)
	if got.Status != job.Running || got.Resumes != 1 || got.ProgressMin != 10 {
		t.Fatalf("job should resume from checkpoint 10: %s", b)
	}
	if got.Placement[0].Pool != "od" {
		t.Fatalf("resume should move off the reclaimed spot pool: %s", b)
	}

	s.Advance(20)
	_, b = do(t, h, "GET", "/v1/jobs/"+j.ID, nil)
	if got = decode(t, b); got.Status != job.Succeeded {
		t.Fatalf("job should finish 20 min after resume: %s", b)
	}
	_, m := do(t, h, "GET", "/metrics", nil)
	for _, want := range []string{
		"gpu_job_preemptions_total 1", "gpu_job_resumes_total 1",
		"gpu_lost_gpu_minutes_total 28", "gpu_jobs_completed_total 1",
		`gpu_pool_gpus{pool="od",capacity="on-demand",state="allocated"} 0`,
	} {
		if !strings.Contains(string(m), want) {
			t.Fatalf("metrics missing %q:\n%s", want, m)
		}
	}
}

func TestQueueDrainsByPriorityWhenCapacityFrees(t *testing.T) {
	s := testServer(t)
	h := s.Handler()
	big := map[string]any{"name": "big", "tenant": "t1", "gpus": 16, "gpu_type": "H100", "duration_min": 5, "allow_spot": true}
	lo := map[string]any{"name": "lo", "tenant": "t1", "gpus": 8, "duration_min": 5, "priority": 1}
	hi := map[string]any{"name": "hi", "tenant": "t1", "gpus": 8, "duration_min": 5, "priority": 9}
	do(t, h, "POST", "/v1/jobs", big) // takes the tenant's whole 16-GPU quota
	_, bl := do(t, h, "POST", "/v1/jobs", lo)
	_, bh := do(t, h, "POST", "/v1/jobs", hi)
	if decode(t, bl).Status != job.Queued || decode(t, bh).Status != job.Queued {
		t.Fatal("quota should queue lo and hi")
	}
	s.Advance(5) // big finishes; hi and lo are on-demand only and one 8-GPU od node is free
	_, bh = do(t, h, "GET", "/v1/jobs/"+decode(t, bh).ID, nil)
	_, bl = do(t, h, "GET", "/v1/jobs/"+decode(t, bl).ID, nil)
	if decode(t, bh).Status != job.Running || decode(t, bl).Status != job.Queued {
		t.Fatalf("higher priority must win the free node: hi=%s lo=%s", bh, bl)
	}
}

func TestValidationAndCancel(t *testing.T) {
	h := testServer(t).Handler()
	if code, _ := do(t, h, "POST", "/v1/jobs", map[string]any{"name": "x"}); code != 400 {
		t.Fatalf("invalid job should 400, got %d", code)
	}
	_, b := do(t, h, "POST", "/v1/jobs", trainSpec)
	id := decode(t, b).ID
	if code, _ := do(t, h, "DELETE", "/v1/jobs/"+id, nil); code != 200 {
		t.Fatalf("cancel = %d", code)
	}
	if code, _ := do(t, h, "DELETE", "/v1/jobs/"+id, nil); code != 409 {
		t.Fatalf("second cancel should 409, got %d", code)
	}
	if code, _ := do(t, h, "GET", "/v1/jobs/nope", nil); code != 404 {
		t.Fatalf("missing job should 404, got %d", code)
	}
}
