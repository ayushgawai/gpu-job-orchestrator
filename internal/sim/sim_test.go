package sim

import (
	"testing"

	"github.com/ayushgawai/gpu-job-orchestrator/internal/cluster"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/job"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/scheduler"
)

// End-to-end: a spot job is preempted mid-run, loses only un-checkpointed
// work, resumes from its checkpoint on another pool, and finishes.
func TestPreemptionAndResumeE2E(t *testing.T) {
	c, err := cluster.New([]cluster.PoolSpec{
		{Name: "spot", Provider: "aws", GPUType: "H100", Capacity: cluster.Spot, PricePerGPUHour: 1, GPUsPerNode: 8, Nodes: 2, MeanHoursToPreempt: 1},
		{Name: "od", Provider: "gcp", GPUType: "H100", Capacity: cluster.OnDemand, PricePerGPUHour: 5, GPUsPerNode: 8, Nodes: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	s := scheduler.New(c, scheduler.BestFit, nil)
	j := &job.Job{ID: "train-1", Status: job.Queued, Spec: job.Spec{
		Name: "train-1", Tenant: "t1", GPUs: 4, GPUType: "H100",
		DurationMin: 30, CheckpointEveryMin: 10, AllowSpot: true,
	}}
	allocs, err := s.Schedule(j)
	if err != nil {
		t.Fatal(err)
	}
	if !allocs[0].Spot {
		t.Fatal("job should start on spot")
	}
	j.Placement, j.Status = allocs, job.Running

	for i := 0; i < 17; i++ { // checkpoint at 10, 7 minutes un-checkpointed
		j.Advance(1)
	}
	affected := s.PreemptNode(allocs[0].NodeID)
	if len(affected) != 1 || affected[0] != j.ID {
		t.Fatalf("preemption should affect train-1, got %v", affected)
	}
	if lost := j.Preempt(allocs[0].Pool); lost != 7*4 {
		t.Fatalf("lost GPU-min = %v, want 28", lost)
	}
	s.Release(j)
	if j.ProgressMin != 10 || j.Status != job.Queued {
		t.Fatalf("after preempt progress=%v status=%s, want 10/queued", j.ProgressMin, j.Status)
	}

	allocs, err = s.Schedule(j)
	if err != nil {
		t.Fatalf("resume failed: %v", err)
	}
	if allocs[0].Pool != "od" {
		t.Fatalf("resume should move to another pool, got %s", allocs[0].Pool)
	}
	j.Placement, j.Status = allocs, job.Running
	j.Resumes++
	steps := 0
	for !j.Advance(1) {
		steps++
	}
	if steps+1 != 20 {
		t.Fatalf("resumed job needed %d more minutes, want 20 (30 - checkpoint 10)", steps+1)
	}
	s.Release(j)
	if c.FreeGPUs() != 16 { // 8 on healthy spot node + 8 on-demand; preempted node still down
		t.Fatalf("free GPUs = %d, want 16", c.FreeGPUs())
	}
}

func TestSimulationCompletesWorkload(t *testing.T) {
	best, first := Compare(42, 300, 3)
	for _, r := range []Result{best, first} {
		if r.Completed != 300 {
			t.Fatalf("%s completed %d/300", r.Mode, r.Completed)
		}
		if r.Utilization <= 0 || r.Utilization > 1 {
			t.Fatalf("%s utilization out of range: %v", r.Mode, r.Utilization)
		}
		if r.Recoveries != r.Preemptions {
			t.Fatalf("%s: every preemption must recover: preempt=%d recovered=%d", r.Mode, r.Preemptions, r.Recoveries)
		}
	}
	if best.Preemptions == 0 {
		t.Fatal("expected some spot preemptions in a 300-job run")
	}
}

func TestGenerateDeterministic(t *testing.T) {
	a, b := Generate(7, 50, 3), Generate(7, 50, 3)
	for i := range a {
		if a[i].Spec != b[i].Spec || a[i].SubmittedAtMin != b[i].SubmittedAtMin {
			t.Fatalf("job %d differs between runs with same seed", i)
		}
	}
}
