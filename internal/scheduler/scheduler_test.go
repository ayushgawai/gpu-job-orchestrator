package scheduler

import (
	"testing"

	"github.com/ayushgawai/gpu-job-orchestrator/internal/cluster"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/job"
)

func smallCluster(t *testing.T) *cluster.Cluster {
	t.Helper()
	c, err := cluster.New([]cluster.PoolSpec{
		{Name: "od", Provider: "aws", GPUType: "H100", Capacity: cluster.OnDemand, PricePerGPUHour: 5, GPUsPerNode: 8, Nodes: 2},
		{Name: "spot", Provider: "aws", GPUType: "H100", Capacity: cluster.Spot, PricePerGPUHour: 1, GPUsPerNode: 8, Nodes: 2, MeanHoursToPreempt: 10},
	})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func mkJob(id string, gpus int, allowSpot bool) *job.Job {
	return &job.Job{ID: id, Spec: job.Spec{Name: id, Tenant: "t1", GPUs: gpus, GPUType: "H100", DurationMin: 60, AllowSpot: allowSpot}}
}

func TestGangAllOrNothing(t *testing.T) {
	s := New(smallCluster(t), BestFit, nil)
	if _, err := s.Schedule(mkJob("big", 40, true)); err == nil {
		t.Fatal("gang larger than any pool must not be split across pools")
	}
	if got := s.Cluster.FreeGPUs(); got != 32 {
		t.Fatalf("failed schedule leaked GPUs: free=%d want 32", got)
	}
}

func TestBestFitPacksTightestNode(t *testing.T) {
	c := smallCluster(t)
	s := New(c, BestFit, nil)
	// Occupy 6 of 8 GPUs on first spot node indirectly via a 6-GPU job, then
	// a 2-GPU job should land on that same node (tightest fit), not a fresh one.
	a, _ := s.Schedule(mkJob("a", 6, true))
	b, err := s.Schedule(mkJob("b", 2, true))
	if err != nil {
		t.Fatal(err)
	}
	if b[0].NodeID != a[0].NodeID {
		t.Fatalf("best-fit should pack onto %s, got %s", a[0].NodeID, b[0].NodeID)
	}
}

func TestSpotPreferredThenFallback(t *testing.T) {
	s := New(smallCluster(t), BestFit, nil)
	j1 := mkJob("s1", 16, true) // fills all spot
	a, err := s.Schedule(j1)
	if err != nil {
		t.Fatal(err)
	}
	if !a[0].Spot {
		t.Fatal("spot-allowed job should prefer spot")
	}
	a2, err := s.Schedule(mkJob("s2", 8, true)) // spot full -> on-demand fallback
	if err != nil {
		t.Fatal(err)
	}
	if a2[0].Spot {
		t.Fatal("expected on-demand fallback when spot is full")
	}
	if _, err := s.Schedule(mkJob("s3", 8, false)); err != nil {
		t.Fatalf("on-demand-only job should still place: %v", err)
	}
}

func TestQuotaEnforced(t *testing.T) {
	s := New(smallCluster(t), BestFit, map[string]int{"t1": 8})
	if _, err := s.Schedule(mkJob("q1", 8, false)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Schedule(mkJob("q2", 1, false)); err == nil {
		t.Fatal("quota must block second job")
	}
	if got := s.TenantUsed("t1"); got != 8 {
		t.Fatalf("rejected job must not count against quota: used=%d", got)
	}
}

func TestResumeMovesOffPreemptedPool(t *testing.T) {
	s := New(smallCluster(t), BestFit, nil)
	j := mkJob("r1", 8, true)
	a, err := s.Schedule(j)
	if err != nil {
		t.Fatal(err)
	}
	if a[0].Pool != "spot" {
		t.Fatalf("expected spot, got %s", a[0].Pool)
	}
	for _, id := range s.PreemptNode(a[0].NodeID) {
		if id != j.ID {
			t.Fatalf("unexpected job %s", id)
		}
		j.Preempt(a[0].Pool)
		s.Release(j)
	}
	// spot still has a healthy node, but the resumed job should leave the pool
	b, err := s.Schedule(j)
	if err != nil {
		t.Fatal(err)
	}
	if b[0].Pool != "od" {
		t.Fatalf("resume should move to another pool, got %s", b[0].Pool)
	}
}

func TestResumeFallsBackToSamePoolWhenOnlyFit(t *testing.T) {
	s := New(smallCluster(t), BestFit, nil)
	j := mkJob("r2", 8, true)
	j.AvoidPool = "spot"
	if _, err := s.Schedule(mkJob("fill", 16, false)); err != nil { // fill on-demand
		t.Fatal(err)
	}
	a, err := s.Schedule(j)
	if err != nil || a[0].Pool != "spot" {
		t.Fatalf("avoided pool must still be used when nothing else fits: %v %v", a, err)
	}
}

func TestRestoreUpNodeIsNoop(t *testing.T) {
	s := New(smallCluster(t), BestFit, nil)
	a, _ := s.Schedule(mkJob("x", 8, true))
	s.RestoreNode(a[0].NodeID)
	if n, _ := s.Cluster.Node(a[0].NodeID); n.Free != 0 {
		t.Fatalf("restore of an up node wiped its allocation: free=%d", n.Free)
	}
}

func TestQuotaReleasedFreesCapacity(t *testing.T) {
	s := New(smallCluster(t), BestFit, map[string]int{"t1": 8})
	j := mkJob("q1", 8, false)
	if _, err := s.Schedule(j); err != nil {
		t.Fatal(err)
	}
	s.Release(j)
	if _, err := s.Schedule(mkJob("q2", 8, false)); err != nil {
		t.Fatalf("after release quota should allow new job: %v", err)
	}
}

func TestPriorityOrdering(t *testing.T) {
	lo := &job.Job{ID: "lo", Spec: job.Spec{Priority: 1}, SubmittedAtMin: 0}
	hi := &job.Job{ID: "hi", Spec: job.Spec{Priority: 9}, SubmittedAtMin: 10}
	q := []*job.Job{lo, hi}
	SortQueue(q)
	if q[0].ID != "hi" {
		t.Fatal("higher priority must schedule first")
	}
}

func TestFirstFitUsesTopologyOrder(t *testing.T) {
	s := New(smallCluster(t), FirstFit, nil)
	a, err := s.Schedule(mkJob("f1", 4, true))
	if err != nil {
		t.Fatal(err)
	}
	if a[0].Pool != "od" { // od is first in topology even though spot is cheaper
		t.Fatalf("first-fit should take first pool in order, got %s", a[0].Pool)
	}
}
