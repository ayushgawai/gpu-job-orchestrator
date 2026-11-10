// Package scheduler places gang jobs onto a cluster.
//
// Best-fit: among pools that can host the whole gang, prefer spot (when
// allowed) then cheapest price, and pack onto the tightest-fitting nodes.
// FirstFit is the baseline: pools/nodes in topology order, first that fits.
// ponytail: O(pools x nodes) scan per job; add indexed free-lists if the
// cluster grows past a few thousand nodes.
package scheduler

import (
	"fmt"
	"sort"

	"github.com/ayushgawai/gpu-job-orchestrator/internal/cluster"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/job"
)

type Mode int

const (
	BestFit Mode = iota
	FirstFit
)

type Scheduler struct {
	Cluster *cluster.Cluster
	Mode    Mode
	Quotas  map[string]int // tenant -> max GPUs; missing = unlimited
	used    map[string]int // tenant -> GPUs currently placed
	placed  map[string][]job.Alloc
}

func New(c *cluster.Cluster, mode Mode, quotas map[string]int) *Scheduler {
	if quotas == nil {
		quotas = map[string]int{}
	}
	return &Scheduler{Cluster: c, Mode: mode, Quotas: quotas, used: map[string]int{}, placed: map[string][]job.Alloc{}}
}

func (s *Scheduler) TenantUsed(tenant string) int { return s.used[tenant] }

// Schedule tries to place the whole gang. On error nothing is allocated.
func (s *Scheduler) Schedule(j *job.Job) ([]job.Alloc, error) {
	if allocs, ok := s.placed[j.ID]; ok {
		return allocs, nil // idempotent re-schedule
	}
	if q, ok := s.Quotas[j.Spec.Tenant]; ok && s.used[j.Spec.Tenant]+j.Spec.GPUs > q {
		return nil, fmt.Errorf("tenant %s quota exceeded: %d+%d > %d", j.Spec.Tenant, s.used[j.Spec.Tenant], j.Spec.GPUs, q)
	}
	for _, p := range s.candidatePools(j) {
		allocs, ok := s.placeInPool(p, j)
		if !ok {
			continue
		}
		for _, a := range allocs {
			n, _ := s.Cluster.Node(a.NodeID)
			n.Free -= a.GPUs
			n.Jobs[j.ID] = a.GPUs
		}
		s.placed[j.ID] = allocs
		s.used[j.Spec.Tenant] += j.Spec.GPUs
		return allocs, nil
	}
	return nil, fmt.Errorf("no pool can host gang of %d x %s", j.Spec.GPUs, orAny(j.Spec.GPUType))
}

func orAny(s string) string {
	if s == "" {
		return "any GPU"
	}
	return s
}

func (s *Scheduler) candidatePools(j *job.Job) []*cluster.Pool {
	var pools []*cluster.Pool
	for _, p := range s.Cluster.Pools {
		if j.Spec.GPUType != "" && p.GPUType != j.Spec.GPUType {
			continue
		}
		if p.Capacity == cluster.Spot && !j.Spec.AllowSpot {
			continue
		}
		pools = append(pools, p) // Cluster.Pools is already topology order (first-fit)
	}
	if s.Mode == BestFit {
		// spot fallback order: spot first (cheap capacity), then on-demand;
		// within each class, cheapest first.
		sort.SliceStable(pools, func(a, b int) bool {
			if (pools[a].Capacity == cluster.Spot) != (pools[b].Capacity == cluster.Spot) {
				return pools[a].Capacity == cluster.Spot
			}
			return pools[a].PricePerGPUHour < pools[b].PricePerGPUHour
		})
	}
	// A resumed job tries the pool that just preempted it last, so it moves
	// to another pool when one fits but never starves if that pool is the only fit.
	sort.SliceStable(pools, func(a, b int) bool {
		return pools[a].Name != j.AvoidPool && pools[b].Name == j.AvoidPool
	})
	return pools
}

// placeInPool returns a gang placement inside one pool, or false.
// Best-fit picks tightest nodes first; first-fit takes topology order.
func (s *Scheduler) placeInPool(p *cluster.Pool, j *job.Job) ([]job.Alloc, bool) {
	nodes := make([]*cluster.Node, 0, len(p.Nodes))
	free := 0
	for _, n := range p.Nodes {
		if n.Up && n.Free > 0 {
			nodes = append(nodes, n)
			free += n.Free
		}
	}
	if free < j.Spec.GPUs {
		return nil, false // gang: never split across pools
	}
	if s.Mode == BestFit {
		sort.SliceStable(nodes, func(a, b int) bool { return nodes[a].Free < nodes[b].Free })
	}
	remaining := j.Spec.GPUs
	var allocs []job.Alloc
	for _, n := range nodes {
		if remaining == 0 {
			break
		}
		take := min(n.Free, remaining)
		allocs = append(allocs, job.Alloc{NodeID: n.ID, Pool: p.Name, GPUs: take, Spot: p.Capacity == cluster.Spot})
		remaining -= take
	}
	return allocs, true
}

// Release frees a job's placement. Safe to call twice.
func (s *Scheduler) Release(j *job.Job) {
	allocs, ok := s.placed[j.ID]
	if !ok {
		return
	}
	for _, a := range allocs {
		if n, ok := s.Cluster.Node(a.NodeID); ok {
			n.Free = min(n.Free+a.GPUs, n.Total)
			delete(n.Jobs, j.ID)
		}
	}
	s.used[j.Spec.Tenant] -= j.Spec.GPUs
	delete(s.placed, j.ID)
	j.Placement = nil
}

// PreemptNode marks a node down and returns IDs of jobs that lost GPUs on it.
// Callers call Job.Preempt and Release for each, which frees the rest of the gang.
func (s *Scheduler) PreemptNode(nodeID string) []string {
	n, ok := s.Cluster.Node(nodeID)
	if !ok || !n.Up {
		return nil
	}
	n.Up = false
	var ids []string
	for id := range n.Jobs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// RestoreNode brings a down node back with all GPUs free. No-op on up nodes.
func (s *Scheduler) RestoreNode(nodeID string) {
	if n, ok := s.Cluster.Node(nodeID); ok && !n.Up {
		n.Up = true
		n.Free = n.Total
		n.Jobs = map[string]int{}
	}
}

// SortQueue orders queued jobs: priority desc, then submission order.
// ponytail: no aging; a long low-priority job can starve. Add wait-based
// aging if starvation shows up in sim output.
func SortQueue(q []*job.Job) {
	sort.SliceStable(q, func(a, b int) bool {
		if q[a].Spec.Priority != q[b].Spec.Priority {
			return q[a].Spec.Priority > q[b].Spec.Priority
		}
		return q[a].SubmittedAtMin < q[b].SubmittedAtMin
	})
}
