// Package sim replays a workload against a scheduler in 1-minute ticks.
//
// Preemption model: each up spot node is reclaimed with probability
// 1/(mean_hours*60) per tick, stays down 5 ticks, then returns empty.
// A preempted job loses work since its last checkpoint (counted as lost
// GPU-minutes) and requeues; when placed again it resumes from the
// checkpoint, not from zero, preferring a different pool.
package sim

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/ayushgawai/gpu-job-orchestrator/internal/cluster"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/job"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/scheduler"
)

type Result struct {
	Mode             string
	Jobs             int
	Completed        int
	Minutes          int     // simulated minutes until the last job finished
	Utilization      float64 // allocated GPU-min / total GPU-min over the run
	WaitP50Min       float64 // submit -> first start
	WaitP95Min       float64
	Preemptions      int
	Recoveries       int // restarts from checkpoint after a preemption
	CrossPool        int // recoveries that landed on a different pool
	LostGPUMin       float64
	AvgUSDPerGPUHour float64 // spend / GPU-hours used
}

var tenants = []string{"alpha", "beta", "gamma", "delta"}

// Quotas are the per-tenant GPU caps used by the simulator and the server.
func Quotas() map[string]int {
	return map[string]int{"alpha": 160, "beta": 160, "gamma": 128, "delta": 128}
}

// Generate builds a deterministic mixed training/inference workload.
func Generate(seed int64, n int) []*job.Job {
	r := rand.New(rand.NewSource(seed))
	gpuTypes := []string{"H100", "A100", "A10G", "L4", ""}
	sizes := []int{1, 2, 4, 4, 8, 8, 16}
	jobs := make([]*job.Job, 0, n)
	arrival := 0.0
	for i := 0; i < n; i++ {
		arrival += r.ExpFloat64() * 3 // ~20 jobs/hour, ~50 simulated hours for 1,000 jobs
		gt := gpuTypes[r.Intn(len(gpuTypes))]
		spec := job.Spec{
			Name: fmt.Sprintf("job-%04d", i), Tenant: tenants[i%len(tenants)],
			GPUs: sizes[r.Intn(len(sizes))], GPUType: gt,
			Priority: 1 + r.Intn(10), DurationMin: 10 + r.Intn(230),
			CheckpointEveryMin: 10, AllowSpot: r.Float64() < 0.7,
		}
		if gt == "L4" {
			spec.AllowSpot = true // L4 exists only as spot in the default topology
		}
		jobs = append(jobs, &job.Job{ID: spec.Name, Spec: spec, Status: job.Queued, SubmittedAtMin: arrival})
	}
	return jobs
}

type downNode struct {
	id    string
	until int
}

func Run(mode scheduler.Mode, seed int64, n int) Result {
	jobs := Generate(seed, n)
	cl := cluster.MustDefault()
	s := scheduler.New(cl, mode, Quotas())
	price := map[string]float64{}
	for _, p := range cl.Pools {
		price[p.Name] = p.PricePerGPUHour
	}
	res := Result{Mode: modeName(mode), Jobs: n}

	var queue, running []*job.Job
	var waits []float64
	var downs []downNode
	var allocMin, totalMin, costMin float64
	r := rand.New(rand.NewSource(seed + 99))
	next := 0
	const maxTick = 60 * 24 * 30 // 30 simulated days, hard stop

	tick := 0
	for ; tick < maxTick && res.Completed < n; tick++ {
		now := float64(tick)
		for next < n && jobs[next].SubmittedAtMin <= now {
			queue = append(queue, jobs[next])
			next++
		}
		still := downs[:0]
		for _, d := range downs {
			if tick >= d.until {
				s.RestoreNode(d.id)
			} else {
				still = append(still, d)
			}
		}
		downs = still

		// spot preemptions: roll back to checkpoint, free the whole gang, requeue
		byID := map[string]*job.Job{}
		for _, j := range running {
			byID[j.ID] = j
		}
		for _, nd := range cl.Nodes {
			if !nd.Up || nd.Pool.Capacity != cluster.Spot || nd.Pool.MeanHoursToPreempt <= 0 {
				continue
			}
			if r.Float64() >= 1.0/(nd.Pool.MeanHoursToPreempt*60) {
				continue
			}
			downs = append(downs, downNode{nd.ID, tick + 5})
			for _, id := range s.PreemptNode(nd.ID) {
				if j := byID[id]; j != nil && j.Status == job.Running {
					res.LostGPUMin += j.Preempt(nd.Pool.Name)
					s.Release(j)
					queue = append(queue, j)
					res.Preemptions++
				}
			}
		}
		alive := running[:0]
		for _, j := range running {
			if j.Status == job.Running {
				alive = append(alive, j)
			}
		}
		running = alive

		scheduler.SortQueue(queue)
		waiting := queue[:0]
		for _, j := range queue {
			allocs, err := s.Schedule(j)
			if err != nil {
				waiting = append(waiting, j)
				continue
			}
			if j.Preemptions == 0 {
				waits = append(waits, now-j.SubmittedAtMin)
			} else {
				res.Recoveries++
				j.Resumes++
				if allocs[0].Pool != j.AvoidPool {
					res.CrossPool++
				}
			}
			j.Placement = allocs
			j.Status = job.Running
			running = append(running, j)
		}
		queue = waiting

		stillRunning := running[:0]
		for _, j := range running {
			allocMin += float64(j.Spec.GPUs)
			for _, a := range j.Placement {
				costMin += float64(a.GPUs) * price[a.Pool]
			}
			if j.Advance(1) {
				j.Status = job.Succeeded
				s.Release(j)
				res.Completed++
			} else {
				stillRunning = append(stillRunning, j)
			}
		}
		running = stillRunning
		totalMin += float64(cl.TotalGPUs())
	}
	res.Minutes = tick
	if totalMin > 0 {
		res.Utilization = allocMin / totalMin
	}
	if allocMin > 0 {
		res.AvgUSDPerGPUHour = costMin / allocMin
	}
	res.WaitP50Min = pct(waits, 50)
	res.WaitP95Min = pct(waits, 95)
	return res
}

func modeName(m scheduler.Mode) string {
	if m == scheduler.FirstFit {
		return "first-fit"
	}
	return "best-fit"
}

func pct(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	s := append([]float64{}, vals...)
	sort.Float64s(s)
	i := int(p / 100 * float64(len(s)))
	if i >= len(s) {
		i = len(s) - 1
	}
	return s[i]
}

// Compare runs both schedulers on the same workload.
func Compare(seed int64, n int) (best, first Result) {
	return Run(scheduler.BestFit, seed, n), Run(scheduler.FirstFit, seed, n)
}
