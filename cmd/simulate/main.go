// simulate replays ~1,000 mixed jobs against best-fit and first-fit.
package main

import (
	"flag"
	"fmt"

	"github.com/ayushgawai/gpu-job-orchestrator/internal/sim"
)

func main() {
	seed := flag.Int64("seed", 42, "workload seed")
	n := flag.Int("jobs", 1000, "number of jobs")
	gap := flag.Float64("gap", 3, "mean minutes between job arrivals (lower = more contention)")
	flag.Parse()
	best, first := sim.Compare(*seed, *n, *gap)
	fmt.Printf("workload: %d jobs, seed %d, mean arrival gap %.1f min, 512 simulated GPUs, 1-min ticks\n\n", *n, *seed, *gap)
	fmt.Printf("%-10s %5s %8s %6s %9s %9s %8s %9s %10s %9s %7s\n",
		"scheduler", "done", "makespan", "util", "waitP50m", "waitP95m", "preempt", "recovered", "crossPool", "lostGPUm", "$/GPUh")
	for _, r := range []sim.Result{best, first} {
		fmt.Printf("%-10s %5d %7.1fh %5.1f%% %9.0f %9.0f %8d %9d %10d %9.0f %7.2f\n",
			r.Mode, r.Completed, float64(r.Minutes)/60, r.Utilization*100, r.WaitP50Min, r.WaitP95Min,
			r.Preemptions, r.Recoveries, r.CrossPool, r.LostGPUMin, r.AvgUSDPerGPUHour)
	}
	fmt.Printf("\nbest-fit vs first-fit: utilization %+.1f pts, wait p50 %+.0f min, wait p95 %+.0f min, $/GPU-hour %+.2f\n",
		(best.Utilization-first.Utilization)*100, best.WaitP50Min-first.WaitP50Min,
		best.WaitP95Min-first.WaitP95Min, best.AvgUSDPerGPUHour-first.AvgUSDPerGPUHour)
}
