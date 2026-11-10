// Control plane entrypoint: REST API over the simulated 512-GPU cluster.
package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/ayushgawai/gpu-job-orchestrator/internal/api"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/cluster"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/sim"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	minute := flag.Duration("minute", time.Second, "wall time per simulated job-minute")
	flag.Parse()
	s := api.New(cluster.MustDefault(), sim.Quotas())
	go func() {
		for range time.Tick(*minute) {
			s.Advance(1)
		}
	}()
	log.Printf("gpu-job-orchestrator listening on %s (512 simulated GPUs, 1 job-minute = %s)", *addr, *minute)
	log.Fatal(http.ListenAndServe(*addr, s.Handler()))
}
