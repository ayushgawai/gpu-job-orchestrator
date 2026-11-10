// loadtest hammers the API and reports latency percentiles (p99 included).
// With no -base it starts the server in-process on a loopback port.
// Mix: 1/3 health checks, 1/3 submits, 1/6 idempotent replays, 1/6 pool reads.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"sync"
	"time"

	"github.com/ayushgawai/gpu-job-orchestrator/internal/api"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/cluster"
	"github.com/ayushgawai/gpu-job-orchestrator/internal/sim"
)

func main() {
	base := flag.String("base", "", "server base URL (empty = in-process server)")
	n := flag.Int("n", 5000, "requests")
	conc := flag.Int("c", 32, "concurrency")
	flag.Parse()
	if *base == "" {
		ts := httptest.NewServer(api.New(cluster.MustDefault(), sim.Quotas()).Handler())
		defer ts.Close()
		*base = ts.URL
	}

	lat := make([]float64, 0, *n)
	errs := 0
	var mu sync.Mutex
	sem := make(chan struct{}, *conc)
	var wg sync.WaitGroup
	client := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{MaxIdleConnsPerHost: *conc}}
	start := time.Now()
	for i := 0; i < *n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()
			var req *http.Request
			switch i % 6 {
			case 0, 3:
				req, _ = http.NewRequest("GET", *base+"/healthz", nil)
			case 1, 4:
				body := fmt.Sprintf(`{"name":"lt-%d","tenant":"alpha","gpus":1,"duration_min":5}`, i)
				req, _ = http.NewRequest("POST", *base+"/v1/jobs", bytes.NewBufferString(body))
				req.Header.Set("Idempotency-Key", fmt.Sprintf("lt-%d", i))
			case 2:
				body := fmt.Sprintf(`{"name":"lt-%d","tenant":"alpha","gpus":1,"duration_min":5}`, i-1)
				req, _ = http.NewRequest("POST", *base+"/v1/jobs", bytes.NewBufferString(body))
				req.Header.Set("Idempotency-Key", fmt.Sprintf("lt-%d", i-1)) // replay
			default:
				req, _ = http.NewRequest("GET", *base+"/v1/pools", nil)
			}
			t0 := time.Now()
			resp, err := client.Do(req)
			ok := err == nil
			if ok {
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				ok = resp.StatusCode < 400
			}
			ms := float64(time.Since(t0).Microseconds()) / 1000
			mu.Lock()
			defer mu.Unlock()
			if !ok {
				errs++
				return
			}
			lat = append(lat, ms)
		}(i)
	}
	wg.Wait()
	elapsed := time.Since(start)
	sort.Float64s(lat)
	p := func(q float64) float64 {
		if len(lat) == 0 {
			return 0
		}
		return lat[min(int(q*float64(len(lat))), len(lat)-1)]
	}
	fmt.Printf("requests=%d errors=%d concurrency=%d elapsed=%s throughput=%.0f req/s\n",
		*n, errs, *conc, elapsed.Round(time.Millisecond), float64(*n)/elapsed.Seconds())
	fmt.Printf("latency ms: p50=%.2f p95=%.2f p99=%.2f max=%.2f\n", p(.50), p(.95), p(.99), p(1))
}
