// Package job defines the job model shared by scheduler, API and simulator.
package job

import (
	"fmt"
	"math"
)

type Status string

const (
	Queued    Status = "queued"
	Running   Status = "running"
	Succeeded Status = "succeeded"
	Cancelled Status = "cancelled"
)

type Spec struct {
	Name               string `json:"name"`
	Tenant             string `json:"tenant"`
	GPUs               int    `json:"gpus"`               // gang size: all-or-nothing
	GPUType            string `json:"gpu_type,omitempty"` // empty = any
	Priority           int    `json:"priority"`           // higher wins
	DurationMin        int    `json:"duration_min"`
	CheckpointEveryMin int    `json:"checkpoint_every_min,omitempty"`
	AllowSpot          bool   `json:"allow_spot"`
	IdempotencyKey     string `json:"idempotency_key,omitempty"`
}

func (s Spec) Validate() error {
	if s.Name == "" || s.Tenant == "" {
		return fmt.Errorf("name and tenant are required")
	}
	if s.GPUs <= 0 || s.GPUs > 512 {
		return fmt.Errorf("gpus must be in 1..512")
	}
	if s.DurationMin <= 0 {
		return fmt.Errorf("duration_min must be > 0")
	}
	if s.Priority < 0 || s.Priority > 100 {
		return fmt.Errorf("priority must be in 0..100")
	}
	if s.CheckpointEveryMin < 0 {
		return fmt.Errorf("checkpoint_every_min must be >= 0")
	}
	return nil
}

// Alloc is GPUs held by a job on one node.
type Alloc struct {
	NodeID string `json:"node_id"`
	Pool   string `json:"pool"`
	GPUs   int    `json:"gpus"`
	Spot   bool   `json:"spot"`
}

type Job struct {
	ID             string  `json:"id"`
	Spec           Spec    `json:"spec"`
	Status         Status  `json:"status"`
	Placement      []Alloc `json:"placement,omitempty"`
	SubmittedAtMin float64 `json:"-"`
	ProgressMin    float64 `json:"progress_min"`   // completed work
	CheckpointMin  float64 `json:"checkpoint_min"` // last durable progress; survives preemption
	Preemptions    int     `json:"preemptions"`
	Resumes        int     `json:"resumes"`
	AvoidPool      string  `json:"-"` // pool that last preempted us; tried last on resume
}

// Advance runs the job for dt minutes, checkpointing on its interval.
// Returns true when the job has finished.
func (j *Job) Advance(dt float64) bool {
	j.ProgressMin = math.Min(j.ProgressMin+dt, float64(j.Spec.DurationMin))
	if every := float64(j.Spec.CheckpointEveryMin); every > 0 {
		j.CheckpointMin = math.Floor(j.ProgressMin/every) * every
	}
	return j.ProgressMin >= float64(j.Spec.DurationMin)
}

// Preempt rolls the job back to its last checkpoint and requeues it, steering
// the next placement away from the reclaimed pool. The caller releases GPUs.
// Returns GPU-minutes of work lost since that checkpoint.
func (j *Job) Preempt(pool string) float64 {
	lost := (j.ProgressMin - j.CheckpointMin) * float64(j.Spec.GPUs)
	j.ProgressMin = j.CheckpointMin
	j.Preemptions++
	j.Status = Queued
	j.AvoidPool = pool
	return lost
}
