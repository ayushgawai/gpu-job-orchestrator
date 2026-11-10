// Package cluster models simulated GPU pools and nodes across clouds and on-prem.
package cluster

import "fmt"

// Capacity is the purchasing model of a pool.
type Capacity string

const (
	Spot     Capacity = "spot"
	OnDemand Capacity = "on-demand"
)

// PoolSpec describes a homogeneous group of GPU nodes.
type PoolSpec struct {
	Name            string   `json:"name"`
	Provider        string   `json:"provider"` // aws | gcp | onprem
	Region          string   `json:"region"`
	GPUType         string   `json:"gpu_type"`
	Capacity        Capacity `json:"capacity"`
	PricePerGPUHour float64  `json:"price_per_gpu_hour"`
	GPUsPerNode     int      `json:"gpus_per_node"`
	Nodes           int      `json:"nodes"`
	// MeanHoursToPreempt is the mean time between spot reclaims per node.
	// Ignored for on-demand pools.
	MeanHoursToPreempt float64 `json:"mean_hours_to_preempt,omitempty"`
}

// Pool is a runtime pool with its nodes.
type Pool struct {
	PoolSpec
	Index int
	Nodes []*Node
}

// Node is one machine with a fixed number of GPUs.
type Node struct {
	ID    string
	Pool  *Pool
	Index int // global, stable ordering used by first-fit
	Total int
	Free  int
	Up    bool
	Jobs  map[string]int // job ID -> GPUs held on this node
}

// Cluster is the full set of pools.
type Cluster struct {
	Pools []*Pool
	Nodes []*Node
	byID  map[string]*Node
}

// DefaultTopology returns 512 GPUs across AWS, GCP and on-prem.
// Prices are illustrative per-GPU-hour figures, not quotes.
func DefaultTopology() []PoolSpec {
	return []PoolSpec{
		{Name: "aws-use1-h100-od", Provider: "aws", Region: "us-east-1", GPUType: "H100", Capacity: OnDemand, PricePerGPUHour: 6.90, GPUsPerNode: 8, Nodes: 4},
		{Name: "aws-use1-h100-spot", Provider: "aws", Region: "us-east-1", GPUType: "H100", Capacity: Spot, PricePerGPUHour: 2.75, GPUsPerNode: 8, Nodes: 6, MeanHoursToPreempt: 18},
		{Name: "aws-usw2-a10g-od", Provider: "aws", Region: "us-west-2", GPUType: "A10G", Capacity: OnDemand, PricePerGPUHour: 1.45, GPUsPerNode: 4, Nodes: 8},
		{Name: "aws-usw2-a10g-spot", Provider: "aws", Region: "us-west-2", GPUType: "A10G", Capacity: Spot, PricePerGPUHour: 0.55, GPUsPerNode: 4, Nodes: 16, MeanHoursToPreempt: 30},
		{Name: "gcp-usc1-a100-od", Provider: "gcp", Region: "us-central1", GPUType: "A100", Capacity: OnDemand, PricePerGPUHour: 3.70, GPUsPerNode: 8, Nodes: 6},
		{Name: "gcp-usc1-a100-spot", Provider: "gcp", Region: "us-central1", GPUType: "A100", Capacity: Spot, PricePerGPUHour: 1.45, GPUsPerNode: 8, Nodes: 8, MeanHoursToPreempt: 12},
		{Name: "gcp-euw4-l4-spot", Provider: "gcp", Region: "europe-west4", GPUType: "L4", Capacity: Spot, PricePerGPUHour: 0.30, GPUsPerNode: 8, Nodes: 8, MeanHoursToPreempt: 24},
		{Name: "onprem-dc1-a100", Provider: "onprem", Region: "dc1", GPUType: "A100", Capacity: OnDemand, PricePerGPUHour: 1.10, GPUsPerNode: 8, Nodes: 12},
		{Name: "onprem-dc1-h100", Provider: "onprem", Region: "dc1", GPUType: "H100", Capacity: OnDemand, PricePerGPUHour: 2.40, GPUsPerNode: 8, Nodes: 8},
	}
}

// New builds a cluster from pool specs. Pool names must be unique.
func New(specs []PoolSpec) (*Cluster, error) {
	c := &Cluster{byID: map[string]*Node{}}
	seen := map[string]bool{}
	for pi, s := range specs {
		if s.Name == "" || seen[s.Name] {
			return nil, fmt.Errorf("pool %d: empty or duplicate name %q", pi, s.Name)
		}
		if s.GPUsPerNode <= 0 || s.Nodes <= 0 {
			return nil, fmt.Errorf("pool %s: gpus_per_node and nodes must be > 0", s.Name)
		}
		if s.Capacity != Spot && s.Capacity != OnDemand {
			return nil, fmt.Errorf("pool %s: unknown capacity %q", s.Name, s.Capacity)
		}
		seen[s.Name] = true
		p := &Pool{PoolSpec: s, Index: pi}
		for i := 0; i < s.Nodes; i++ {
			n := &Node{
				ID:    fmt.Sprintf("%s-n%02d", s.Name, i),
				Pool:  p,
				Index: len(c.Nodes),
				Total: s.GPUsPerNode,
				Free:  s.GPUsPerNode,
				Up:    true,
				Jobs:  map[string]int{},
			}
			p.Nodes = append(p.Nodes, n)
			c.Nodes = append(c.Nodes, n)
			c.byID[n.ID] = n
		}
		c.Pools = append(c.Pools, p)
	}
	return c, nil
}

// MustDefault returns the default 512-GPU cluster.
func MustDefault() *Cluster {
	c, err := New(DefaultTopology())
	if err != nil {
		panic(err)
	}
	return c
}

// Node returns a node by ID.
func (c *Cluster) Node(id string) (*Node, bool) {
	n, ok := c.byID[id]
	return n, ok
}

// TotalGPUs counts every GPU, including those on nodes that are down.
func (c *Cluster) TotalGPUs() int {
	t := 0
	for _, n := range c.Nodes {
		t += n.Total
	}
	return t
}

// FreeGPUs counts free GPUs on nodes that are up.
func (c *Cluster) FreeGPUs() int {
	f := 0
	for _, n := range c.Nodes {
		if n.Up {
			f += n.Free
		}
	}
	return f
}

// Allocated counts GPUs held by jobs.
func (c *Cluster) Allocated() int {
	a := 0
	for _, n := range c.Nodes {
		a += n.Total - n.Free
	}
	return a
}

// PoolGPUs returns (total, allocated, up) GPUs for a pool.
func (p *Pool) PoolGPUs() (total, allocated, up int) {
	for _, n := range p.Nodes {
		total += n.Total
		allocated += n.Total - n.Free
		if n.Up {
			up += n.Total
		}
	}
	return
}
