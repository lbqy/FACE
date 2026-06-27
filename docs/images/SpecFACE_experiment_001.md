# SpecFACE Micro Experiment 001

This single-instance analytical experiment uses the lightweight OME search from `docs/images/SpecFACE_plan.md` and the FACE WSC baseline configuration. The instance model is corrected to use core-group granularity: the OME searches integer workflow partitions over `2x2` dies, each die has `2x4` core groups, for `32` groups total and `16` cores per group.

The mapping figure renders one cell per core group. The timeline figure uses time on the x-axis and one row per workflow so pipeline busy/idle behavior is visible directly.

Output directory: `/home/lbqy/astra-sim/examples/llm_serving/specface/outputs`
Plot directory: `/home/lbqy/astra-sim/docs/images/specface_experiment_001`

## Default-alpha Snapshot

Default acceptance rate: `0.8`

| mode | gamma | fallback | ns/token | speedup vs FACE | partition TP/TV/DP/DD (groups) | bottleneck |
|---|---:|---:|---:|---:|---|---|
| FACE | 0 | 0 | 73001.9 | 1.000 | 32/0/0/0 | hbm |
| SpecFACE-fixed-partition | 1 | 0 | 60886.3 | 1.199 | 13/9/4/6 | hbm |
| SpecFACE-full | 1 | 0 | 60886.3 | 1.199 | 9/9/3/11 | hbm |
| SpecFACE-no-fallback | 1 | 0 | 60886.3 | 1.199 | 9/9/3/11 | hbm |
| SpecFACE-no-journal | 1 | 0 | 60889.0 | 1.199 | 9/9/3/11 | hbm |
| SpecFACE-static-gamma | 1 | 0 | 60886.3 | 1.199 | 9/9/3/11 | hbm |
| SpecFACE-static-gamma | 2 | 0 | 61772.7 | 1.182 | 11/12/2/7 | target_compute |
| SpecFACE-static-gamma | 4 | 1 | 73001.9 | 1.000 | 9/7/5/11 | target_compute |
| SpecFACE-static-gamma | 8 | 1 | 73001.9 | 1.000 | 9/7/5/11 | target_compute |
| SpecFACE-static-gamma | 12 | 1 | 73001.9 | 1.000 | 9/7/5/11 | target_compute |
| SpecFACE-static-gamma | 16 | 1 | 73001.9 | 1.000 | 9/7/5/11 | target_compute |

## Figures

### Workflow Mapping Layouts

![Workflow Mapping Layouts](specface_experiment_001/workflow_mapping_layouts.png)

### Workflow Timelines

![Workflow Timelines](specface_experiment_001/workflow_timelines.png)

### Workflow Partitions

![Workflow Partitions](specface_experiment_001/workflow_partitions.png)

### Online Overlap

![Online Overlap](specface_experiment_001/online_overlap.png)

### Resource Breakdown

![Resource Breakdown](specface_experiment_001/resource_breakdown.png)

### Workload Speedup

![Workload Speedup](specface_experiment_001/workload_speedup.png)

## Workload-Class Speedups

| workload | prompt | output | context | gamma | fallback | TPOT speedup | E2E speedup | bottleneck |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| trace_mixed | 747 | 41 | 747 | 1 | 0 | 1.199 | 0.315 | hbm |
| short_prompt | 256 | 64 | 256 | 1 | 1 | 1.000 | 1.000 | target_compute |
| long_prompt | 2048 | 64 | 2048 | 4 | 0 | 1.491 | 0.334 | hbm |
| long_decode | 512 | 256 | 512 | 1 | 0 | 1.199 | 0.539 | hbm |
| balanced_long | 1024 | 128 | 1024 | 2 | 0 | 1.391 | 0.460 | hbm |

## Notes

At alpha=0.8, the best full SpecFACE point selects gamma=1 and estimates 1.199x speedup over FACE decode ns/token. Journal peak is 0.52 MB against 46.20 MB available in the draft decode island.
These numbers are not cycle-accurate; they are meant to drive the first microarchitecture sweep and identify useful regions for the simulator implementation.
`specface_core_group_mapping.csv` contains one row per workflow/core-group cell with die coordinates, group coordinates, assigned workflow, and cores per group.
