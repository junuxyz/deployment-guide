### Exp 3. 4P+4D disaggregated

This recipe is the 4P/4D counterpart to [Exp 2](../02-disagg-routing/README.md).
Both experiments use the same Qwen3-32B-FP8 model revision, Dynamo and vLLM
runtime, round-robin frontend, TP=2 workers, NIXL/RDMA transfer settings,
resource requests, cache mounts, and fixed-schedule AIPerf workload.

The serving-topology change is limited to the worker replica counts:

| Worker role | Exp 2 | Exp 3 |
|---|---:|---:|
| Prefill | 6 | 4 |
| Decode | 2 | 4 |

Resource names and artifact IDs are unique to Exp 3 so that its deployment and
benchmark outputs cannot collide with Exp 2.

## Files

| File | Purpose |
|---|---|
| [`deploy.yaml`](./deploy.yaml) | 4P/4D TP=2 `DynamoGraphDeployment` |
| [`perf.yaml`](./perf.yaml) | Mooncake FAST25 fixed-schedule AIPerf Job |

## Shared benchmark settings

- Qwen3-32B-FP8 revision `aa55da1`
- 16 H100 GPUs total
- 4 prefill workers and 4 decode workers, TP=2 each
- round-robin routing
- 40,960-token maximum context length
- 0.90 GPU-memory utilization
- 128-token KV-cache blocks
- TTFT/ITL goodput thresholds of 2,000 ms / 25 ms
- CPU KV-cache offloading disabled

Use the [parent vLLM experiment workflow](../README.md#how-to-run-the-experiments)
with `RECIPE=03-disagg-routing-4p4d` to validate, deploy, benchmark, and clean up
this recipe.
