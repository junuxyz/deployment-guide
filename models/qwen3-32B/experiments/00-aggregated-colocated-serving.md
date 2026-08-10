# Experiment 0: Aggregated / colocated serving with round-robin routing

This deployment is the control baseline for the four experiments. Each SGLang worker performs both prefill and decode locally, while the Dynamo frontend distributes requests across workers with round-robin routing. It does not use P/D disaggregation.

## Configuration under test

| Component | Placement | Replicas | GPUs per replica | Mode |
|---|---|---:|---:|---|
| Frontend | `gpu05` | 1 | 0 | Round-robin |
| Colocated worker | `gpu05` + `gpu06` | 8 | 2 | SGLang aggregated |

With no other GPU workloads present, the eight TP=2 workers consume all 16 GPUs: four workers on each node. Every request completes prefill and decode on the same worker.

Fixed settings: TP=2, page size 64, model revision
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`, memory fraction 0.82, and 40 GiB `/dev/shm` per worker pod.

## 1. Confirm a clean cluster

Run on `gpu05`:

```bash
export NAMESPACE=qwen32-bench

kubectl get dynamographdeployments -n "$NAMESPACE"
kubectl get pods -n "$NAMESPACE" -o wide
kubectl get nodes -L qwen.nvidia.com/role
kubectl get pvc model-cache -n "$NAMESPACE"
```

There must be no active experiment DGD and no unrelated GPU workload.

## 2. Create the deployment manifest

Save the following as
`/ephemeral/shared/qwen3-32b/manifests/00-aggregated-colocated-serving.yaml`:

**Run on `gpu05` — Kubernetes admin terminal:** create the deployment file on
the shared filesystem.

```bash
tee /ephemeral/shared/qwen3-32b/manifests/00-aggregated-colocated-serving.yaml >/dev/null <<'EOF'
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen32-agg
  namespace: qwen32-bench
spec:
  backendFramework: sglang
  env:
    - name: HF_HOME
      value: /model-store
    - name: HF_HUB_OFFLINE
      value: "1"
    - name: TRANSFORMERS_OFFLINE
      value: "1"
  components:
    - name: Frontend
      type: frontend
      replicas: 1
      podTemplate:
        spec:
          nodeSelector:
            qwen.nvidia.com/role: prefill
          tolerations:
            - key: node-role.kubernetes.io/control-plane
              operator: Exists
              effect: NoSchedule
            - key: nvidia.com/gpu
              operator: Equal
              value: "true"
              effect: NoSchedule
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
              imagePullPolicy: IfNotPresent
              env:
                - name: DYN_HTTP_PORT
                  value: "8000"
                - name: DYN_ROUTER_MODE
                  value: round-robin
              envFrom:
                - secretRef:
                    name: hf-token-secret
                    optional: true
              ports:
                - name: http
                  containerPort: 8000
              volumeMounts:
                - name: model-cache
                  mountPath: /model-store
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache

    - name: decode
      type: worker
      replicas: 8
      sharedMemorySize: 40Gi
      podTemplate:
        spec:
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                  - matchExpressions:
                      - key: qwen.nvidia.com/role
                        operator: In
                        values:
                          - prefill
                          - decode
          tolerations:
            - key: node-role.kubernetes.io/control-plane
              operator: Exists
              effect: NoSchedule
            - key: nvidia.com/gpu
              operator: Equal
              value: "true"
              effect: NoSchedule
          terminationGracePeriodSeconds: 120
          containers:
            - name: main
              image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0
              imagePullPolicy: IfNotPresent
              command:
                - python3
                - -m
                - dynamo.sglang
              args:
                - --model-path
                - /model-store/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
                - --served-model-name
                - Qwen/Qwen3-32B-FP8
                - --tp-size
                - "2"
                - --page-size
                - "64"
                - --context-length
                - "40960"
                - --trust-remote-code
                - --skip-tokenizer-init
                - --mem-fraction-static
                - "0.82"
                - --host
                - 0.0.0.0
                - --enable-metrics
                - --disable-piecewise-cuda-graph
              env:
                - name: DYN_SYSTEM_PORT
                  value: "8082"
                - name: GLOO_SOCKET_IFNAME
                  value: eth0
                - name: NCCL_SOCKET_IFNAME
                  value: eth0
              envFrom:
                - secretRef:
                    name: hf-token-secret
                    optional: true
              resources:
                requests:
                  nvidia.com/gpu: "2"
                limits:
                  nvidia.com/gpu: "2"
              securityContext:
                runAsUser: 0
                capabilities:
                  add:
                    - IPC_LOCK
                    - SYS_RESOURCE
              ports:
                - name: metrics
                  containerPort: 8082
              volumeMounts:
                - name: model-cache
                  mountPath: /model-store
                  readOnly: true
          volumes:
            - name: model-cache
              persistentVolumeClaim:
                claimName: model-cache
EOF
```

## 3. Apply and watch startup

```bash
kubectl apply -f /ephemeral/shared/qwen3-32b/manifests/00-aggregated-colocated-serving.yaml
kubectl get dynamographdeployment qwen32-agg -n "$NAMESPACE" -w
```

In another terminal:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-agg \
  -o wide -w
```

Model initialization can take several minutes. Once nine pods exist, wait for readiness:

```bash
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-agg \
  -n "$NAMESPACE" --timeout=45m
```

If a pod is `Pending` or restarts, inspect it before retrying:

```bash
kubectl get events -n "$NAMESPACE" --sort-by=.lastTimestamp | tail -80
kubectl describe pod POD_NAME -n "$NAMESPACE"
kubectl logs POD_NAME -n "$NAMESPACE" --all-containers --tail=300
```

Do not reduce the replica count or GPU request merely to make pods schedule. Fix the cluster prerequisite that the event or description names.

## 4. Verify topology and aggregated mode

Exactly four worker pods must be on `gpu05`, four on `gpu06`, and the frontend
on `gpu05`:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-agg \
  -L nvidia.com/dynamo-component-type -o wide
```

Confirm that all worker command lines omit disaggregation and cache-event
flags:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-agg,nvidia.com/dynamo-component-type=worker' \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].args}{"\n"}{end}'
```

The output must contain neither `--disaggregation-mode` nor
`--kv-events-config`. Worker logs must not report NIXL bootstrap or KV-transfer
initialization:

```bash
kubectl logs -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-agg,nvidia.com/dynamo-component-type=worker' \
  --all-containers --prefix --tail=1000 --max-log-requests=8 \
  | grep -Ei 'disaggregation|nixl|kv.transfer' || true
```

## 5. Check the API and exercise round-robin routing

Open a private port-forward from `gpu05`; this is for readiness testing only:

```bash
export FRONTEND_POD="$(kubectl get pods -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-agg,nvidia.com/dynamo-component-type=frontend' \
  -o jsonpath='{.items[0].metadata.name}')"

kubectl port-forward -n "$NAMESPACE" pod/"$FRONTEND_POD" 8000:8000
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role": "user", "content": "Reply with exactly READY. /no_think"}],
    "temperature": 0,
    "max_tokens": 16,
    "stream": false
  }'
```

Send at least two full routing cycles:

```bash
for request in $(seq 1 16); do
  curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data "{\"model\":\"Qwen/Qwen3-32B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"Return request number ${request}. /no_think\"}],\"temperature\":0,\"max_tokens\":32}" \
    >/dev/null
done
```

Inspect all worker logs and confirm that every replica handled traffic:

```bash
kubectl logs -n "$NAMESPACE" \
  -l 'nvidia.com/dynamo-graph-deployment-name=qwen32-agg,nvidia.com/dynamo-component-type=worker' \
  --all-containers --prefix --since=10m --max-log-requests=8
```

## 6. Acceptance criteria

Experiment 0 is deployment-ready only when:

- all nine pods remain Ready with zero restarts
- four worker pods run on `gpu05`, four run on `gpu06`, and the frontend runs on `gpu05`
- all 16 GPUs are allocated exactly once;
- `/v1/models` lists `Qwen/Qwen3-32B-FP8`
- the smoke response is valid and all eight workers handle requests
- the frontend is explicitly in `round-robin` mode
- every worker performs colocated prefill and decode
- no disaggregation, NIXL transfer, KV-event, HiCache, or RDMA setting is present.


## 7. Shutdown

Stop any port-forward, then remove only this experiment:

```bash
kubectl delete dynamographdeployment qwen32-agg -n "$NAMESPACE" --wait=true
kubectl wait --for=delete pod \
  -l nvidia.com/dynamo-graph-deployment-name=qwen32-agg \
  -n "$NAMESPACE" --timeout=10m
kubectl get pods -n "$NAMESPACE" -o wide
```


## References

- [Dynamo v1.3.0 aggregated SGLang manifest](https://github.com/ai-dynamo/dynamo/blob/v1.3.0/examples/backends/sglang/deploy/v1beta1/agg.yaml)
- [Dynamo SGLang backend](https://docs.nvidia.com/dynamo/latest/backends/sglang/sglang.html)

