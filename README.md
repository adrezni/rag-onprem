# Granite Vision 3.3 2B on OpenShift (CPU-only, vLLM)

Apply in this order:

```sh
oc apply -f 01-pvc.yaml
oc apply -f 02-download-model.yaml
oc logs -f job/download-granite-vision      # wait for it to finish
```

```sh
oc apply -f 03-servingruntime.yaml
oc apply -f 04-inferenceservice.yaml
oc get inferenceservice granite-vision-model -w   # wait for READY: True
```

## Notes

- Model repo is `ibm-granite/granite-vision-3.3-2b` (full precision, NOT the -GGUF
  variant) since vLLM requires `config.json` + safetensors, not GGUF files.
- `--max-model-len=8192` and `VLLM_CPU_KVCACHE_SPACE=2` (GiB) are tuned to fit inside
  a 12Gi memory limit on CPU-only inference. Increase both together if you have more
  RAM headroom and want longer context / multi-page document support.

Test with:

```sh
oc port-forward svc/granite-vision-model-predictor 8080:80
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "granite-vision-model", "messages": [{"role": "user", "content": "Hello"}]}'
```
