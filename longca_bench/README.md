# Benchmark Experiments for LongCA-Bench


## Kernel-Level Attention Performance and Flexibility

TODO ... (add more instructions to reproduce the experiments)

basic running command:

```bash
cd longca_bench/attn

bash run_benchmark.sh
```

sparse attention bench
```bash
PYTHONPATH=./ python longca_bench/attn/run_block_sparse_benchmark.py
```


## Module-Level Distributed Attention Performance and Scalability

TODO ... (add more instructions to reproduce the experiments)

basic running command:

```bash
cd longca_bench/dist_attn

export PYTHONPATH="${PYTHONPATH}:/path/to/LongCA-Bench/"

bash run_benchmark.sh
```
