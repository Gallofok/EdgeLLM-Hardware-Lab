# Raspberry Pi 3 context-depth study

**REAL DEVICE RUN**

Real Raspberry Pi 3 run; inspect telemetry and failures.

**Measurement caveat:** Power/thermal flags observed; historical flags may predate this run

## Environment and reproducibility

- Device: Raspberry Pi 3 Model B Rev 1.2
- Architecture: aarch64
- OS: 'PRETTY_NAME="Debian GNU/Linux 13 (trixie)"\nNAME="Debian GNU/Linux"\nVERSION_ID="13"\nVERSION="13 (trixie)"\nVERSION_CODENAME=trixie\nDEBIAN_VERSION_FULL=13.6\nID=debian\nHOME_URL="https://www.debian.org/"\nSUPPORT_URL="https://www.debian.org/support"\nBUG_REPORT_URL="https://bugs.debian.org/"'
- Kernel: 6.18.39+rpt-rpi-v8
- Cores: 4
- RAM (MiB): 905.10546875
- CPU features: ['fp', 'asimd', 'evtstrm', 'crc32', 'cpuid']
- llama.cpp: `a7a98e0fffed794396b3fbad4dcdbbc184963645`
- Model: `SmolLM2-135M-Instruct-Q4_K_M.gguf`
- Model SHA256: `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`
- Model revision: `09816acd5d99df7be770d85ea30822623dab342c`
- Started: 2026-10-31T07:20:19.112811+00:00; finished: 2026-10-31T07:44:40.599635+00:00
- Run status: ok; suite: context

Protocol: `{"batch_size": 128, "cooldown_s": 5, "gpu_layers": 0, "mmap": true, "ordering": "depth, threads 1/2/4, prefill then decode", "sample_interval_s": 1, "timeout_s": 1800, "ubatch_size": 128, "warmup": "native default enabled"}`

## Throughput

| Phase | Depth | Threads | Samples (t/s) | Mean ± sample SD (t/s) | Efficiency | Peak RSS (MiB) | Source |
|---|---:|---:|---|---:|---:|---:|---|
| prefill | 0 | 1 | 7.9849, 7.9915, 7.9896 | 7.9887 ± 0.0034 | 1.000 | 130.84 | `raw/prefill-t1-d0.json` row 0 |
| decode | 0 | 1 | 6.0666, 6.1298, 6.1351 | 6.1105 ± 0.0381 | 1.000 | 125.52 | `raw/decode-t1-d0.json` row 0 |
| prefill | 0 | 2 | 15.8358, 15.8280, 15.8352 | 15.8330 ± 0.0044 | 0.991 | 130.80 | `raw/prefill-t2-d0.json` row 0 |
| decode | 0 | 2 | 11.0222, 11.0516, 11.0352 | 11.0363 ± 0.0148 | 0.903 | 125.52 | `raw/decode-t2-d0.json` row 0 |
| prefill | 0 | 4 | 30.4999, 30.7050, 30.7494 | 30.6514 ± 0.1331 | 0.959 | 130.50 | `raw/prefill-t4-d0.json` row 0 |
| decode | 0 | 4 | 16.4232, 16.5272, 16.4901 | 16.4801 ± 0.0527 | 0.674 | 125.52 | `raw/decode-t4-d0.json` row 0 |
| prefill | 128 | 1 | 7.5217, 7.5222, 7.5210 | 7.5217 ± 0.0006 | 1.000 | 133.46 | `raw/prefill-t1-d128.json` row 0 |
| decode | 128 | 1 | 5.7225, 5.7289, 5.7240 | 5.7251 ± 0.0034 | 1.000 | 131.26 | `raw/decode-t1-d128.json` row 0 |
| prefill | 128 | 2 | 14.8902, 14.8839, 14.8672 | 14.8804 ± 0.0119 | 0.989 | 133.38 | `raw/prefill-t2-d128.json` row 0 |
| decode | 128 | 2 | 10.2378, 10.2385, 10.2269 | 10.2344 ± 0.0065 | 0.894 | 131.09 | `raw/decode-t2-d128.json` row 0 |
| prefill | 128 | 4 | 28.8067, 28.8831, 28.8227 | 28.8375 ± 0.0403 | 0.958 | 133.00 | `raw/prefill-t4-d128.json` row 0 |
| decode | 128 | 4 | 15.2731, 15.3351, 15.2764 | 15.2949 ± 0.0349 | 0.668 | 131.12 | `raw/decode-t4-d128.json` row 0 |
| prefill | 512 | 1 | 6.3916, 6.3605, 6.3915 | 6.3812 ± 0.0179 | 1.000 | 143.21 | `raw/prefill-t1-d512.json` row 0 |
| decode | 512 | 1 | 4.8111, 4.7739, 4.8091 | 4.7980 ± 0.0209 | 1.000 | 140.46 | `raw/decode-t1-d512.json` row 0 |
| prefill | 512 | 2 | 12.6804, 12.6796, 12.6745 | 12.6782 ± 0.0032 | 0.993 | 143.07 | `raw/prefill-t2-d512.json` row 0 |
| decode | 512 | 2 | 8.5020, 8.5014, 8.4932 | 8.4989 ± 0.0049 | 0.886 | 140.52 | `raw/decode-t2-d512.json` row 0 |
| prefill | 512 | 4 | 24.5046, 23.6600, 24.5521 | 24.2389 ± 0.5019 | 0.950 | 142.88 | `raw/prefill-t4-d512.json` row 0 |
| decode | 512 | 4 | 12.3987, 12.5860, 12.4789 | 12.4879 ± 0.0940 | 0.651 | 140.37 | `raw/decode-t4-d512.json` row 0 |

![Throughput plot](pi3_baseline.png)

## Resources and reliability

Resource values below cover whole benchmark processes including model load and warmup.

```json
{
  "samples": 1373,
  "aggregation": {
    "temperature_c": "maximum",
    "frequency_mhz": "minimum",
    "available_memory_mb": "minimum",
    "cpu_utilization_pct": "mean"
  },
  "temperature_c": 80.054,
  "frequency_mhz": 600.0,
  "available_memory_mb": 609.1015625,
  "cpu_utilization_pct": 43.54936316145969,
  "throttle_flags_observed": [
    "frequency_capped_now",
    "frequency_capped_occurred"
  ],
  "throttle_missing_samples": 0,
  "unavailable_reasons": [
    "No prior CPU sample",
    "Unavailable (KeyError)"
  ],
  "warnings": [
    "Power/thermal flags observed; historical flags may predate this run"
  ]
}
```

## Failures and OOM evidence

```json
[]
```


## Analysis and limitations

Compare phases and context depths separately. Efficiency is (throughput at T threads / throughput at 1 thread) / T. This is thread scaling within an unmodified backend, not a custom-kernel optimization claim.
Highest observed depth-0 prefill throughput in this run: 4 thread(s); see the table and raw samples.
Highest observed depth-0 decode throughput in this run: 4 thread(s); see the table and raw samples.

CV > 10% heuristic flagged: none among successful cases.
Only three timed repetitions are used; this is not a confidence interval or a stability proof.
Decode at depth 0 starts with an empty cache; it does not follow the 128-token prefill test. The context suite uses native --n-depth to prefill the cache outside the timed region.
Peak RSS uses Linux wait4 per child, not whole-system or process-tree memory. Periodic VmHWM/RSS and sensors are also retained. All MB columns mean MiB.
CPU utilization is whole-system busy time (0–100%), not process CPU percent. Frequency sampling uses cpu0 as a representative core; 1-second sampling can miss short events.
Historical throttle bits can predate this run; missing vcgencmd means unknown, not no throttling.
SIGKILL is not proof of OOM. Cgroup OOM counter deltas are supporting, shared-scope evidence; confirm with locally reviewed kernel logs before attributing OOM to this benchmark.
Model loading, native warmup and cooldown affect conditions. Fixed order and uncontrolled ambient temperature, background services, cooling, power and swap limit causal conclusions.

Every throughput sample maps to raw_file + raw_row_index + repetition (1-based) in benchmark.csv. Raw SHA256 is verified when regenerating reports. Keep the entire run directory with its raw evidence.
