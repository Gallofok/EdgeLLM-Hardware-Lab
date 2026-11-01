# Three-way inference control

Physical Raspberry Pi 3. A = original official llama-bench; B = integrated driver OFF;
C = the identical integrated executable/libraries ON (upstream-order profile).

| Phase | Threads | A original t/s | B OFF t/s | C ON t/s | B/A change | C/B change | C/A change |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 7.992703 | 7.974461 | 7.200697 | -0.228% | -9.703% | -9.909% |
| prefill | 4 | 30.655071 | 30.542164 | 27.654947 | -0.368% | -9.453% | -9.787% |
| decode | 1 | 6.128669 | 6.103680 | 5.018863 | -0.408% | -17.773% | -18.108% |
| decode | 4 | 16.286603 | 15.312457 | 13.811560 | -5.969% | -7.548% | -15.187% |

Changes are mean within-triplet throughput ratios minus one, not latency changes.
Six sessions cover all six variant orders at each of 1/4 threads; thread order
alternates. There are 36 formal processes and
216 native throughput observations. Each process
uses three native repetitions, separate pp128/tg16, depth0, GPU0, batch/ubatch128,
mmap1, poll0 and official warmup. Fresh processes use the pinned official
random-token harness/default C RNG seed1. Decode is not conditioned on pp128.

The JSON retains six matched triplets per phase/thread, sample standard deviations
of process means and ratios, and ranges. These are descriptive, not confidence
intervals. CSV rows trace every repetition to raw SHA256, row and integer duration.
Fixture mode executes no real model, sensor or dispatch gates.

Before physical timing: fixed model numerical gate; three smoke processes;
eight separate no-warmup audited proofs for ON/OFF × prefill/decode × 1/4 threads.
OFF proofs require zero handled rows and positive disabled fallback counts. ON
proofs require positive handled rows. All gates are excluded from formal statistics;
formal auditing is off. B and C use the same executable hash and library set.
Original and integrated artifacts must equal the preceding physical study's hashes
and remain unchanged throughout capture. Kernel code and tolerances are unchanged.

B/A includes build, wrapper, code layout and the disabled integration path; it does
not isolate wrapper cost. C/B localizes the effect of enabling the compatible path
in one build, including dispatch, kernel and downstream inference effects; it does
not identify an instruction-level root cause. No cross-machine or general speedup
claim follows. Six triplets on one board/model/configuration remain a small sample.

Source: `4a22e177b098bd614f0503f16112a38339d10886`. Formal lifecycle samples:
1796; maximum temperature 78.44 C;
frequencies MHz [1200.0]; raw throttle ['0x20000', '0x70000', '0x70005'];
current flags ['throttled_now', 'under_voltage_now']; unavailable throttle samples
0; peak child lifecycle RSS
130.8125 MiB. Cgroup OOM deltas (unknown is null):
[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None].

Sampling includes initialization/warmup/teardown, cannot assign every sample to a
timed phase, and cannot exclude short throttle events. Historical flags predate
or may include earlier runs. Temperature/scheduling may drift despite order balance.
No affinity, governor, clock, voltage, cooling, boot or swap setting is changed.
Private stderr/help/loader/compiler paths stay private; reviewed capture bytes are
retained unchanged. Compile/runtime defaults stay off. No kernel tuning is included.
