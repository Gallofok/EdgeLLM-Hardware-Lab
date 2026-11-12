# Cooled rerun of the dispatch end-to-end captures: protocol fixed in advance

Both [dispatch end-to-end captures](q5_dispatch_analysis.md) ended in no decision
because the board reached its frequency-capping temperature. This document fixes the
changes for a rerun. **It was committed before the rerun and contains no
measurement.** The candidate, its library, the kernel gates and the throughput bars
are unchanged.

## What changes, and only this

**A cool-down before every process.** Before each smoke, proof and formal process the
runner idles until the board reads 55 degrees Celsius or less, and gives up after 15
minutes. It runs nothing and changes no governor, clock, voltage or cooling setting.
Both variants wait under the same rule, so neither starts a process on heat left by
the other. Each wait and both temperatures are recorded with the process.

**A cleanliness test that still works after the firmware flag has latched.** The
"capping has occurred" flag set during the earlier capture stays set until a reboot,
so the earlier test, every sample reading `0x0`, can no longer pass on this boot
whatever happens. For a cooled capture, conditions are clean when:

- no sample shows any flag as currently active;
- every sample is below 80 degrees Celsius, the temperature at which capping first
  appeared; and
- no throttle sample is missing.

This is weaker than the earlier test in one respect: a latched flag also catches an
event between two one-second samples, and this test cannot. The temperature bound is
there to compensate, since capping on this board follows temperature. A reboot would
restore the stricter test; it was not done because it interrupts a machine the
project does not administer.

## What does not change

The decision rule is the one in the [dispatch protocol](q5_dispatch.md): prefill at
1, 2 and 4 threads and decode at 1 and 2 threads must each have a mean paired ratio
above 1 with at least three of four pairs favouring the candidate; decode at 4
threads must have a mean paired ratio of at least 0.99; unclean conditions give no
decision. Each model is judged separately, and the candidate is described as
generalizing only if both are accepted.

The timed library must match the hashes recorded by the kernel captures that already
passed. The model gate, smoke processes and audited proofs run again before timing.

```bash
python3 scripts/integration/two_row_e2e.py collect --candidate two_row_dispatch \
  --cool-to 55 --build build/ggml-dispatch \
  --kernel-capture .state/dispatch-kernel-576 --output .state/dispatch-e2e-135m-cooled
python3 scripts/integration/two_row_e2e.py collect --candidate two_row_dispatch \
  --cool-to 55 --versions config/models/smollm2-360m-q4km.json --build build/ggml-dispatch \
  --kernel-capture .state/dispatch-kernel-960 --output .state/dispatch-e2e-360m-cooled
```

## Consequences fixed in advance

Cooled captures are not directly comparable with the earlier back-to-back captures:
every process starts cooler, so absolute throughput may differ for both variants.
Only the paired ratios within a cooled capture are used. A cool start also means the
figures describe short bursts, not sustained load on this uncooled board; sustained
four-thread load is exactly what overheated it.

If a cooled capture is still unclean, it is published as no decision and the
remaining remedy is hardware cooling.
