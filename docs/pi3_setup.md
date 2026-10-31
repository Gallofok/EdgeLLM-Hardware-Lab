# Raspberry Pi 3 setup and recovery

## Inspect before installation

Use your existing private LAN or VPN SSH connection, or the local terminal. This
project does not configure networking or require public SSH. Do not publish the
machine's hostname, IP, serial, SSH configuration or credentials.

```bash
bash scripts/pi/diagnose.sh
# Optional custom destination / checkout:
bash scripts/pi/diagnose.sh --output .state/pi-check.json --llama-dir vendor/llama.cpp
```

The JSON records device-tree model, Linux distribution, architecture, userspace
bitness, kernel, core count, instruction features, RAM, available memory, thermal
sensor, cpu0 frequency, raw/decoded power flags, disk capacity and llama.cpp
commit. No llama checkout yet means a null commit with a reason, not a failure.
Missing sensors are null and explicit. A non-ARM64 system returns exit code 2.

Pi 3 variants must be detected rather than assumed to have 1 GiB RAM. Available
memory is more useful than the label on the board. A 512 MiB device may fail even
with a 105 MB model because weights, KV cache, buffers and the OS need memory.
The model is fixed; do not silently choose a different quantization after failure.

## If the OS is 32-bit

Back up the SD card and user data first. On another computer, use Raspberry Pi
Imager to prepare **a spare SD card** with Raspberry Pi OS Lite **64-bit**.
Configure local access, boot the spare card, verify `uname -m` is `aarch64` and
`getconf LONG_BIT` is `64`, restore needed files, then rerun diagnostics. Review
Imager settings yourself. These scripts never flash a card or modify the OS boot
partition. Retain the original SD card as your rollback.

## Install

```bash
bash scripts/pi/setup.sh
```

APT installs Git, CMake, build-essential, curl, CA certificates, Python and
Matplotlib. Existing packages are handled by APT. On a prepared system you may
set `SKIP_APT=1`; the required commands must already work. Builds default to
`BUILD_JOBS=1`. Git fetches only the pinned llama.cpp commit and checks out a
detached HEAD; a dirty existing checkout is refused. Repeated setup reuses the
checkout, incremental build and verified model without deleting user data.

CMake targets generic `armv8-a` with `GGML_NATIVE=OFF`, avoiding optional newer
ARM instructions that Cortex-A53 lacks. CUDA, Metal, Vulkan, OpenCL and BLAS are
disabled; the benchmark additionally requests zero GPU layers. Build type is
Release. Setup saves compiler/CMake/Python versions, build flags, binary SHA256,
CMake-cache SHA256 and version pins to `.state/setup.json`.

The model URL includes a full Hugging Face revision. Downloads go to `.part`,
then size, GGUF header and upstream SHA256 are checked before atomic rename.
A corrupt existing model is refused rather than silently overwritten. To retry,
move it aside locally and rerun setup. Large files remain ignored by Git.

## Before a real experiment

Use a suitable stable power supply and note cooling, ambient conditions, power
source, background applications and swap configuration in your lab notes. Do
not change overclock, voltage or boot settings for this baseline. Close heavy
applications yourself if appropriate. Run `command -v vcgencmd` and
`vcgencmd get_throttled`; if unavailable or permission-denied, inspect your OS's
Raspberry Pi utilities installation/permissions locally. The sampler will record
null and a reason rather than claiming no throttling.

```bash
bash scripts/pi/benchmark.sh --suite baseline
```

Preflight requires Raspberry Pi 3, ARM64, a clean pinned checkout, unchanged
recorded binary/CMake cache, the exact model hash and options verified against
that binary's `--help`. Setup's help is saved, and each real suite also captures
actual help. Formal data is excluded unless native commit, parameters, duration,
repetition count and summaries match expectations.

## Failures and recovery

- Setup exits with a line number and private `.state/setup-*.log`. Inspect the
  actual compiler, disk, APT or network failure before retrying.
- Out-of-memory compilation: keep `BUILD_JOBS=1`; stop optional applications
  yourself if appropriate. Do not automatically resize swap or change the OS.
- Model/context allocation failure: retain native stderr, process status and
  telemetry. Record available memory, swap and context settings before evaluating
  a smaller model or configuration as a separate documented experiment.
- Timeout: the scheduler kills the benchmark process group and records the
  timeout separately from OOM. Increase `--timeout` deliberately for slow runs.
- SIGKILL / suspected OOM: inspect `journalctl -k --since '10 minutes ago'` locally
  for a kernel message naming the process. Do not publish full kernel logs. A
  cgroup-wide OOM delta can reflect another process and does not confirm cause.
- Low voltage, thermal cap or unstable results: keep the run and its flags. Fix
  the external cause, then create a new run directory. Never overwrite history.
- Ctrl-C: the child process group is stopped, per-case evidence saved, and a
  partial failure report generated when the parent can finish cleanup. SIGKILL,
  power loss or storage failure cannot guarantee final cleanup.
- Missing plotting package: `sudo apt-get install python3-matplotlib`, then
  regenerate with `python3 scripts/report.py --input <run-directory>`.

To retry without overwriting prior data:

```bash
bash scripts/pi/benchmark.sh --output results/pi3/retry-01 --reports reports/retry-01
```

No private device has been registered as a runner or made publicly accessible.
