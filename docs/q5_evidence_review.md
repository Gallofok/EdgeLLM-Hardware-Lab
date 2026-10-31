# Reviewed public Q5 baseline evidence

This publication follows the project's public, reproducible benchmark scope and
its prohibition on publishing device identifiers or credentials. The complete
original run remains private. The public dataset contains 83 checksummed inputs
plus its manifest, about 108 KB before derived reports. It contains no model
weights, executables, full compiler cache/command dump, instruction listing,
perf recording, address-bearing trace, environment-variable dump or setup log.

| Public record | Contents |
|---|---|
| Native timing JSON | Synthetic operand shape/seed, fixed symbol/ABI commit, integer batch durations, iteration counts, numerical errors/checksums |
| Sensor CSV | Run ID, timestamp/elapsed time, temperature, frequency, memory quantities, CPU utilization, raw/decoded throttle flags, null reasons |
| Process JSON | Exit/timeout/interruption, elapsed time, direct-child peak RSS, OOM evidence availability |
| Correctness gate | Pass message and explicit original-upstream check status |
| Setup provenance | Public dependency/model/engine versions and SHA256, compiler version, CPU build flags |
| Environment snapshots | Device product model, architecture, OS/kernel version, cores/features, memory/storage quantities, sensors; no identity fields |
| Build provenance | Allowlisted compiler version, relative source names, optimization/language flags and artifact hashes |
| Manifest / SOURCE | Commands with project/build placeholders, exact experiment protocol, evidence hashes and hashes of omitted private metadata |

All retained original timing, process, sensor, gate, stderr and setup bytes match
the private originals exactly. The public manifest explicitly updates its input
list; original manifest SHA256 is retained. Only derived build provenance is new.
Complete compiler cache/commands, redundant engine help and reference disassembly
are omitted, with original hashes retained privately and in SOURCE/provenance.

The schema review found no hostname, network address, serial number, username,
SSH/Wi-Fi configuration or secret field. A byte scan rejects home-directory
prefixes, IPv4 literals and private-key headers. Diagnostics never collect
nodename, interfaces, complete cpuinfo or environment variables. Sensor quantities
and public software/artifact identities are intentional experimental evidence.
A recorded-data test also rejects reintroducing the omitted complete dumps or
those identifier patterns. These checks support this specific reviewed release;
they do not automatically approve future logs or newly collected fields.
