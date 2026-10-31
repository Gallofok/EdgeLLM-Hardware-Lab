"""Read bounded GGUF v3 metadata/tensor descriptors without reading tensor contents."""

import math
import struct
from pathlib import Path

# Sizes/layouts from the pinned ggml.h and ggml-common.h, not filename labels.
TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}
FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


def inventory(path):
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        limit = min(size, 32 * 1024 * 1024)

        def read(count):
            if count < 0 or stream.tell() + count > limit:
                raise ValueError("GGUF header exceeds file/32-MiB metadata bound")
            data = stream.read(count)
            if len(data) != count:
                raise ValueError("Truncated GGUF header")
            return data

        def number(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

        def string(keep=True):
            length = number("Q")
            if length > 1024 * 1024:
                raise ValueError("GGUF string exceeds 1-MiB limit")
            data = read(length)
            return data.decode("utf-8") if keep else None

        def value(kind, keep=False):
            if kind in FORMATS:
                result = number(FORMATS[kind])
                if keep and isinstance(result, float) and not math.isfinite(result):
                    raise ValueError("Non-finite GGUF metadata")
                return result if keep else None
            if kind == 8:
                return string(keep)
            if kind == 9:
                item_type, count = number("I"), number("Q")
                if item_type == 9 or count > 1_000_000:
                    raise ValueError("Unsupported nested/oversized GGUF metadata array")
                for _ in range(count):
                    value(item_type)
                return None
            raise ValueError(f"Unknown GGUF metadata type {kind}")

        if read(4) != b"GGUF" or number("I") != 3:
            raise ValueError("GGUF v3 is required")
        tensor_count, metadata_count = number("Q"), number("Q")
        if not 0 < tensor_count <= 100_000 or metadata_count > 100_000:
            raise ValueError("Invalid GGUF descriptor counts")
        metadata, keys = {}, set()
        for _ in range(metadata_count):
            key, kind = string(), number("I")
            if key in keys:
                raise ValueError("Duplicate GGUF metadata key")
            keys.add(key)
            keep = key in ("general.architecture", "general.alignment", "general.file_type") or (
                key.endswith(
                    (
                        ".block_count",
                        ".embedding_length",
                        ".feed_forward_length",
                        ".attention.head_count",
                        ".attention.head_count_kv",
                    )
                )
            )
            item = value(kind, keep)
            if keep and item is not None:
                metadata[key] = item
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or not 0 < alignment <= 4096 or alignment & (alignment - 1):
            raise ValueError("Invalid GGUF alignment")
        tensors, names = [], set()
        for _ in range(tensor_count):
            name, dimensions = string(), number("I")
            if name in names or not 1 <= dimensions <= 4:
                raise ValueError("Duplicate tensor name or invalid dimension count")
            names.add(name)
            shape = [number("Q") for _ in range(dimensions)]
            if any(n == 0 or n > 2**31 for n in shape):
                raise ValueError("Invalid GGUF tensor shape")
            kind, offset = number("I"), number("Q")
            if kind not in TYPES:
                raise ValueError(f"Unsupported pinned-model tensor type {kind}")
            label, block, block_bytes = TYPES[kind]
            if shape[0] % block or offset % alignment:
                raise ValueError("Invalid tensor block width or alignment")
            elements = math.prod(shape)
            tensors.append(
                {
                    "name": name,
                    "shape_ggml": shape,
                    "type_id": kind,
                    "type": label,
                    "elements": elements,
                    "storage_bytes": elements // block * block_bytes,
                    "relative_data_offset": offset,
                }
            )
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        end = 0
        for tensor in sorted(tensors, key=lambda tensor: tensor["relative_data_offset"]):
            offset = tensor["relative_data_offset"]
            if offset < end or data_start + offset + tensor["storage_bytes"] > size:
                raise ValueError("Overlapping tensor storage or truncated GGUF data")
            end = offset + tensor["storage_bytes"]
    grouped = {}
    for tensor in tensors:
        group = grouped.setdefault(
            tensor["type"], {"tensors": 0, "elements": 0, "storage_bytes": 0}
        )
        for key, increment in (
            ("tensors", 1),
            ("elements", tensor["elements"]),
            ("storage_bytes", tensor["storage_bytes"]),
        ):
            group[key] += increment
    return {
        "schema_version": 1,
        "gguf_version": 3,
        "file_size_bytes": size,
        "tensor_data_offset": data_start,
        "metadata": metadata,
        "tensor_count": len(tensors),
        "total_elements": sum(t["elements"] for t in tensors),
        "total_tensor_storage_bytes": sum(t["storage_bytes"] for t in tensors),
        "by_type": grouped,
        "tensors": tensors,
        "notice": "Storage inventory, not runtime operator costs; tensor contents are not read",
    }
