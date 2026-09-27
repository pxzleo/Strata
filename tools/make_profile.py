"""Extend a STRP expert ranking with measured routing from --dump-routing.

The existing ranking stays first; newly observed pairs follow by descending
frequency. Verify-window records carry zero weights because that callback has
only expert IDs, which are all this tool needs.
"""

import argparse
import collections
import struct
from pathlib import Path

LAYERS = 48
EXPERTS = 512
HEADER = struct.Struct("<4s5I")
RECORD = struct.Struct("<2i")


def read_profile(path: Path) -> list[tuple[int, int]]:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: incomplete STRP header")
    magic, version, layers, experts, slots, count = HEADER.unpack_from(data)
    if (magic, version, layers, experts) != (b"STRP", 1, LAYERS, EXPERTS):
        raise ValueError(f"{path}: incompatible STRP header")
    if count > slots or len(data) < HEADER.size + count * 4:
        raise ValueError(f"{path}: incomplete ranked pairs")
    pairs = list(struct.iter_unpack("<HH", data[HEADER.size : HEADER.size + count * 4]))
    if len(set(pairs)) != len(pairs) or any(l >= LAYERS or e >= EXPERTS for l, e in pairs):
        raise ValueError(f"{path}: duplicate or out-of-range pair")
    return pairs


def count_trace(path: Path, counts: collections.Counter[tuple[int, int]]) -> int:
    records = 0
    with path.open("rb") as stream:
        while header := stream.read(RECORD.size):
            if len(header) != RECORD.size:
                raise ValueError(f"{path}: truncated record header")
            layer, k = RECORD.unpack(header)
            if not 0 <= layer < LAYERS or not 0 < k <= EXPERTS:
                raise ValueError(f"{path}: invalid layer {layer} or k {k} at record {records}")
            payload = stream.read(k * 8)  # k int32 IDs, followed by k float32 weights
            if len(payload) != k * 8:
                raise ValueError(f"{path}: truncated record {records}")
            for (expert,) in struct.iter_unpack("<i", payload[: k * 4]):
                if not 0 <= expert < EXPERTS:
                    raise ValueError(f"{path}: invalid expert {expert} at record {records}")
                counts[layer, expert] += 1
            records += 1
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--base-profile", required=True, type=Path)
    parser.add_argument("--slots", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = read_profile(args.base_profile)
    if not len(base) < args.slots <= LAYERS * EXPERTS:
        parser.error(f"--slots must be between {len(base) + 1} and {LAYERS * EXPERTS}")
    counts: collections.Counter[tuple[int, int]] = collections.Counter()
    records = sum(count_trace(path, counts) for path in args.traces)
    base_pairs = set(base)
    new = sorted((pair for pair in counts if pair not in base_pairs), key=lambda pair: (-counts[pair], pair))
    if len(base) + len(new) < args.slots:
        parser.error(f"only {len(base) + len(new)} ranked pairs available; collect more routing data")
    ranked = base + new[: args.slots - len(base)]
    inverse = [0xFFFFFFFF] * (LAYERS * EXPERTS)
    for slot, (layer, expert) in enumerate(ranked):
        inverse[layer * EXPERTS + expert] = slot
    with args.output.open("wb") as stream:
        stream.write(HEADER.pack(b"STRP", 1, LAYERS, EXPERTS, args.slots, len(ranked)))
        for pair in ranked:
            stream.write(struct.pack("<HH", *pair))
        stream.write(struct.pack(f"<{len(inverse)}I", *inverse))
    print(f"{args.output}: {len(ranked)} pairs; {records} routing records; {len(new)} new pairs observed")


if __name__ == "__main__":
    main()
