"""Read selected public LIBERO HDF5 ranges, without downloading a whole suite."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request

import h5py

ROOT = Path(__file__).resolve().parents[1]
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
PATH = "libero_goal/put_the_wine_bottle_on_the_rack_demo.hdf5"
URL = f"https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/{REVISION}/{PATH}"
SIZE = 878958730
EXPECTED_FULL_SHA256 = "f9092aa70734fc4083e97fc58c3ba25f87c614d18326182ddc7a455f0ab4da2e"


class PublicRanges(io.RawIOBase):
    def __init__(self, output: Path):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.position = 0
        self.block_size = 1024 * 1024
        self.transferred = 0
        self.requests = []
        self.start = time.monotonic()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)

    def read(self, n=-1):
        if n is None or n < 0:
            n = SIZE - self.position
        stop = min(SIZE, self.position + n)
        chunks = []
        while self.position < stop:
            block = self.position // self.block_size
            begin = block * self.block_size
            end = min(SIZE, begin + self.block_size) - 1
            path = self.output / f"block_{block:06d}.bin"
            if path.exists():
                data = path.read_bytes()
                if len(data) != end - begin + 1:
                    raise ValueError("incomplete cached public range")
            else:
                if self.transferred + end - begin + 1 > 128 * 1024 * 1024:
                    raise RuntimeError("public range transfer cap 128 MiB")
                if time.monotonic() - self.start > 300:
                    raise RuntimeError("public range wall limit 300s")
                request = urllib.request.Request(URL, headers={
                    "Range": f"bytes={begin}-{end}",
                    "User-Agent": "course-public-demo-inspection/1",
                })
                with urllib.request.urlopen(request, timeout=30) as response:
                    expected = f"bytes {begin}-{end}/{SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("public endpoint did not honor exact HTTP byte range")
                    data = response.read(end - begin + 2)
                if len(data) != end - begin + 1:
                    raise RuntimeError("short or oversized public range")
                path.write_bytes(data)
                self.transferred += len(data)
                self.requests.append({"start": begin, "end": end, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            local_begin = self.position - begin
            count = min(stop - self.position, len(data) - local_begin)
            chunks.append(data[local_begin:local_begin + count])
            self.position += count
        return b"".join(chunks)

    def receipt(self):
        return {
            "public_repo": "yifengzhu-hf/LIBERO-datasets", "revision": REVISION,
            "path": PATH, "url": URL, "whole_file_size": SIZE,
            "expected_whole_file_lfs_sha256": EXPECTED_FULL_SHA256,
            "whole_file_sha_verified": False,
            "note": "Only selected byte ranges read. Expected full-file SHA is Hub metadata, not a locally computed full-file checksum.",
            "HTTP_authorization": "none", "range_bytes_transferred_this_process": self.transferred,
            "ranges_transferred_this_process": self.requests,
            "wall_s": time.monotonic() - self.start,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    out = ROOT / "results/libero_goal9_original_demo_v1"
    remote = PublicRanges(out / "public_range_cache")
    try:
        with h5py.File(remote, "r") as f:
            group = f["data"]
            names = sorted(group, key=lambda s: int(s.rsplit("_", 1)[-1]))
            selected = [names[0], names[len(names) // 2], names[-1]]
            metadata = {"episode_count": len(names), "selected_episodes_by_order": selected,
                        "data_attribute_keys": list(group.attrs), "episodes": {}}
            metadata["data_attributes"] = {}
            for key in group.attrs:
                value = group.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode("utf8")
                elif not isinstance(value, str):
                    value = value.tolist() if hasattr(value, "tolist") else value
                metadata["data_attributes"][key] = value
            for name in selected:
                demo = group[name]
                info = {"attributes": {k: str(v)[:500] for k, v in demo.attrs.items()}, "datasets": {}}
                demo.visititems(lambda k, obj: info["datasets"].update({k: {"shape": list(obj.shape), "dtype": str(obj.dtype)}}) if isinstance(obj, h5py.Dataset) else None)
                metadata["episodes"][name] = info
            (out / "HDF5_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps(metadata), flush=True)
            if args.extract:
                import numpy as np
                for name in selected:
                    demo = group[name]
                    dest = out / name
                    dest.mkdir(exist_ok=True)
                    arrays = {}
                    for key in ["actions", "states", "obs/agentview_rgb", "obs/eye_in_hand_rgb", "obs/gripper_states", "obs/ee_pos", "obs/ee_ori", "robot_states", "dones", "rewards"]:
                        if key in demo:
                            arrays[key.replace("/", "__")] = demo[key][:]
                    np.savez_compressed(dest / "PUBLIC_DEMO_ARRAYS.npz", **arrays)
                    print(json.dumps({"extracted": name, "arrays": {k: list(v.shape) for k, v in arrays.items()}, "transferred_bytes": remote.transferred}), flush=True)
    finally:
        (out / ("RANGE_EXTRACT_RECEIPT.json" if args.extract else "RANGE_INSPECTION_RECEIPT.json")).write_text(json.dumps(remote.receipt(), indent=2) + "\n")
        remote.close()


if __name__ == "__main__":
    main()
