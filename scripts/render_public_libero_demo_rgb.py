"""Display original public HDF5 RGB; no environment or policy is constructed."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/libero_goal9_original_demo_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", choices=["demo_0", "demo_25", "demo_49"])
    args = parser.parse_args()
    meta = json.loads((BASE / "HDF5_METADATA.json").read_text())
    env = json.loads(meta["data_attributes"]["env_args"])
    fps = env["env_kwargs"]["control_freq"]
    assert fps == 20 and meta["data_attributes"]["macros_image_convention"] == "opengl"
    out = BASE / args.episode
    source = out / "PUBLIC_DEMO_ARRAYS.npz"
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    actions = arrays["actions"]
    count = len(actions)
    close = np.flatnonzero(actions[:, 6] > 0.5)
    first_close = int(close[0]) if len(close) else None
    indices = {"BEFORE_CLOSE": max(0, first_close - 5), "AFTER_CLOSE": min(count - 1, first_close + 20), "FINAL": count - 1} if first_close is not None else {"FINAL": count - 1}
    font = ImageFont.load_default()
    raw_video = out / "PUBLIC_HUMAN_DEMO_STORED_RGB_20HZ.mp4"
    if raw_video.exists():
        raise FileExistsError(raw_video)
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (768, 484))
    assert writer.isOpened()
    for i in range(count):
        canvas = Image.new("RGB", (768, 484), "#081014")
        d = ImageDraw.Draw(canvas)
        d.text((8, 3), f"PUBLIC LIBERO HUMAN DEMO | {args.episode} | stored RGB | 20Hz 1x | frame {i}/{count-1}", fill="white", font=font)
        d.text((8, 21), "No simulation / new policy / eval-init mapping | vertical flip only", fill="white", font=font)
        for column, (key, label) in enumerate([("obs__agentview_rgb", "agentview"), ("obs__eye_in_hand_rgb", "wrist")]):
            raw = arrays[key][i]
            assert raw.shape == (128, 128, 3) and raw.dtype == np.uint8
            natural = raw[::-1, :, :]
            assert np.array_equal(natural[::-1], raw)
            d.text((column * 384 + 8, 42), label, fill="white", font=font)
            canvas.paste(Image.fromarray(natural).resize((384, 384), Image.Resampling.NEAREST), (column * 384, 60))
        gap = abs(np.diff(arrays["obs__gripper_states"][i])[0]) * 1000
        d.text((8, 448), f"gripper command {actions[i,6]:+.1f} | joint-separation proxy {gap:.2f}mm (not object contact test)", fill="white", font=font)
        d.text((8, 466), "yifengzhu-hf/LIBERO-datasets / f13aa24a3da8 | source128 RGB -> 3x nearest", fill="white", font=font)
        rgb = np.asarray(canvas)
        writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
        for label, index in indices.items():
            if i == index:
                canvas.save(out / f"{label}_FRAME{i:03d}.png")
    writer.release()
    video = out / "PUBLIC_HUMAN_DEMO_STORED_RGB_20HZ_H264.mp4"
    subprocess.run(["/opt/homebrew/bin/ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-n", "-i", str(raw_video), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], check=True, timeout=30)
    probe = json.loads(subprocess.check_output(["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], timeout=10))
    stream = probe["streams"][0]
    cap = cv2.VideoCapture(str(video))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    assert decoded == count and stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    b = video.read_bytes()
    position = 0
    atoms = []
    while position + 8 <= len(b):
        size, kind = struct.unpack(">I4s", b[position:position + 8])
        atoms.append({"type": kind.decode("ascii"), "offset": position})
        if size == 1:
            size = struct.unpack(">Q", b[position + 8:position + 16])[0]
        if not size:
            break
        position += size
    assert next(x["offset"] for x in atoms if x["type"] == "moov") < next(x["offset"] for x in atoms if x["type"] == "mdat")
    result = {"status": "PUBLIC_ORIGINAL_HUMAN_DEMO_RGB_VIDEO_VERIFIED", "episode": args.episode,
              "source": str(source), "source_subset_npz_sha256": sha(source),
              "source_array_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
              "full_hdf5_sha_locally_verified": False, "output": str(video), "output_sha256": sha(video),
              "frames": count, "decoded_frames": decoded, "fps": fps,
              "fps_source": "HDF5 data.attrs.env_args.env_kwargs.control_freq=20",
              "duration_s": float(probe["format"]["duration"]), "speed": 1,
              "source_image_convention": "opengl", "display_transform": "vertical flip only; RGB channel order unchanged; nearest-neighbor 3x display enlargement; separate text margins",
              "display_is_lossless_raw_rgb": False, "display_note": "Raw extracted arrays preserved without pixel change; H264 is lossy display encoding.",
              "codec": stream["codec_name"], "pix_fmt": stream["pix_fmt"], "faststart": True,
              "first_positive_close_command_frame": first_close, "stills": indices,
              "gripper_proxy_mm_at_stills": {k: float(abs(np.diff(arrays["obs__gripper_states"][i])[0]) * 1000) for k, i in indices.items()},
              "native_or_quality_recomputed": False, "mapped_to_current_evaluation_init": False,
              "new_simulation_steps": 0, "new_policy_inference": 0, "new_Gemini_calls": 0, "new_GPU_pods": 0}
    (out / "VIDEO_RECEIPT.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "RENDER_SCRIPT.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
