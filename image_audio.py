#!/usr/bin/env python3
"""
Image <-> MP3 converter (spectrogram encoding).

Each image row becomes a sine-wave frequency, each image column a slice of
time, and pixel brightness becomes loudness. Decoding runs the analysis in
reverse. MP3 is lossy, so the recovered image is an approximation (a bit
soft/blurry horizontally), not a bit-exact copy.

Requirements:  pip install numpy scipy pillow
               ffmpeg installed and on PATH

Usage:
  python image_audio.py encode photo.jpg out.mp3 [--height 128] [--color]
  python image_audio.py decode out.mp3 restored.png [--height 128] [--color]
  (use the same --height / --color for decode as for encode)
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import correlate

SR = 22050          # sample rate
N = 1024            # samples per image column (~46 ms)
BIN_STEP = 2        # spacing between pixel-row frequencies, in units of SR/N (21.5 Hz)
BIN_START = 8       # lowest frequency bin (~170 Hz)
PAD = SR // 4       # silence before/after


def _freq_bins(n_rows):
    if BIN_START + BIN_STEP * n_rows >= N // 2:
        raise ValueError(f"Too many rows ({n_rows}); use --height <= {(N // 2 - BIN_START) // BIN_STEP}"
                         " (or <= 83 with --color)")
    return BIN_START + BIN_STEP * np.arange(n_rows)


def _basis(n_rows):
    """Sine bank, one per image row. Frames are 2N long (hop N, Hann window)."""
    rng = np.random.default_rng(1234)                     # fixed phases -> decoder can rebuild
    phases = rng.uniform(0, 2 * np.pi, n_rows)
    t = np.arange(2 * N)
    return np.cos(2 * np.pi * _freq_bins(n_rows)[:, None] * t[None, :] / N + phases[:, None])


def _synthesize(amps):
    """amps: (columns, rows) in 0..1 -> mono float signal."""
    cols, rows = amps.shape
    hann = np.hanning(2 * N + 1)[:-1]                     # periodic Hann, overlaps sum to 1
    frames = (amps @ _basis(rows)) * hann
    out = np.zeros((cols + 1) * N)
    for c in range(cols):
        out[c * N: c * N + 2 * N] += frames[c]
    return out


def _run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


# ----------------------------------------------------------------- encode
def encode(image_path, mp3_path, height=128, color=False):
    img = Image.open(image_path).convert("RGB" if color else "L")
    width = max(1, round(img.width * height / img.height))
    img = img.resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64) / 255.0       # (H, W) or (H, W, 3)

    if color:   # stack R, G, B as three frequency bands (top row = highest freq)
        arr = np.concatenate([arr[:, :, i] for i in range(3)], axis=0)
    amps = arr[::-1].T                                    # (columns, rows), row 0 = low freq

    sync = np.ones((1, amps.shape[1]))                    # sync/calibration column
    sig = _synthesize(np.vstack([sync, amps]))
    sig = sig / np.max(np.abs(sig)) * 0.9
    sig = np.concatenate([np.zeros(PAD), sig, np.zeros(PAD)])

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        wavfile.write(wav, SR, (sig * 32767).astype(np.int16))
        _run_ffmpeg(["-i", wav, "-codec:a", "libmp3lame", "-b:a", "192k", mp3_path])
    print(f"Wrote {mp3_path}  ({width}x{height} image, {len(sig) / SR:.1f}s audio)")


# ----------------------------------------------------------------- decode
def decode(mp3_path, out_path, height=128, color=False):
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        _run_ffmpeg(["-i", mp3_path, "-ac", "1", "-ar", str(SR), wav])
        _, sig = wavfile.read(wav)
    sig = sig.astype(np.float64) / 32768.0

    rows = height * (3 if color else 1)
    freqs = _freq_bins(rows)

    # Locate the start by cross-correlating with the known sync column.
    ref = _synthesize(np.ones((1, rows)))[: 2 * N]
    head = sig[: SR * 3]
    start = int(np.argmax(correlate(head, ref, mode="valid", method="fft")))

    sig = sig[start:]
    # Image width follows from the audio length: sync column + W columns + 1 tail hop + trailing pad.
    width = max(1, round((len(sig) - PAD) / N) - 2)
    n_frames = width + 1
    sig = np.concatenate([sig, np.zeros(2 * N)])
    hann = np.hanning(2 * N + 1)[:-1]
    spec = np.empty((n_frames, rows))
    for c in range(n_frames):
        seg = sig[c * N: c * N + 2 * N] * hann
        spec[c] = np.abs(np.fft.rfft(seg))[2 * freqs]     # bin index in a 2N-point FFT

    # Frame 0 is the sync column: use it to calibrate each frequency's gain.
    calib = np.maximum(spec[0], 1e-9)
    img = spec[1:] / calib
    img = np.clip(img, 0, 1).T[::-1]                      # (rows, columns), row 0 = top

    if color:
        planes = [img[i * height:(i + 1) * height] for i in range(3)]
        out = np.stack(planes, axis=-1)
    else:
        out = img
    Image.fromarray((out * 255).astype(np.uint8)).save(out_path)
    print(f"Wrote {out_path}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Image <-> MP3 spectrogram converter")
    p.add_argument("mode", choices=["encode", "decode"])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--height", type=int, default=128, help="pixel rows (default 128; max 83 with --color)")
    p.add_argument("--color", action="store_true", help="store RGB instead of grayscale")
    a = p.parse_args()
    (encode if a.mode == "encode" else decode)(a.src, a.dst, a.height, a.color)
