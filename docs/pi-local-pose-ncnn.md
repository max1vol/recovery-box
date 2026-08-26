# Pi 3 local V4L2/NCNN pose backend

RecoveryBox's production Pi 3 lane is `recoverybox.device.pi_pose_v4l2`. One
child process owns V4L2 mmap capture, libyuv conversion, NanoDet, and RTMPose.
Only its numeric observation crosses private IPC to the Guardian parent. Raw
frames never enter IPC, Tailscale, the status file, a log, or persistent storage.
The lower-level estimator remains in `recoverybox.device.pi_pose_ncnn`.

## Pinned provenance

The runtime is Tencent NCNN `1.0.20260526`, BSD-3-Clause. The Raspberry Pi OS
13 / CPython 3.13 / armv7 wheel is pinned as follows:

- file: `ncnn-1.0.20260526-cp313-cp313-manylinux_2_31_armv7l.whl`
- size: `2,976,956` bytes
- SHA-256: `28c1e2a574b8f9bcbcc8c95de94d7814fb25b1a106f12dae3b7a8a4344d2db4b`
- source: `https://files.pythonhosted.org/packages/2f/2f/1a8f4c5d83213ac459865fbea4af7051ab63c5934fd2813cccfaf9bf6409/ncnn-1.0.20260526-cp313-cp313-manylinux_2_31_armv7l.whl`

The installed initializer, native extension, and vendored OpenMP library are
also checked before import:

- `ncnn/__init__.py`: `118` bytes,
  `ef90c76a49b37e74b0cd89f1da9502e764ee6b24a8da44860f4af894dc5838fe`
- `ncnn/ncnn.cpython-313-arm-linux-gnueabihf.so`: `6,611,269` bytes,
  `b64fdd46904e1a3379fd71f4179b2a06c3109eb4a6588414a5e0fdc22c7811c9`
- `ncnn.libs/libgomp-39027b09.so.1.0.0`: `184,421` bytes,
  `d94a8c0b47d2371b67b9417ff21cff03204f162f6db9c8000d78b11ce389caf9`

The top-down model is OpenMMLab RTMPose-t from the official MMDeploy archive.
MMPose and MMDeploy are Apache-2.0 projects. Retain their upstream notices; the
download archive itself does not contain a separate license file.

- archive size: `24,967,723` bytes
- archive SHA-256: `1c0481b760419a2140bf814d396335711ae1ffba91566395d1a529b4a177cd5d`
- `rtmpose-t.param`: `16,845` bytes,
  `0348745dcfded9842b546a4e904c410e6c9640a09b1713b1dba055c25b646595`
- `rtmpose-t.bin`: `13,332,744` bytes,
  `6fd3738741cfe14c82e40762434cdbe29c484b8a5a0b2bdefbc83d8bacb94c7c`
- source: `https://mmdeploy-oss.openmmlab.com/model/mmpose/rtmpose-t-ncnn-155ab7.zip`

RTMPose cannot prove that a person exists. The required presence gate is the
official NanoDet-m INT8 NCNN release from the Apache-2.0 NanoDet project:

- archive size: `910,094` bytes
- archive SHA-256: `2ac8f6ea9b5bb1cd52f809229c4492d7402b419daa11b068a13478b0fd33bc32`
- `nanodet-m-int8.param`: `17,127` bytes,
  `ecce99dba4f9bd9298eb753905917a1eedf84e94755a3915868b155755097f04`
- `nanodet-m-int8.bin`: `1,004,492` bytes,
  `719679139b0762d01663508a0893b026fdec09955f881d789b3b6bbc1ca900e1`
- source: `https://github.com/RangiLyu/nanodet/releases/download/v0.4.0/ncnn-nanodet-m-int8.zip`

Prepare the verified runtime and four model files with:

```sh
scripts/fetch-pi-pose-ncnn-runtime.sh /opt/recoverybox/runtime/ncnn
scripts/fetch-pi-pose-ncnn-models.sh /opt/recoverybox/models/ncnn
```

The deployment script performs these fetches in a temporary local stage, then
installs fresh root-owned files under `/opt/recoverybox`; operators normally do
not invoke them directly on the Pi. The estimator independently checks every
size and digest before loading it. It
also requires NCNN to be imported from the configured pinned runtime directory,
with an exact version match. Vulkan and FP16 paths are disabled on Pi 3; the
thermal-conscious default is two CPU inference threads.

## Fail-closed inference contract

The estimator accepts exactly one immutable BGRA frame, its dimensions, and a
same-clock monotonic capture timestamp. It does not retain the frame.

1. NanoDet checks only COCO class `person` and applies person-only NMS.
2. Zero, multiple, too-small, or low-confidence people return no pose.
3. The selected box receives the canonical RTMPose 1.25 padding and 192:256
   aspect adjustment. If that crop would leave the image, the result is
   non-assessable instead of extrapolated.
4. RTMPose returns exactly 17 COCO-order numeric points. SimCC X/Y rows are
   decoded by argmax, confidence is `min(max_x, max_y, person_score)`, and the
   crop transform is inverted into original-frame normalized coordinates.
5. Capture-to-completion evidence aged exactly 500 ms or more is discarded.
   A stale result never exposes points.

The Guardian parent owns an independent 500 ms watchdog. NCNN extraction runs
in the capture child, so native inference cannot starve that watchdog. Worker
death, a missed numeric record, or a deadline miss becomes `CAMERA_TIMEOUT`
without waiting for inference to return. The deployed V4L2 configuration is
640x480 YUYV at requested 10 FPS with eight mmap buffers; each cycle drains old
ready buffers before accepting the newest capture for inference.

## Live Pi 3 evidence from 2026-08-26

The device was a Raspberry Pi 3 running armv7 Raspberry Pi OS 13 and CPython
3.13.5. Benchmarks started only while `vcgencmd get_throttled` reported
`0xd0000`. No camera frame was saved or transmitted, and audio was never used.

- RTMPose-t FP32, two threads: warm inference `165.7-170.0 ms`.
- RTMPose-t FP32, four threads: warm inference `123.0-125.1 ms`.
- NanoDet-m INT8, two threads: first `259.958 ms`, warm `195.560 ms`, including
  NCNN input construction and all six output extractions.
- Warm two-thread detector plus pose model budget: approximately `365.6 ms`.

The camera separately delivered 20/20 640x480 YUYV frames. With eight mmap
buffers it reported exact 10.000 FPS timestamps at approximately 99.95 ms
intervals; no frame was saved or transmitted. The room contained no suitably
framed person, so `assessable=0` is truthful camera/detector evidence. It is not
evidence that RTMPose produced a human skeleton.

Short model bursts previously activated the Pi's soft-temperature-limit bit.
The silent local service can therefore be deployed fail-closed, but a reviewed
synthetic-person fixture through the exact RTMPose runtime, a fully framed human
RTMPose result, and sustained non-throttled capture-to-Guardian age below 500 ms
remain separate acceptance evidence. Do not weaken the no-person, ambiguity,
crop, confidence, or freshness gates to obtain it.
