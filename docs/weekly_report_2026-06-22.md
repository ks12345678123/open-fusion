# Open-Fusion Reproduction Weekly Report

Date: 2026-06-22

## 1. Work Summary

This week focused on reproducing Open-Fusion on a Windows 11 workstation with Docker Desktop, WSL2 Linux containers, and an NVIDIA RTX 4080 SUPER GPU.

The current reproduction environment can run the core Open-Fusion geometry reconstruction and semantic `vlfusion` pipeline on the ICL dataset. A 300-frame semantic run on `icl/kt0` completed successfully and generated point cloud and mesh outputs.

## 2. Hardware And Runtime

- OS: Windows 11
- GPU: NVIDIA GeForce RTX 4080 SUPER
- VRAM: 16 GB
- NVIDIA driver: 596.36
- Docker Desktop: available
- Docker context: `desktop-linux`
- GPU container runtime: available via `--gpus all`
- Main Docker image: `openfusion:local`

GPU access was verified inside a CUDA container with `nvidia-smi`, and the OpenFusion container can access CUDA through PyTorch.

## 3. Repository And Data Preparation

The OpenFusion repository was cloned to:

```text
E:\OpenFusion
```

Prepared assets:

- Official OpenFusion source code
- PyTorch CUDA base image:

```text
pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
```

- Local reproduction image:

```text
openfusion:local
```

- SEEM checkpoint:

```text
openfusion/zoo/xdecoder_seem/checkpoints/seem_focall_v1.pt
```

- ICL dataset trajectories:

```text
sample/icl/living_room/kt0
sample/icl/living_room/kt1
sample/icl/living_room/kt2
sample/icl/living_room/kt3
```

The ICL sample data currently occupies about 2.3 GB.

## 4. Local Compatibility Changes

Several Windows/Docker compatibility changes were added.

### Docker Build

Added:

```text
docker/Dockerfile.local
```

The official Dockerfile uses Ubuntu official apt mirrors. During reproduction, package downloads from `archive.ubuntu.com` were unstable, so the local Dockerfile switches Ubuntu apt sources to the Tsinghua mirror while preserving the original dependency list.

### Windows Run Scripts

Added:

```text
tools/build_windows.ps1
tools/run_windows.ps1
tools/download_icl_windows.ps1
```

These scripts make the Windows workflow reproducible without manually writing long Docker commands.

### Headless Execution

Modified:

```text
main.py
```

Added:

```text
--no-vis
```

This avoids blocking execution on Open3D GUI windows in Windows/Docker environments while still saving `.ply` and `.glb` outputs.

### SEEM Tokenizer Compatibility

Modified:

```text
openfusion/zoo/xdecoder_seem/xdecoder/language/LangEncoder/build.py
```

The SEEM model uses `transformers==4.19.2`, which failed to load the CLIP tokenizer from the current HuggingFace cache layout. A fallback path was added using `huggingface_hub.snapshot_download`, then loading the tokenizer from the local snapshot directory.

## 5. Validation Results

The following checks passed:

- Docker `hello-world` container
- CUDA container with `nvidia-smi`
- PyTorch CUDA availability inside `openfusion:local`
- `open3d`, `detectron2`, `cv2`, and `openfusion` imports
- SEEM model loading
- ICL geometry reconstruction
- ICL semantic `vlfusion` execution

The main validation command was:

```powershell
powershell -ExecutionPolicy Bypass -File E:\OpenFusion\tools\run_windows.ps1 --algo vlfusion --data icl --scene kt0 --frames 300 --device cuda:0 --no-vis
```

The 300-frame semantic run completed successfully. The observed processing rate was about 10.8 FPS for this short run.

Generated outputs:

```text
icl_kt0/color_pc.ply
icl_kt0/color_mesh.ply
icl_kt0/color_mesh.glb
icl_kt0/semantic_pc.ply
```

The latest output sizes were approximately:

- `color_pc.ply`: 4.97 MB
- `color_mesh.ply`: 14.09 MB
- `color_mesh.glb`: 14.63 MB
- `semantic_pc.ply`: 4.97 MB

## 6. Issues Encountered And Fixes

### Docker Was Installed But Engine Was Not Initially Ready

The Docker CLI was available, but the Docker Desktop backend was not running. After starting Docker Desktop and using the `desktop-linux` context, Linux containers worked correctly.

### GPU Runtime Needed Verification

The GPU path was verified with:

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The container detected the RTX 4080 SUPER correctly.

### Official Data Script Was Not Windows-Friendly

The official `tools/download.sh` had Linux assumptions and CRLF-related issues when used from the Windows-mounted repository. A Windows-native ICL downloader was added.

### Official Docker Build Failed On Ubuntu Apt Mirror

The first build failed while downloading packages from Ubuntu official mirrors. A local Dockerfile using a more stable mirror fixed this.

### SEEM Tokenizer Failed To Load

SEEM checkpoint loading initially failed at `openai/clip-vit-base-patch32` tokenizer loading. A HuggingFace snapshot fallback fixed the issue.

## 7. Current Completion Level

README-level reproduction is mostly ready for ICL:

- Environment setup: complete
- Docker image build: complete
- SEEM checkpoint: complete
- ICL dataset: complete
- Geometry reconstruction: complete
- Semantic pipeline: complete for short and medium runs

Remaining work for broader official reproduction:

- Run full ICL trajectories
- Download and test Replica
- Inspect visual quality in CloudCompare, MeshLab, Open3D, or WSLg
- Prepare ScanNet only if quantitative reproduction is required
- Recreate paper-level metrics such as mAcc and f-mIoU, which are not fully automated in the official repository

## 8. Next Plan

Next steps:

1. Run full `ICL kt0` and optionally `kt1-kt3`.
2. Visualize `semantic_pc.ply` and `color_mesh.glb`.
3. Download Replica and reproduce the qualitative examples.
4. Decide whether to implement additional ScanNet evaluation scripts for paper-level quantitative metrics.

