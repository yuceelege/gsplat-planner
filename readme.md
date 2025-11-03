## Gaussian Splat Planner

A compact pipeline for drone navigation in a synthetic indoor scene using 3D Gaussian Splatting for environment representation, a learned visual pose estimator for localization, RRT-based planning, and a PID controller for trajectory following.

### What this project does
- **Environment**: Represent the scene with a Gaussian Splat model trained via nerfstudio.
- **Occupancy**: Convert splats to a conservative obstacle/occlusion point cloud for planning.
- **Planning**: Plan collision-free paths with RRT, then smooth and time-sample them.
- **Localization**: Render images, estimate pose with a ResNet18-based `PoseNet`.
- **Control**: Track the path with a PID controller in a simple drone dynamics model.
- **Visualization**: Live simulation and plotting of drone motion and estimates.

### Repository layout
- `maps/scene_map_*.json`: Scene definitions used for dataset generation and rendering.
- `generate.py`: Create a synthetic image+pose dataset for splat training.
- `occupancy.py`: Load a trained splat checkpoint, filter/scale, and export a planning point cloud.
- `network.py`: `PoseNet` definition (ResNet18 backbone).
- `agent.py`: Minimal drone dynamics model.
- `pid_controller.py`: PID controller with limits and filtering.
- `live_test.py`: Live sim with rendering, pose estimation, control, and replanning.
- `full_test_with_plotting.py`: Batch/longer-run testing and plots.
- `gs_render.py`, `train_posenet.py`, `utils.py`, `rasterization.py`, `plotting*.py`: Support scripts/utilities.

### Requirements
- Python env with PyTorch, torchvision, numpy, matplotlib, scipy, open3d, pyrender, PIL, imageio, trimesh.
- nerfstudio installed for splat training (for data creation and `ns-train`).

### Quick start
1) Generate a dataset for splat training
```bash
python generate.py --size 1000
```
This creates `dataset/` with `images/`, `poses.csv`, and `transforms.json`.

2) Train a Gaussian Splat (nerfstudio)
```bash
ns-train splatfacto --data /absolute/path/to/dataset
```
After training, copy the resulting `.ckpt` to `splats/` (e.g., `splats/splat_1.ckpt`).

3) Build occupancy/occlusion point cloud from the splat
Set `ckpt_path` and `cache_file` in `occupancy.py` if needed, then:
```bash
python occupancy.py
```
This writes a cached `*.pkl` you can reuse (move into `occlusion_maps/` if preferred).

4) Collect rendered data for pose estimator (if required by your workflow)
```bash
python gs_render.py
```

5) Train the pose estimator
```bash
python train_posenet.py
```
Place trained models in `estimators/` (e.g., `estimators/pose_net_1.pth`).

6) Run simulation
```bash
python live_test.py
```
Adjust in-file paths for: map JSON, occlusion map cache (`CACHE`), and estimator weights.

### Safety and robustness
- Planning in bounded workspace with a conservative obstacle set (opacity mask, neighbor pruning, outlier removal, optional inflation via radii).
- PID with integrator clamping, derivative filtering, input limits.
- Deviation-triggered replanning (`REPLAN_DIST`) to recover when estimates drift.
- Blended pose (network + dead-reckoning) to reduce noise-induced control spikes.

### Notes
- File sizes (approx): each splat ~120 MB, each pose net ~45 MB, occupancy maps ~few MB.
- If you switch scenes (`maps/scene_map_*.json`), ensure corresponding checkpoints, caches, and estimator weights are aligned.
- Use absolute paths where convenient; adjust constants in scripts to point at your files.

### Minimal checklist
- [ ] Generate dataset (`generate.py`) and train splat (`ns-train splatfacto`).
- [ ] Export occupancy map (`occupancy.py`).
- [ ] Train pose net (`train_posenet.py`).
- [ ] Place artifacts: `splats/`, `occlusion_maps/`, `estimators/`.
- [ ] Run `live_test.py` or `full_test_with_plotting.py`.