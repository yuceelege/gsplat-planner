import os
import torch
import numpy as np
from scipy.spatial import cKDTree
import pickle
import matplotlib.pyplot as plt
import open3d as o3d

n_iters = 5
neigh_radius = 0.005
opacity_thresh = 0.8
outlier_nb = 20
outlier_std = 2.0
ckpt_path = "splats/splat_2.ckpt"
cache_file = "gsplat_cache_2.pkl"

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
        fm = data["fm"]
        radii = data["radii"]
        outlier_idx = data["outlier_idx"]
else:
    #load the model means, opacities and sclaes
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    sd = checkpoint.get("state_dict", checkpoint)
    pipe = sd["pipeline"]
    means = pipe["_model.gauss_params.means"].cpu().numpy()
    opacities = pipe["_model.gauss_params.opacities"].cpu().numpy().squeeze()
    scales = pipe["_model.gauss_params.scales"].cpu().numpy()
    # mask by opacity
    mask = opacities > opacity_thresh
    fm = means[mask]
    fs = scales[mask]
    radii = np.mean(fs, axis=1)
    radii[radii <= 0] = 1e-3
    # prune isolated splats
    for _ in range(n_iters):
        tree = cKDTree(fm[:, :2])
        neighs = tree.query_ball_point(fm[:, :2], r=neigh_radius)
        ok = np.array([len(nb) > 1 for nb in neighs])
        if ok.all():
            break
        fm = fm[ok]
        radii = radii[ok]
    # rescale into new scene bounds (assumed but can be overcome with 2 known points)
    old_min = np.array([-1.5, -1.5, -0.375])
    old_max = np.array([ 1.5,  1.5,  0.375])
    new_min = np.array([ 0.0,  0.0,  0.0])
    new_max = np.array([20.0, 20.0,  5.0])
    scale   = (new_max - new_min) / (old_max - old_min)
    fm = (fm - old_min) * scale + new_min
    radii *= scale[0]
    # remove statistical outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fm)
    pcd_clean, outlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=outlier_nb,
        std_ratio=outlier_std
    )
    # save for next time
    to_save = {"fm": fm,"radii": radii,"outlier_idx": outlier_idx}
    with open(cache_file, "wb") as f:
        pickle.dump(to_save, f)
#plotting details for the bloated point cloud/occlusion map
pcd_pts = fm[outlier_idx]
pcd_colors = np.tile(np.array([0.2, 0.2, 0.2]), (pcd_pts.shape[0], 1))
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    pcd_pts[:, 0],
    pcd_pts[:, 1],
    pcd_pts[:, 2],
    c=pcd_colors,
    s=1,
    alpha=0.2
)
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_zlim(0, 6)
ax.set_axis_off() 
ax.grid(False)
plt.tight_layout()
plt.show()
