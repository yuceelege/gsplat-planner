import torch 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from rasterization import rasterize_gaussians
from scipy.spatial.transform import Rotation 
import json
import imageio

width = 640
height = 480
f = 579.4112549695428


#gsplat dataparser transforms info (obtained after traning)
dt1 = {
    "transform": [
        [1.0, -2.5790756197352493e-08, 0.0001724449248285964, -9.993560791015625],
        [-2.5790756197352493e-08, 0.9999999403953552, 0.00029911877936683595, -9.908042907714844],
        [-0.0001724449248285964, -0.00029911877936683595, 0.9999999403953552, -3.0036022663116455]
    ],
    "scale": 0.1416838742332938
}

dt2 = {
    "transform": [
        [
            0.9999982118606567,
            -1.272946519748075e-06,
            -0.0018778726225718856,
            -9.934263229370117
        ],
        [
            -1.272946519748075e-06,
            0.9999991059303284,
            -0.0013557308120653033,
            -9.943338394165039
        ],
        [
            0.0018778726225718856,
            0.0013557308120653033,
            0.9999973177909851,
            -3.0574982166290283
        ]
    ],
    "scale": 0.14173056357232994
}

dt = {
    "transform": [
        [
            0.999997615814209,
            -1.1328575055813417e-06,
            -0.002182568656280637,
            -10.148791313171387
        ],
        [
            -1.1328575055813417e-06,
            0.999999463558197,
            -0.0010380941675975919,
            -10.089828491210938
        ],
        [
            0.002182568656280637,
            0.0010380941675975919,
            0.999997079372406,
            -3.081277847290039
        ]
    ],
    "scale": 0.13973326140845715
}

def get_viewmat(optimized_camera_to_world):
    #this for taking camera matrix and converting to view matrix
    #essential for GS or nerf
    R = optimized_camera_to_world[:, :3, :3]
    T = optimized_camera_to_world[:, :3, 3:4]
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

if __name__ == "__main__":
    os.makedirs("test", exist_ok=True)
    #read the poses from colmap compatible dataset and make dataframe
    with open("dataset/transforms.json", "r") as jf:
        frames = json.load(jf)["frames"]

    transform = np.array(dt['transform'], dtype=np.float32)
    transform_ap = np.vstack((transform, np.array([0,0,0,1], dtype=np.float32)))
    scale = dt['scale']
    tmp = np.linalg.inv(np.eye(4, dtype=np.float32))

    script_dir = os.path.dirname(os.path.realpath(__file__))
    #loads the gsplat model, means convs, feautres
    checkpoint_fn = os.path.join(script_dir, '.', "splats/splat_3.ckpt")
    res = torch.load(checkpoint_fn)
    means = res['pipeline']['_model.gauss_params.means']
    quats = res['pipeline']['_model.gauss_params.quats']
    opacities = res['pipeline']['_model.gauss_params.opacities']
    scales = res['pipeline']['_model.gauss_params.scales']
    colors = torch.sigmoid(res['pipeline']['_model.gauss_params.features_dc'])
    #eliminate unnecessary gaussians for computational efficiency
    mask_opacities = torch.sigmoid(opacities).squeeze() > 0.15
    mask_scale     = torch.all(scales > -8.0, dim=1)
    keep = mask_opacities & mask_scale
    means, quats, opacities, scales, colors = [
        x[keep] for x in (means, quats, opacities, scales, colors)
    ]

    for idx, frame in enumerate(frames):
        camera_pose = np.array(frame['transform_matrix'], dtype=np.float32)
        #global transform to align camera pose with gs
        camera_pose_transformed = tmp @ transform_ap @ camera_pose
        #get the R+T matrix
        camera_pose_transformed = camera_pose_transformed[:3, :]
        camera_pose_transformed[:3, 3] *= scale
        camera_pose_transformed = torch.Tensor(camera_pose_transformed)[None].to(means.device)
        #get the view matrix for rasterization
        view_mats = get_viewmat(camera_pose_transformed)
        Ks = torch.tensor([[
            [f, 0, width/2],
            [0, f, height/2],
            [0, 0, 1]
        ]], device=means.device)
        with torch.no_grad():
            out = rasterize_gaussians(means,quats / quats.norm(dim=-1, keepdim=True),torch.exp(scales),
                torch.sigmoid(opacities).squeeze(-1),
                colors,
                view_mats,
                Ks,
                width,
                height,
                eps2d=0.0
            )
        render = out['render']
        #dimensional adjustments
        if render.ndim == 4:
            img = render[0, ..., :3]
        else:
            img = render[..., :3]
        res_rgb = img.cpu().numpy().clip(0,1)
        #save rasterized image
        fn = f"images/gt_{idx:04d}.png"
        imageio.imwrite(fn, (res_rgb * 255).astype(np.uint8))
        print(f"Saved {fn}")
