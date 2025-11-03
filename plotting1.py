import os
import numpy as np
import pyrender
import torch
import pickle
import matplotlib.pyplot as plt
from trimesh.transformations import euler_matrix
from PIL import Image
import torchvision.transforms as T
from network import PoseNet
from utils import *


#responsible for plotting the pose estimation reusults when we fix a trajectory
WIDTH, HEIGHT = 640, 480
FOV_Y = np.pi / 4
CACHE_FILE = 'occlusion_maps/gsplat_cache_1.pkl'
CENTER = np.array([10.0, 10.0])

def load_pcd_data(cache_file):
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    return data['fm'][data['outlier_idx']], data['colors']

def init_scene():
    sc = load_scene_from_json('maps/scene_map_1.json')
    sc.ambient_light = np.array([0.5, 0.5, 0.5])
    return sc

def make_camera():
    return pyrender.PerspectiveCamera(yfov=FOV_Y)

def compute_camera_pose(pos, yaw):
    T = euler_matrix(np.pi/2, 0.0, yaw-np.pi/2, axes='sxyz')
    T[:3, 3] = pos
    return T

if __name__ == '__main__':
    pcd_pts, pcd_colors = load_pcd_data(CACHE_FILE)
    scene = init_scene()
    camera = make_camera()
    renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)
    cam_node = scene.add(camera, pose=np.eye(4))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseNet(num_outputs=7).to(device)
    model.load_state_dict(torch.load('estimators/pose_net_1.pth', map_location=device))
    model.eval()
    # normalization transform for resnet18 imagenet weights, constants found from source cited on report
    transform = T.Compose([
        T.Pad((0, 80, 0, 80)),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    shapes = ['Circle', 'Ellipse', 'Figure-8', 'Square']
    num_steps = 100
    t = np.linspace(0, 1, num_steps)
    all_real = []
    all_pred = []
    #parametrized definitions of the shapes
    for name in shapes:
        if name == 'Circle':
            R = 6
            x = CENTER[0] + R * np.cos(2 * np.pi * t)
            y = CENTER[1] + R * np.sin(2 * np.pi * t)
        elif name == 'Ellipse':
            a, b = 7, 3 
            x = CENTER[0] + a * np.cos(2 * np.pi * t)
            y = CENTER[1] + b * np.sin(2 * np.pi * t)
        elif name == 'Figure-8':
            theta = 2 * np.pi * t
            A = 6
            x = CENTER[0] + A * np.sin(theta)
            y = CENTER[1] + A * np.sin(theta) * np.cos(theta)
        else: 
            L = 6
            x = np.empty(num_steps)
            y = np.empty(num_steps)
            seg_len = num_steps // 4
            for i in range(num_steps):
                seg = i // seg_len
                frac = (i % seg_len) / seg_len
                if seg == 0:
                    x[i] = CENTER[0] + L * (1 - 2 * frac)
                    y[i] = CENTER[1] + L
                elif seg == 1:
                    x[i] = CENTER[0] - L
                    y[i] = CENTER[1] + L * (1 - 2 * frac)
                elif seg == 2:
                    x[i] = CENTER[0] - L * (1 - 2 * frac)
                    y[i] = CENTER[1] - L
                else:
                    x[i] = CENTER[0] + L
                    y[i] = CENTER[1] - L * (1 - 2 * frac)

        z = np.full(num_steps, 3)
        real_positions = np.stack([x, y, z], axis=1)
        yaws = np.arctan2(CENTER[1] - y, CENTER[0] - x)
        preds = []
        for pos, yaw in zip(real_positions, yaws):
            #render images with poses and prepare for inference
            scene.set_pose(cam_node, compute_camera_pose(pos, yaw))
            color, _ = renderer.render(scene)
            img = Image.fromarray(color)
            inp = transform(img).unsqueeze(0).to(device)
            #find the pose
            with torch.no_grad():
                out = model(inp).cpu().squeeze().numpy()
            preds.append(out[:3])
        all_real.append(real_positions)
        all_pred.append(np.array(preds))

    renderer.delete()
    fig = plt.figure(figsize=(12, 12))
    axs = fig.subplots(2, 2, subplot_kw={'projection': '3d'}).flatten()
    #plot
    for i, ax in enumerate(axs):
        real = all_real[i]
        pred = all_pred[i]
        ax.scatter(*pcd_pts.T, c=pcd_colors, s=1, alpha=0.2)
        ax.plot(real[:, 0], real[:, 1], real[:, 2], linewidth=3, label='real')
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], linewidth=3, label='pred')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_zlim(0, 6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(shapes[i])
        ax.legend()

    plt.tight_layout()
    plt.savefig('trajectory_collage.png', dpi=300)
    plt.show()
