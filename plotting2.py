import os
import numpy as np
import pyrender
import torch
import pickle
from trimesh.transformations import euler_matrix, euler_from_quaternion
from PIL import Image
import torchvision.transforms as T
from network import PoseNet
from utils import load_scene_from_json
#same code for plotting1.py but this is responsible for printing quantitative results
# Constants
WIDTH, HEIGHT = 640, 480
FOV_Y = np.pi / 4
#pick the map you want
CACHE_FILE = 'occlusion_maps/gsplat_cache_3.pkl'
CENTER = np.array([10.0, 10.0])

def load_pcd_data(cache_file):
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    return data['fm'][data['outlier_idx']], data['colors']

def init_scene():
    sc = load_scene_from_json('maps/scene_map_3.json')
    sc.ambient_light = np.array([0.5, 0.5, 0.5])
    return sc

def make_camera():
    return pyrender.PerspectiveCamera(yfov=FOV_Y)

def compute_camera_pose(pos, yaw):
    T = euler_matrix(np.pi/2, 0.0, yaw - np.pi/2, axes='sxyz')
    T[:3, 3] = pos
    return T
# same process as plotting1.py
if __name__ == '__main__':
    pcd_pts, pcd_colors = load_pcd_data(CACHE_FILE)
    scene = init_scene()
    camera = make_camera()
    renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)
    cam_node = scene.add(camera, pose=np.eye(4))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseNet(num_outputs=7).to(device)
    model.load_state_dict(torch.load(
        'estimators/pose_net_3.pth', map_location=device))
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
        z = np.full(num_steps, 3.0)
        real_pos = np.stack([x, y, z], axis=1)
        base_yaws = np.arctan2(CENTER[1] - y, CENTER[0] - x)
        commanded_yaws = base_yaws - np.pi/2
        preds_pos = []
        preds_yaw = []
        prev_q = None
        for pos, cyaw in zip(real_pos, base_yaws):
            scene.set_pose(cam_node, compute_camera_pose(pos, cyaw))
            color, _ = renderer.render(scene)
            img = Image.fromarray(color)
            inp = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp).cpu().numpy().squeeze()
            preds_pos.append(out[:3])
            q = out[3:7]
            if prev_q is not None and np.dot(prev_q, q) < 0:
                q = -q
            prev_q = q.copy()
            pyaw = euler_from_quaternion(q, axes='sxyz')[2]
            preds_yaw.append(pyaw)        
        preds_pos = np.array(preds_pos)
        preds_yaw = np.array(preds_yaw)
        pos_err = preds_pos - real_pos
        rmse_xyz = np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
        yaw_err = (preds_yaw - commanded_yaws + np.pi) % (2*np.pi) - np.pi
        rmse_yaw = np.sqrt(np.mean(yaw_err**2))
        print(f"{name}: Pos RMSE = {rmse_xyz:.3f} m, Yaw RMSE = {np.degrees(rmse_yaw):.2f}°")
    renderer.delete()
    