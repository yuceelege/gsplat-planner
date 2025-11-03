import argparse
import json
import numpy as np
import pyrender
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from trimesh.transformations import euler_matrix, quaternion_matrix
from PIL import Image
import torchvision.transforms as T

from network import PoseNet
from utils import *
from agent import Drone
from pid_controller import PIDController



# sim params
WIDTH, HEIGHT = 640, 480
FOV_Y = np.pi/4
v_ref = 2.0
SIM_DT = 0.1
REPLAN_DIST = 2.0
# this parameter was for testing, keep true default
USE_NET_FEEDBACK = True
EPSILON = 1.0
# start-goal defs for 3 maps
START1, GOAL1 = [3,3,3], [17,17,3]
START2, GOAL2 = [2,18,3], [18,2,3]
START3, GOAL3 = [2,18,3], [18,2,3]
# setup dirs for each map
NOISE_LEVELS = np.arange(0.01, 0.16, 0.02)
MAP_CONFIGS = [
    {'id':1, 'cache':'occlusion_maps/gsplat_cache_1.pkl', 'scene':'maps/scene_map_1.json', 'pose':'estimators/pose_net_1.pth', 'start':START1, 'goal':GOAL1},
    {'id':2, 'cache':'occlusion_maps/gsplat_cache_2.pkl', 'scene':'maps/scene_map_2.json', 'pose':'estimators/pose_net_2.pth', 'start':START2, 'goal':GOAL2},
    {'id':3, 'cache':'occlusion_maps/gsplat_cache_3.pkl', 'scene':'maps/scene_map_3.json', 'pose':'estimators/pose_net_3.pth', 'start':START3, 'goal':GOAL3},
]

# normalization transform for imagenet weights, constants found from source cited on report
transform = T.Compose([
    T.Pad((0,80,0,80)),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def plan(start, goal, pts):
    # apply rrt given start, end and room borders
    tree, raw, obs = rrt(start, goal, pts, xr=(0,20), yr=(0,20), zr=(2.8,3.2))
    # checks points in rrt and connects for smooth shortcuts
    path = smooth_shortcuts(raw, obs)
    # samples along path with constant speed for discrete PID
    sampled = sample_along_path(path, v_ref, SIM_DT)
    return sampled


def is_pose_valid(eye, objects):
    return not any(point_inside_object(eye, obj) for obj in objects)


def run_experiment(scene_file, pose_file, pts, objects, start, goal, noise, visualize=False):
    #initializations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 7 bc we estimate quaternions then map back to euler
    model = PoseNet(num_outputs=7).to(device)
    model.load_state_dict(torch.load(pose_file, map_location=device))
    model.eval()

    scene = load_scene_from_json(scene_file)
    scene.ambient_light = np.array([0.5]*3)
    camera = pyrender.PerspectiveCamera(yfov=FOV_Y)
    renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)
    cam_node = scene.add(camera, pose=np.eye(4))

    sampled = plan(start, goal, pts)
    ideal_length = sum(np.linalg.norm(sampled[i+1]-sampled[i]) for i in range(len(sampled)-1))

    init_tan = sampled[1] - sampled[0]
    theta0 = np.arctan2(init_tan[1], init_tan[0])
    drone = Drone([*start, v_ref, 0.0, theta0])
    pid = PIDController(dt=SIM_DT)
    # yaw and xyz noise set equal wlog
    NOISE_STD_POS = NOISE_STD_YAW = noise

    steps = 0
    traveled = 0.0
    prev_pos = np.array(start)
    traj = []
    success = True

    last_pred, last_yaw = None, theta0
    idx = 0
    while idx < len(sampled)-1:
        pos = drone.state[:3].copy()
        traj.append(pos)
        if not is_pose_valid(pos, objects):
            success = False
            break
        if np.linalg.norm(pos - sampled[-1]) < EPSILON:
            break
        # obtain camera pose by taking drone pose, doing adjust, and render
        T2 = euler_matrix(np.pi/2,0,drone.state[5]-np.pi/2,axes='sxyz')
        T2[:3,3] = pos
        scene.set_pose(cam_node, T2)
        img, _ = renderer.render(scene)

        inp = transform(Image.fromarray(img)).unsqueeze(0).to(device)
        # inference and get quatern
        with torch.no_grad(): 
            out = model(inp).cpu().squeeze().numpy()
        pred_pos, pred_q = out[:3], out[3:7]
        R_pred = quaternion_matrix(pred_q)[:3,:3]
        R_est = T2[:3,:3] @ R_pred @ euler_matrix(np.pi/2,0,-np.pi/2,axes='sxyz')[:3,:3]
        yaw_m = np.arctan2(R_est[1,0], R_est[0,0])
        # dead reckoning for performance improvement
        if last_pred is None:
            last_pred = pred_pos.copy()
        else:
            v = drone.state[3]
            ins = last_pred + np.array([v*np.cos(last_yaw), v*np.sin(last_yaw), drone.state[4]])*SIM_DT
            # change weights for adjustment, this is default
            last_pred = 0.55*ins + 0.45*pred_pos
        last_yaw = yaw_m

        fb = drone.state.copy()
        if USE_NET_FEEDBACK:
            fb[0:3], fb[5] = last_pred, yaw_m
        #update pid based on estimated pose
        u = pid.compute_control(fb, sampled[idx], (sampled[idx+1]-sampled[idx])/SIM_DT, sampled[idx+1]-sampled[idx])
        drone.update_state(u, SIM_DT)
        # inject noise on next state
        drone.state[0:3] += np.random.normal(scale=NOISE_STD_POS, size=3)
        drone.state[5] += np.random.normal(scale=NOISE_STD_YAW)

        steps += 1
        d = np.linalg.norm(drone.state[:3] - prev_pos)
        traveled += d
        prev_pos = drone.state[:3].copy()
        idx += 1

    renderer.delete()
    time = steps * SIM_DT
    #unreported metric for test
    eff = ideal_length / traveled if traveled > 0 else 0.0
    return time, eff, success, np.array(traj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10, help='Trials per noise')
    parser.add_argument('--visualize', action='store_true', help='Enable vis')
    args = parser.parse_args()
    #load map infos
    map_data = {}
    for cfg in MAP_CONFIGS:
        pts, _ = load_pcd_data(cfg['cache'])
        objects = json.load(open(cfg['scene']))['objects']
        map_data[cfg['id']] = {'pts':pts, 'objects':objects}
    # the rest goes for running the experiment for each noise,map pair 10 times and saving trajectory plots and printing results
    for cfg in MAP_CONFIGS:
        data = map_data[cfg['id']]
        for noise in NOISE_LEVELS:
            print(f"Map {cfg['id']} | Noise {noise:.2f}")
            results, trajs = [], []
            for run in range(1, args.runs+1):
                t, eff, ok, traj = run_experiment(
                    cfg['scene'], cfg['pose'], data['pts'], data['objects'], cfg['start'], cfg['goal'], noise, args.visualize
                )
                results.append((t, eff, ok))
                trajs.append(traj)
                print(f"  Run {run}/{args.runs}: time={t:.2f}, eff={eff:.3f}, ok={ok}")

            avg_t = np.mean([r[0] for r in results])
            avg_e = np.mean([r[1] for r in results])
            succ = sum(r[2] for r in results)
            print(f"--> Avg time {avg_t:.2f}s | Avg eff {avg_e:.3f} | Success {succ}/{args.runs}")

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data['pts'][:,0], data['pts'][:,1], data['pts'][:,2], c='gray', s=1, alpha=0.6)
            cmap = mpl.colormaps['tab10'].resampled(args.runs)
            for i, traj in enumerate(trajs):
                ax.plot(traj[:,0], traj[:,1], traj[:,2], c=cmap(i), lw=3)
            ax.view_init(elev=90, azim=-90)
            filename = f"map{cfg['id']}_noise{int(noise*100):02d}_runs{args.runs}.png"
            fig.savefig(filename, dpi=150)
            print(f"Saved plot to '{filename}'")
            plt.close(fig)

if __name__ == '__main__':
    main()
