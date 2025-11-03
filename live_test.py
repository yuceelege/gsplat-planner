import numpy as np
import pyrender
import torch
import matplotlib.pyplot as plt
from trimesh.transformations import euler_matrix, quaternion_matrix
from PIL import Image
import torchvision.transforms as T
import imageio                        

from network import PoseNet
from utils import load_scene_from_json, load_pcd_data, rrt, smooth_shortcuts, sample_along_path
from agent import Drone
from pid_controller import PIDController

plt.rcParams['toolbar'] = 'None'

# params
WIDTH, HEIGHT= 640, 480
FOV_Y = np.pi/4
v_ref = 1.0
SIM_DT = 0.1
# pick the occlusion map you want but first set the map at run occupancy.py and run
CACHE = 'occlusion_maps/gsplat_cache_1.pkl'
REPLAN_DIST = 2.0
# this parameter was for testing, keep true default
USE_NET_FEEDBACK = True
NOISE_STD_POS = 0.01
NOISE_STD_YAW = 0.01
#this shows the yaw, pitch,roll angles as color arrows
def draw_axes(ax, origin, R, length=1.0):
    colors = ['r','g','b']
    arrows = []
    for i in range(3):
        vec = R[:, i]
        arr = ax.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            length=length, color=colors[i], arrow_length_ratio=0.1
        )
        arrows.append(arr)
    return arrows

def plan(start, goal, pts, retries=5):
    #similar to full test code
    #create path, smmoth if possible and sample with const speed for PID
    for _ in range(retries):
        tree, raw, obs = rrt(start, goal, pts, xr=(0,20), yr=(0,20), zr=(2.8,3.2))
        if raw is not None:
            path = smooth_shortcuts(raw, obs)
            sampled = sample_along_path(path, v_ref, SIM_DT)
            return tree, path, sampled, obs
    raise RuntimeError("Failed to plan")

if __name__ == '__main__':
    # load necessary stuff for map 1 and set the network
    scene = load_scene_from_json('maps/scene_map_1.json')
    scene.ambient_light = np.array([0.5,0.5,0.5])
    camera2d = pyrender.PerspectiveCamera(yfov=FOV_Y)
    renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)
    cam_node = scene.add(camera2d, pose=np.eye(4))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = PoseNet(num_outputs=7).to(device)
    model.load_state_dict(torch.load('estimators/pose_net_1.pth', map_location=device))
    model.eval()
    # normalization transform for resnet18 imagenet weights, constants found from source cited on report
    transform = T.Compose([
        T.Pad((0,80,0,80)),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],
                    [0.229,0.224,0.225])
    ])
    #initial plan
    pts, _ = load_pcd_data(CACHE)
    start = [3,3,3]
    goal = [17,17,3]
    tree, path, sampled, obs = plan(start, goal, pts)
    #init true drone and PID
    init_tan = sampled[1] - sampled[0]
    theta0 = np.arctan2(init_tan[1], init_tan[0])
    drone = Drone([*start, v_ref, 0.0, theta0])
    pid = PIDController(dt=SIM_DT)
    last_pred_pos = None
    last_yaw_meas = theta0
    #plotting setup
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    plt.ion()
    #writer = imageio.get_writer('simulation.mp4', fps=int(1/SIM_DT))
    def draw_static():
        ax2.cla()
        ax2.scatter(pts[:,0], pts[:,1], pts[:,2], color=[0.5,0.5,0.5], s=1, alpha=0.6)
        for n in tree:
            if n.parent:
                xs, ys, zs = zip(n.pos, n.parent.pos)
                ax2.plot(xs, ys, zs, c='gray', lw=0.5)
        ax2.plot(*np.array(path).T, c='blue', lw=2)
        ax2.set(xlim=(0,20), ylim=(0,20), zlim=(0,5),
                xlabel='X', ylabel='Y', zlabel='Z')
    draw_static()
    img_plot   = ax1.imshow(np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8))
    ax1.axis('off')
    drone_dot, = ax2.plot(*start, 'o', c='orange')
    pred_dot, = ax2.plot(*start, 'o', c='magenta')
    real_axes, est_axes = [], []
    i = 0
    while i < len(sampled)-1:
        #render from true state
        pos_true = drone.state[:3]
        yaw_true = drone.state[5]
        T2 = euler_matrix(np.pi/2, 0, yaw_true - np.pi/2, axes='sxyz')
        T2[:3,3] = pos_true
        scene.set_pose(cam_node, T2)
        img, _ = renderer.render(scene)
        img_plot.set_data(img)
        #writer.append_data(img)
        #inference from rendered image with pose net
        inp = transform(Image.fromarray(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp).cpu().squeeze().numpy()
        pred_pos, pred_q = out[:3], out[3:7]
        R_pred = quaternion_matrix(pred_q)[:3,:3]
        #compute yaw from quaternion
        R_est = T2[:3,:3] @ R_pred @ euler_matrix(
                     np.pi/2, 0, -np.pi/2, axes='sxyz'
                 )[:3,:3]
        yaw_meas = np.arctan2(R_est[1,0], R_est[0,0])
        #dead-reckon if needed
        if last_pred_pos is None:
            last_pred_pos = pred_pos.copy()
        else:
            vx = drone.state[3] * np.cos(last_yaw_meas)
            vy = drone.state[3] * np.sin(last_yaw_meas)
            vz = drone.state[4]
            ins_pos = last_pred_pos + np.array([vx, vy, vz]) * SIM_DT
            #adjust alpha for your preference
            alpha = 0.55
            last_pred_pos = alpha * ins_pos + (1 - alpha) * pred_pos
        last_yaw_meas = yaw_meas
        #build feedback state: use NN-based pos and yaw, true vel (coming from IMU info)
        state_fb = drone.state.copy()
        if USE_NET_FEEDBACK:
            state_fb[0:3] = last_pred_pos
            state_fb[5] = yaw_meas
        #compute control and update true state
        pref = sampled[i]
        tangent = sampled[i+1] - sampled[i]
        u = pid.compute_control(state_fb, pref, tangent/SIM_DT, tangent)
        drone.update_state(u, SIM_DT)
        # noise injection
        drone.state[0:3] += np.random.normal(scale=NOISE_STD_POS, size=3)
        drone.state[5] += np.random.normal(scale=NOISE_STD_YAW)
        # plot update
        for a in real_axes + est_axes:
            a.remove()
        real_axes.clear(); est_axes.clear()

        R_real = euler_matrix(0,0,drone.state[5],axes='sxyz')[:3,:3]
        real_axes.extend(draw_axes(ax2, pos_true, R_real,   length=0.8))
        est_axes.extend(draw_axes(ax2, last_pred_pos, R_est, length=0.8))

        drone_dot.set_data([pos_true[0]], [pos_true[1]])
        drone_dot.set_3d_properties([pos_true[2]])
        pred_dot.set_data([last_pred_pos[0]], [last_pred_pos[1]])
        pred_dot.set_3d_properties([last_pred_pos[2]])
        plt.pause(SIM_DT)
        # replanning when too much away from ref traj (can happen if pose est outputs bad val)
        if np.linalg.norm(pos_true - pref) > REPLAN_DIST:
            start = pos_true.tolist()
            tree, path, sampled, obs = plan(start, goal, pts)
            last_pred_pos = None
            draw_static()
            real_axes.clear(); est_axes.clear()
            drone_dot, = ax2.plot(*pos_true, 'o', c='orange')
            pred_dot,  = ax2.plot(*start,  'o', c='magenta')
            i = 0
            continue
        i += 1
    renderer.delete()
    #writer.close()                 
    plt.ioff()
    plt.show()
