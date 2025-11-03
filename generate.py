import json
import numpy as np
import pyrender
import matplotlib.pyplot as plt
from trimesh.transformations import euler_matrix
import argparse
import os
import csv
from utils import *
#this portion is dedicated to generate dataset for training gsplat

#checks x,y,z of pose inside obj
def valid_pose(eye, objects):
    for obj in objects:
        if point_inside_object(eye, obj):
            return False
    return True

#defines the room border,makes sure camera is not inside objects and samples from certain directions
def sample_valid_camera_pose(scene_data, max_attempts=1000):
    objects = scene_data["objects"]
    attempts = 0
    while attempts < max_attempts:
        x = np.random.uniform(3, 17)
        y = np.random.uniform(3, 17)
        z = np.random.uniform(1.5, 4.5)
        eye = np.array([x, y, z])
        if valid_pose(eye, objects):
            # distribution subject to change by adjusting these
            roll = (np.pi / 2) + np.random.uniform(-0.1, 0.1)
            pitch = 0.0 + np.random.uniform(-0.1, 0.1)
            yaw = np.random.uniform(0, 2 * np.pi)
            T = euler_matrix(roll, pitch, yaw, axes='sxyz')
            T[:3, 3] = eye
            return eye, T, (roll, pitch, yaw)
        attempts += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate (pose,Img) dataset")
    #sets the dataset sizwe
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    args = parser.parse_args()
    
    dataset_dir = "dataset"
    #choose your map (1,2,3)
    with open("maps/scene_map_3.json", 'r') as f:
        scene_data = json.load(f)
    scene = load_scene_from_json("maps/scene_map_3.json")
    scene.ambient_light = np.array([0.5, 0.5, 0.5])
    fov = np.pi / 4
    cam = pyrender.PerspectiveCamera(yfov=fov)
    #directory adjustments
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    poses_file = os.path.join(dataset_dir, "poses.csv")
    dataset_rows = []
    frames = []
    # our images are 640x480
    width = 640
    height = 480
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    
    horizontal_fov = 2 * np.arctan((width / height) * np.tan(fov / 2))
    fl_y = (height / 2) / np.tan(fov / 2)
    fl_x = (width / 2) / np.tan(horizontal_fov / 2)
    cx = width / 2.0
    cy = height / 2.0
    
    #sample, pose and render with that pose and save the img
    for i in range(args.size):
        eye, cam_pose, euler_angles = sample_valid_camera_pose(scene_data)
        tx, ty, tz = eye
        rx, ry, rz = euler_angles
        dataset_rows.append([i, tx, ty, tz, rx, ry, rz])
        image_filename = f"{i:04d}.png"
        frames.append({"file_path": os.path.join("images", image_filename),"transform_matrix": cam_pose.tolist(),"fl_x": fl_x,"fl_y": fl_y,"cx": cx,"cy": cy,"w": width,"h": height})
        cam_node = scene.add(cam, pose=cam_pose)
        color, depth = renderer.render(scene)
        image_path = os.path.join(images_dir, image_filename)
        plt.imsave(image_path, color)
        scene.remove_node(cam_node)
        print(f"Rendered image {i} at pose {eye}")
    renderer.delete()
    #we save the poses in a csv file in COLMAP comptaible way.
    with open(poses_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_number", "tx", "ty", "tz", "rx", "ry", "rz"])
        writer.writerows(dataset_rows)
    
    transforms_data = {
        "camera_angle_x": horizontal_fov,
        "frames": frames
    }
    transforms_file = os.path.join(dataset_dir, "transforms.json")
    with open(transforms_file, "w") as f:
        json.dump(transforms_data, f, indent=4)
    
    print(f"Dataset of {args.size} images and poses saved in '{dataset_dir}'")
