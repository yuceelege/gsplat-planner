import numpy as np
import trimesh
import pyrender
from trimesh.transformations import euler_matrix
import json
import pickle
import numpy as np
from scipy.spatial import cKDTree

#class for RRT nodes
class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent

#load the occlusion map data
def load_pcd_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    pts = data['fm'][data['outlier_idx']]
    colors = data['colors']
    return pts, colors

#for checking if two points line sigment touches an object with sampling + kdtree (required for rrt)
def collision_free(p1, p2, tree, clearance=0.4, samples=10):
    for t in np.linspace(0,1,samples):
        if tree.query(p1 + t*(p2-p1))[0] < clearance:
            return False
    return True

def rrt(start, goal, pts, xr, yr, zr,max_iter=1400, step=1.1, th=1.5, obstacle_bloat=0.6, goal_sample_rate=0.15):
    tree = [Node(np.array(start))]
    obs = cKDTree(pts)
    #we are doing the algo for kdtree because there are thousands of gaussian splats
    for _ in range(max_iter):
        #sample points
        if np.random.rand() < goal_sample_rate:
            sample = np.array(goal)
        else:
            sample = np.array([np.random.uniform(*xr),np.random.uniform(*yr),np.random.uniform(*zr)])
        nearest = min(tree, key=lambda n: np.linalg.norm(n.pos - sample))
        dv = sample - nearest.pos
        d = np.linalg.norm(dv)
        if d == 0: continue
        newp = nearest.pos + dv/d * step
        if not collision_free(nearest.pos, newp, obs, clearance=obstacle_bloat):
            continue
        node = Node(newp, nearest)
        tree.append(node)
        #find the closest node to sampled point, add the point if no collision to the tree
        if np.linalg.norm(newp - goal) < th:
            #if we arrive at the goal, backtrack and get the rrt path
            end = Node(np.array(goal), node)
            tree.append(end)
            path = []
            it = end
            while it:
                path.append(it.pos)
                it = it.parent
            return tree, path[::-1], obs
    return tree, None, obs

#you can always pick any two points in path and if no obstackle in between
#smooth and get shortcut
def smooth_shortcuts(path, obs, its=100, mi=5.0, mx=10.0):
    pts = path.copy()
    for _ in range(its):
        N = len(pts)
        if N < 3: break
        i, j = sorted(np.random.choice(N, 2, False))
        if j <= i+1: continue
        L = np.linalg.norm(pts[j] - pts[i])
        if L < mi or L > mx: continue
        if collision_free(pts[i], pts[j], obs):
            pts = pts[:i+1] + pts[j:]
    return pts

#sampling after smooth path with constant speed for PID
def sample_along_path(path, v, dt):
    pts = np.array(path)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.hstack(([0], np.cumsum(d)))
    total = cum[-1]
    step = v * dt
    N = int(total / step)
    ds = np.linspace(0, total, N+1)
    samples = []
    for sd in ds:
        idx = np.searchsorted(cum, sd) - 1
        idx = np.clip(idx, 0, len(d)-1)
        seg = d[idx]
        t = (sd - cum[idx]) / seg if seg > 0 else 0
        samples.append(pts[idx] + t * (pts[idx+1] - pts[idx]))
    return np.array(samples)
#for plotting
def draw_axes(ax, origin, R, length=1.0):
    colors = ['r','g','b']
    arrows = []
    for i in range(3):
        vec = R[:,i]
        arr = ax.quiver(origin[0], origin[1], origin[2], vec[0], vec[1], vec[2], length=length, color=colors[i], arrow_length_ratio=0.1)
        arrows.append(arr)
    return arrows
#auxillarly functions for environment generation
def create_box(extents, translation, color):
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(translation)
    box.visual.face_colors = color
    return pyrender.Mesh.from_trimesh(box, smooth=False)

def create_rotated_box(extents, translation, color, angle):
    box = trimesh.creation.box(extents=extents)
    T = euler_matrix(0, 0, np.radians(angle), axes='sxyz')
    box.apply_transform(T)
    box.apply_translation(translation)
    box.visual.face_colors = color
    return pyrender.Mesh.from_trimesh(box, smooth=False)

def create_wall(extents, translation, color):
    wall = trimesh.creation.box(extents=extents)
    wall.apply_translation(translation)
    wall.invert()
    wall.visual.face_colors = color
    return pyrender.Mesh.from_trimesh(wall, smooth=False)

def create_pillar(radius, height, translation, color):
    pillar = trimesh.creation.cylinder(radius=radius, height=height)
    pillar.apply_translation(translation)
    pillar.visual.face_colors = color
    return pyrender.Mesh.from_trimesh(pillar, smooth=False)

def create_sphere(radius, translation, color):
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.apply_translation(translation)
    sphere.visual.face_colors = color
    return pyrender.Mesh.from_trimesh(sphere, smooth=False)

def create_cone(height, radius, translation, color):
    cone = trimesh.creation.cone(height=height, radius=radius)
    cone.apply_translation(translation)
    cone.visual.face_colors = color
    return pyrender.Mesh.from_trimesh(cone, smooth=False)

def create_torus(major_radius, minor_radius, translation, color):
    torus = trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius)
    torus.apply_translation(translation)
    torus.visual.face_colors = color
    return pyrender.Mesh.from_trimesh(torus, smooth=False)
#checkes if sampled pose inside object (we will avoid this)
def point_inside_object(eye, obj):
    typ = obj["type"]
    params = obj["parameters"]
    eye = np.array(eye)
    if typ in ["box", "wall"]:
        extents = np.array(params["extents"])
        center = np.array(params["translation"])
        if np.all(np.abs(eye - center) <= extents / 2.0):
            return True
    elif typ == "rotated_box":
        extents = np.array(params["extents"])
        center = np.array(params["translation"])
        r = np.linalg.norm(extents) / 2.0
        if np.linalg.norm(eye - center) < r:
            return True
    elif typ == "pillar":
        radius = params["radius"]
        height = params["height"]
        center = np.array(params["translation"])
        if np.linalg.norm(eye[:2] - center[:2]) < radius and (eye[2] >= center[2] - height/2 and eye[2] <= center[2] + height/2):
            return True
    elif typ == "sphere":
        radius = params["radius"]
        center = np.array(params["translation"])
        if np.linalg.norm(eye - center) < radius:
            return True
    elif typ == "cone":
        radius = params["radius"]
        center = np.array(params["translation"])
        if np.linalg.norm(eye - center) < radius:
            return True
    elif typ == "torus":
        center = np.array(params["translation"])
        threshold = params["major_radius"] + params["minor_radius"]
        if np.linalg.norm(eye - center) < threshold:
            return True
    return False
#for loading json file
def load_scene_from_json(json_file):
    scene = pyrender.Scene()
    with open(json_file, 'r') as f:
        data = json.load(f)
    for obj in data["objects"]:
        typ = obj["type"]
        params = obj["parameters"]
        if typ == "box":
            mesh = create_box(params["extents"], params["translation"], params["color"])
        elif typ == "rotated_box":
            mesh = create_rotated_box(params["extents"], params["translation"], params["color"], params["angle"])
        elif typ == "wall":
            mesh = create_wall(params["extents"], params["translation"], params["color"])
        elif typ == "pillar":
            mesh = create_pillar(params["radius"], params["height"], params["translation"], params["color"])
        elif typ == "sphere":
            mesh = create_sphere(params["radius"], params["translation"], params["color"])
        elif typ == "cone":
            mesh = create_cone(params["height"], params["radius"], params["translation"], params["color"])
        elif typ == "torus":
            mesh = create_torus(params["major_radius"], params["minor_radius"], params["translation"], params["color"])
        else:
            continue
        scene.add(mesh)
    return scene

