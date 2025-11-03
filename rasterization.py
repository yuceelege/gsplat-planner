import torch

def quaternion_to_rotation_matrix(quats):
    quats = quats / quats.norm(dim=1, keepdim=True)
    w, x, y, z = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    N = quats.size(0)
    R = torch.empty(N,3,3,device=quats.device,dtype=quats.dtype)
    #standard formula for the transformation
    R[:,0,0] = 1 - 2*(y**2 + z**2)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)
    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x**2 + z**2)
    R[:,1,2] = 2*(y*z - x*w)
    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x**2 + y**2)
    return R
#resp for computing the bounding rects for gaussians based on image size
def get_rect(coords, radii, width, height):
    mn = coords - radii[:, None]
    mx = coords + radii[:, None]
    mn[...,0].clamp_(0, width-1); mn[...,1].clamp_(0, height-1)
    mx[...,0].clamp_(0, width-1); mx[...,1].clamp_(0, height-1)
    return mn, mx
def render(means2D, cov2D, radii, color, opacity, depths, W, H, tile_size):
    #generate grid for pixsels
    pix = torch.stack(
        torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'),
        dim=-1
    ).to(depths.device)
    rect = get_rect(means2D, radii, W, H)
    #color depth and alpha belending tensors
    rc = torch.zeros(*pix.shape[:2], color.shape[-1], device=pix.device)
    rd = torch.zeros(*pix.shape[:2], 1, device=pix.device)
    ra = torch.zeros(*pix.shape[:2], 1, device=pix.device)

    for h in range(0, H, tile_size):
        #this part is for finding the overlapping gaussians as you look along a tile (like a ray)
        for w in range(0, W, tile_size):
            tlx = rect[0][...,0].clip(min=w)
            tly = rect[0][...,1].clip(min=h)
            brx = rect[1][...,0].clip(max=w+tile_size-1)
            bry = rect[1][...,1].clip(max=h+tile_size-1)
            mask = (brx > tlx) & (bry > tly)
            if not mask.any(): 
                continue
            # this is the important part, using depth we look from top to bottom by masking and collect the gaussians
            #for each tile
            tc = pix[h:h+tile_size, w:w+tile_size].flatten(0,1)
            sd, idx = torch.sort(depths[mask])
            sm = means2D[mask][idx]
            sc = cov2D[mask][idx]
            so = opacity[mask][idx].unsqueeze(1)
            sf = color[mask][idx]
            # compute gaussian weights
            dx = tc[:,None,:] - sm[None,:,:]
            invc = torch.linalg.inv(sc)
            gw = torch.exp(-0.5*(
                dx[:,:,0]**2 * invc[:,0,0] +
                dx[:,:,1]**2 * invc[:,1,1] +
                dx[:,:,0]*dx[:,:,1] * (invc[:,0,1] + invc[:,1,0])
            ))
            #alpha blending part
            a = (gw[...,None] * so).clamp(max=0.99)
            T = torch.cat([torch.ones_like(a[:,:1]), 1 - a[:,:-1]], dim=1).cumprod(dim=1)
            aa = (a * T).sum(1)
            tc_col = (T * a * sf.unsqueeze(0)).sum(1)
            td = ((T * a) * sd.unsqueeze(-1)).sum(1)
            #color back each tile with the blending result
            h2 = min(h + tile_size, H)
            w2 = min(w + tile_size, W)
            rc[h:h2, w:w2] = tc_col.reshape(h2-h, w2-w, -1)
            rd[h:h2, w:w2] = td.reshape(h2-h, w2-w, -1)
            ra[h:h2, w:w2] = aa.reshape(h2-h, w2-w, -1)

    return {
        'render': rc, 'depth': rd, 'alpha': ra,
        'visibility_filter': radii > 0, 'radii': radii
    }
#mapping from 3d to cam space using hom coord space
def camera_transform(means, viewmat, near, far):
    N = means.size(0)
    ones = torch.ones(N,1,device=means.device,dtype=means.dtype)
    hom = torch.cat([means, ones], dim=1)
    cam_hom_coord = (viewmat @ hom.T).T
    cam_coord = cam_hom_coord[:,:3] / cam_hom_coord[:,3:4]
    mask = (cam_coord[:,2] > near) & (cam_coord[:,2] < far)
    return cam_coord, mask
#computing cov matrix in wc
def world_covariance(quats, scales):
    Rg = quaternion_to_rotation_matrix(quats)
    S = torch.diag_embed(scales).to(quats.device)
    M = Rg @ S
    return M @ M.transpose(1,2)
#projection of cov mat to cam space
def camera_covariance(world_cov, viewmat):
    R = viewmat[:3,:3].unsqueeze(0).expand_as(world_cov)
    return R @ world_cov @ R.transpose(1,2)
#projection to pixels
def project_to_image(mc, K):
    proj = (K @ mc.T).T
    return proj[:,:2] / proj[:,2:3]
#converting the 3d gaussian cov to 2d gaussian cov requires a linearization around the given pose
def compute_jacobian(cam_coord, fx, fy, width, height):
    x, y, z = cam_coord[:,0], cam_coord[:,1], cam_coord[:,2]
    limx = 1.3 * (0.5 * width / fx)
    limy = 1.3 * (0.5 * height / fy)
    tx = z * torch.min(limx, torch.max(-limx, x / z))
    ty = z * torch.min(limy, torch.max(-limy, y / z))
    J = torch.zeros(cam_coord.size(0), 2, 3, device=cam_coord.device, dtype=cam_coord.dtype)
    J[:,0,0] = fx / z
    J[:,0,2] = -fx * tx / (z*z)
    J[:,1,1] = fy / z
    J[:,1,2] = -fy * ty / (z*z)
    return J
#computes the 2d covs with the jacobian
def gaussian_2d_params(J, cam_cov, eps2d):
    c2 = J @ cam_cov @ J.transpose(1,2)
    d0 = c2[:,0,0]*c2[:,1,1] - c2[:,0,1]*c2[:,1,0]
    c2 = c2.clone()
    c2[:,0,0] += eps2d
    c2[:,1,1] += eps2d
    db = c2[:,0,0]*c2[:,1,1] - c2[:,0,1]*c2[:,1,0]
    return c2, d0, db
#takes covariance and computes the radii (cov is essentially an ellipsoid)
def compute_radii(d0, c2):
    b = 0.5 * (c2[:,0,0] + c2[:,1,1])
    disc = torch.clamp(b*b - d0, min=0.1).sqrt()
    v1 = b + disc
    v2 = b - disc
    return torch.ceil(3 * torch.sqrt(torch.max(v1, v2)))
#we apply everything we described above in sequential order
def rasterize_gaussians(
    means, quats, scales, opacities, colors,
    viewmats, Ks, width, height,
    near_plane=0.01, far_plane=1e10,
    tile_size=8, eps2d=0.3, radius_clip=0.0
):
    device, dtype = means.device, means.dtype
    viewmat, K = viewmats[0], Ks[0]
    fx, fy = K[0,0], K[1,1]
    mc, mask = camera_transform(means, viewmat, near_plane, far_plane)
    world_cov = world_covariance(quats, scales)
    cam_cov   = camera_covariance(world_cov, viewmat)
    #project to 2D
    m2 = project_to_image(mc, K)
    J = compute_jacobian(mc, fx, fy, width, height)
    c2, d0, db = gaussian_2d_params(J, cam_cov, eps2d)
    mask &= (db > 0)
    r = compute_radii(d0, c2)
    mask &= (r > radius_clip)
    sel = lambda arr: arr[mask]
    m2_f, c2_f, r_f = sel(m2), sel(c2), sel(r)
    col_f  = sel(colors)
    op_f   = sel(opacities)
    depth_f= mc[mask, 2]
    res = render(m2_f, c2_f, r_f, col_f, op_f, depth_f, width, height, tile_size)
    res['overall_mask'] = mask
    return res
