import functools 
import numpy as np 
import jax 
from jax import numpy as jnp 


def _gather_nd_single(params, indices):
  idx = tuple(np.moveaxis(indices, -1, 0))
  return params[idx]

def gather_nd(  # pylint: disable=unused-argument
    params,
    indices,
    batch_dims=0,
    name=None):
  """gather_nd."""
  
  gather_nd_ = _gather_nd_single

  gather_nd_ = functools.reduce(
        lambda g, f: f(g), [jax.vmap] * int(batch_dims),
        gather_nd_
    )
  return gather_nd_(params, indices)


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = x.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = jnp.arange(0, batch_size)
    batch_idx = jnp.reshape(batch_idx, (batch_size, 1, 1))
    b = jnp.tile(batch_idx, (1, height, width))

    indices = jnp.stack([b, y, x], 3)

    return gather_nd(img, indices)


def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.

    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.

    - width: desired width of grid/output. Used
      to downsample or upsample.

    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.

    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.

    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = theta.shape[0]

    # create normalized 2D grid
    x = np.linspace(-1.0, 1.0, width)
    y = np.linspace(-1.0, 1.0, height)
    x_t, y_t = np.meshgrid(x, y)

    # flatten
    x_t_flat = np.reshape(x_t, [-1])
    y_t_flat = np.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = np.ones_like(x_t_flat)
    sampling_grid = np.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = np.expand_dims(sampling_grid, axis=0)
    sampling_grid = np.tile(sampling_grid, np.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = theta.astype(np.float32)
    sampling_grid = sampling_grid.astype(np.float32)

    # transform the sampling grid - batch multiply
    batch_grids = np.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = np.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = img.shape[1]
    W = img.shape[2]
    max_y = int(H - 1)
    max_x = int(W - 1)
    zero = np.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x = 0.5 * ((x + 1.0) * float(max_x-1))
    y = 0.5 * ((y + 1.0) * float(max_y-1))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = x0.astype(np.float32)
    x1 = x1.astype(np.float32)
    y0 = y0.astype(np.float32)
    y1 = y1.astype(np.float32)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = np.expand_dims(wa, axis=3)
    wb = np.expand_dims(wb, axis=3)
    wc = np.expand_dims(wc, axis=3)
    wd = np.expand_dims(wd, axis=3)

    # compute output
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return out


def get_grid(pose, grid_h, grid_w):
    """
    Input:
        `pose` np.array(bs, 3)
        `grid_h, grid_w`
        
    Output:
        `rot_grid` FloatTensor(bs, 2, grid_h, grid_w)
        `trans_grid` FloatTensor(bs, 2, grid_h, grid_w)

    """
    
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.shape[0]
    t = t * np.pi / 180.
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    theta11 = np.stack([cos_t, -sin_t, np.zeros_like(t)], 1)
    theta12 = np.stack([sin_t, cos_t, np.zeros_like(t)], 1)
    theta1 = np.stack([theta11, theta12], 1)

    theta21 = np.stack([np.ones_like(x), -np.zeros_like(x), x], 1)
    theta22 = np.stack([np.zeros_like(x), np.ones_like(x), y], 1)
    theta2 = np.stack([theta21, theta22], 1)

    rot_grid = affine_grid_generator(grid_h, grid_w, theta1)
    trans_grid = affine_grid_generator(grid_h, grid_w, theta2)

    return rot_grid, trans_grid


def get_local_map_boundaries(agent_loc, local_sizes, full_sizes, args):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1.gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]


def init_map_and_pose(args):
    num_scenes = args.num_processes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), int(full_h / args.global_downscaling)
    # Initializing full and local map
    full_map = np.zeros((num_scenes, full_w, full_h, 4))
    local_map = np.zeros((num_scenes, local_w, local_h, 4))

    # Initial full and local pose
    full_pose = np.zeros((num_scenes, 3))
    local_pose = np.zeros((num_scenes, 3))

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

    locs = full_pose
    planner_pose_inputs[:, :3] = locs
    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2, 2:] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c), (local_w, local_h), (full_w, full_h), args)

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                        lmb[e][0] * args.map_resolution / 100.0, 0.]

    for e in range(num_scenes):
        local_map[e] = full_map[e, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3], :]
        local_pose[e] = full_pose[e] - origins[e]
    
    return full_map, local_map, full_pose, local_pose, origins, lmb, planner_pose_inputs


def action2goal(a, local_w, local_h, action_width, action_height):
    global_goals = [[
        int((ai % action_width + 0.5) / action_width * local_w),
        int((ai // action_width + 0.5) / action_height * local_h)
        ] for ai in a ]
    return global_goals


def pose2plane(pose, shape):
    return jnp.broadcast_to(pose, shape)