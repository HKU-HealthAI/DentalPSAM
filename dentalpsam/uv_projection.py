"""Historical three-view UV projection used by DentalPSAM.

The numerical operations are preserved from the server source so existing
rendered views and ``info/*.npz`` files remain reproducible.  New runs should
use :mod:`prepare_uv_views`, which supplies a fixed mesh list, provenance
manifest, and source-data overwrite guards.
"""

import open3d as o3d
import numpy as np
import copy
import cv2
import os


def process_all_files(origin_dir, label_dir, save_dir):

    metrics_dict = {}
    for file in os.listdir(label_dir):
        if file.endswith(".ply"):
            mesh_name = file.split(".")[0]
            print(f"Processing {mesh_name}...")
            origin_file_path = f"{origin_dir}/{mesh_name}.ply"
            label_file_path = f"{label_dir}/{mesh_name}.ply"
            raw_origin_mesh = o3d.io.read_triangle_mesh(origin_file_path)
            raw_label_mesh = o3d.io.read_triangle_mesh(label_file_path)

            iou1, dice1, iou0, dice0 = process_single_mesh(
                raw_origin_mesh, raw_label_mesh, save_dir, mesh_name
            )

            if iou1 is not None:
                metrics_dict[mesh_name] = [iou1, iou0, dice1, dice0]
            else:
                print(
                    f"Warning: Failed to process {mesh_name}, skipping metrics calculation"
                )

    successful_metrics = [
        m
        for m in metrics_dict.values()
        if m is not None and all(v is not None for v in m)
    ]

    if successful_metrics:
        iou1_mean = np.mean([m[0] for m in successful_metrics])
        iou0_mean = np.mean([m[1] for m in successful_metrics])
        dice1_mean = np.mean([m[2] for m in successful_metrics])
        dice0_mean = np.mean([m[3] for m in successful_metrics])
        print(
            f"Successfully processed: {len(successful_metrics)}/{len(metrics_dict)} files"
        )
        print(f"Plaque IoU: {iou1_mean:.3f}, Plaque Dice: {dice1_mean:.3f}")
        print(f"Non-Plaque IoU: {iou0_mean:.3f}, Non-Plaque Dice: {dice0_mean:.3f}")
    else:
        print("No files were successfully processed!")
        iou1_mean = iou0_mean = dice1_mean = dice0_mean = 0.0

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Plaque IoU: {iou1_mean:.3f}, Plaque Dice: {dice1_mean:.3f}\n")
        f.write(f"Non-Plaque IoU: {iou0_mean:.3f}, Non-Plaque Dice: {dice0_mean:.3f}\n")
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")


def process_single_mesh(raw_origin_mesh, raw_label_mesh, save_dir, mesh_name):
    """Render one mesh and write PNG/NPZ metadata to a caller-owned directory.

    Mesh objects are transformed in memory; source PLY files are never saved.
    Returned map-back statistics diagnose rasterization of the labels, not
    model segmentation performance. Use evaluate_dentalpsam.py for the latter.
    """

    raw_origin_mesh.remove_duplicated_triangles()
    raw_label_mesh.remove_duplicated_triangles()

    raw_origin_mesh.translate(-raw_origin_mesh.get_center())
    origin_mesh, edp1, edp2 = recentre_mesh(raw_origin_mesh)

    raw_label_mesh.translate(-raw_label_mesh.get_center())
    label_mesh, _, _ = recentre_mesh(raw_label_mesh)

    up_triangles, unsampled_mesh = extract_up_triangles(origin_mesh)

    in_triangles, out_triangles = separate_io_triangles(unsampled_mesh)

    origin_mesh_c = copy.deepcopy(origin_mesh)
    origin_mesh_c.translate(-origin_mesh_c.get_center())
    vertices = np.asarray(origin_mesh_c.vertices)

    uv_up, vert_depth_up = UVmap_planar(vertices)
    uv_in, vert_depth_in = UVmap_cylindrical(vertices, if_inward=True)
    uv_out, vert_depth_out = UVmap_cylindrical(vertices, if_inward=False)

    uv_norm_up = normalize_uv_coords(uv_up, (-45, 45), (-35, 25))
    uv_norm_in = normalize_uv_coords(uv_in, (0, 2 * np.pi), (-8, 8), with_nl=True)
    uv_norm_out = normalize_uv_coords(uv_out, (0, 2 * np.pi), (-9, 7), with_nl=True)

    up_vert_idx = np.unique(up_triangles.flatten())
    in_vert_idx = np.unique(in_triangles.flatten())
    out_vert_idx = np.unique(out_triangles.flatten())

    vertices_originRGB_raw = np.asarray(origin_mesh.vertex_colors)
    vertices_labelRGB_raw = np.asarray(label_mesh.vertex_colors)

    vertices_originRGB = (vertices_originRGB_raw * 255).astype(np.uint8)
    vertices_labelRGB = ((1 - vertices_labelRGB_raw) * 255).astype(np.uint8)

    img_origin_up, img_label_up, uv_pixel_up, sorted_tri_up = rasterize_uv_to_image(
        uv_norm_up,
        vert_depth_up,
        up_triangles,
        vertices_originRGB,
        vertices_labelRGB,
        px=256,
        is_io=False,
    )
    img_origin_in, img_label_in, uv_pixel_in, sorted_tri_in = rasterize_uv_to_image(
        uv_norm_in,
        vert_depth_in,
        in_triangles,
        vertices_originRGB,
        vertices_labelRGB,
        px=256,
        is_io=True,
    )
    img_origin_out, img_label_out, uv_pixel_out, sorted_tri_out = rasterize_uv_to_image(
        uv_norm_out,
        vert_depth_out,
        out_triangles,
        vertices_originRGB,
        vertices_labelRGB,
        px=256,
        is_io=True,
    )

    if (
        img_origin_up is None
        or img_label_up is None
        or img_origin_in is None
        or img_label_in is None
        or img_origin_out is None
        or img_label_out is None
    ):
        print(f"Error: Failed to generate images for {mesh_name}")
        return None, None, None, None

    if (
        img_origin_up is not None
        and img_label_up is not None
        and img_origin_in is not None
        and img_label_in is not None
        and img_origin_out is not None
        and img_label_out is not None
    ):

        all_empty = True
        for img in [
            img_origin_up,
            img_label_up,
            img_origin_in,
            img_label_in,
            img_origin_out,
            img_label_out,
        ]:
            if np.any(img != 0):
                all_empty = False
                break

        if all_empty:
            print(
                f"Warning: All generated images are empty (no valid triangles) for {mesh_name}"
            )

    save_origin_path = os.path.join(save_dir, "origin")
    save_label_path = os.path.join(save_dir, "label")

    if not os.path.exists(os.path.join(save_dir, "info")):
        os.makedirs(os.path.join(save_dir, "info"))
    if not os.path.exists(save_origin_path):
        os.makedirs(save_origin_path)
    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)

    for idx, (img_origin_xx, img_label_xx) in enumerate(
        zip(
            [img_origin_up, img_origin_in, img_origin_out],
            [img_label_up, img_label_in, img_label_out],
        )
    ):
        origin_bgr = cv2.cvtColor(img_origin_xx, cv2.COLOR_RGB2BGR)
        label_gray = cv2.cvtColor(img_label_xx, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(
            os.path.join(save_origin_path, f"{mesh_name}_{idx}.png"), origin_bgr
        )
        cv2.imwrite(os.path.join(save_label_path, f"{mesh_name}_{idx}.png"), label_gray)

    reconst_info = {
        "uvpx_up": uv_pixel_up,
        "uvpx_in": uv_pixel_in,
        "uvpx_out": uv_pixel_out,
        "tri_up": sorted_tri_up,
        "tri_in": sorted_tri_in,
        "tri_out": sorted_tri_out,
    }

    np.savez_compressed(
        os.path.join(save_dir, "info", f"{mesh_name}.npz"), **reconst_info
    )

    pred_img_label_up = img_label_up / 255
    pred_img_label_in = img_label_in / 255
    pred_img_label_out = img_label_out / 255

    if len(sorted_tri_up) == 0 or len(uv_pixel_up) == 0:
        print(
            f"Warning: No valid triangles for Up region, skipping metrics calculation"
        )
        return None, None, None, None

    if len(sorted_tri_in) == 0 or len(uv_pixel_in) == 0:
        print(
            f"Warning: No valid triangles for In region, skipping metrics calculation"
        )
        return None, None, None, None

    if len(sorted_tri_out) == 0 or len(uv_pixel_out) == 0:
        print(
            f"Warning: No valid triangles for Out region, skipping metrics calculation"
        )
        return None, None, None, None

    tri_uvpx_up = get_tri_center_uv(sorted_tri_up, uv_pixel_up)
    tri_uvpx_in = get_tri_center_uv(sorted_tri_in, uv_pixel_in)
    tri_uvpx_out = get_tri_center_uv(sorted_tri_out, uv_pixel_out)

    tri_pred_label_up = get_tri_pred_label(tri_uvpx_up, pred_img_label_up)
    tri_pred_label_in = get_tri_pred_label(tri_uvpx_in, pred_img_label_in)
    tri_pred_label_out = get_tri_pred_label(tri_uvpx_out, pred_img_label_out)

    tri_GT_labelGRB_up = get_tri_RGB(sorted_tri_up, vertices_labelRGB) / 255
    tri_GT_labelGRB_in = get_tri_RGB(sorted_tri_in, vertices_labelRGB) / 255
    tri_GT_labelGRB_out = get_tri_RGB(sorted_tri_out, vertices_labelRGB) / 255

    metrics_up = compute_metrics_tri(tri_GT_labelGRB_up, tri_pred_label_up)
    metrics_in = compute_metrics_tri(tri_GT_labelGRB_in, tri_pred_label_in)
    metrics_out = compute_metrics_tri(tri_GT_labelGRB_out, tri_pred_label_out)

    iou_mean = (metrics_up[0] + metrics_in[0] + metrics_out[0]) / 3
    dice_mean = (metrics_up[1] + metrics_in[1] + metrics_out[1]) / 3
    print(f"Plaque IoU: {iou_mean:.3f}, Plaque Dice: {dice_mean:.3f}")

    metrics_up0 = compute_metrics_tri(
        tri_GT_labelGRB_up, tri_pred_label_up, is_plaque=False
    )
    metrics_in0 = compute_metrics_tri(
        tri_GT_labelGRB_in, tri_pred_label_in, is_plaque=False
    )
    metrics_out0 = compute_metrics_tri(
        tri_GT_labelGRB_out, tri_pred_label_out, is_plaque=False
    )

    iou0_mean = (metrics_up0[0] + metrics_in0[0] + metrics_out0[0]) / 3
    dice0_mean = (metrics_up0[1] + metrics_in0[1] + metrics_out0[1]) / 3
    print(f"Non-Plaque IoU: {iou0_mean:.3f}, Non-Plaque Dice: {dice0_mean:.3f}")

    return iou_mean, dice_mean, iou0_mean, dice0_mean


def find_key_points(mesh):
    vertices = np.asarray(mesh.vertices)

    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    theta = np.arctan2(z, x) + np.pi / 2
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    endpt1 = vertices[np.argmin(theta)]
    endpt2 = vertices[np.argmax(theta)]

    endpt1[1] = 0
    endpt2[1] = 0
    centre = (endpt1 + endpt2) / 2
    return endpt1, endpt2, centre


def recentre_mesh(mesh):
    mesh_rc = copy.deepcopy(mesh)

    endpt1, endpt2, centre_keypt = find_key_points(mesh)
    vertices = np.asarray(mesh.vertices)
    vertices = vertices - centre_keypt
    endpt1 = endpt1 - centre_keypt
    endpt2 = endpt2 - centre_keypt

    theta = np.arctan2(endpt1[2], endpt1[0])

    R = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    vertices = np.dot(vertices, R.T)
    endpt1 = np.dot(endpt1, R.T)
    endpt2 = np.dot(endpt2, R.T)

    assert endpt1[2] < 1e-6, f"end point 1 {endpt1} is not on x-axis"
    assert endpt2[2] < 1e-6, f"end point 2 {endpt2} is not on x-axis"

    mesh_rc.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh_rc, endpt1, endpt2


def set_difference(A, B):
    """Return the elements in A but not in B"""
    A_view = A.view([("", A.dtype)] * A.shape[1])
    B_view = B.view([("", B.dtype)] * B.shape[1])
    C_view = np.setdiff1d(A_view, B_view)
    C = C_view.view(A.dtype).reshape(-1, A.shape[1])

    return C


def keep_large_component(mesh, max_only=False, min_size=600):
    """Remove small disconnected components from the mesh
    Args:
        max_only: if True, only keep the biggest component
        min_size: only works if max_only is False,
            in which case only components with size >= min_size are kept

    Return:
        mesh: the mesh with small components removed
        triangles: the preserved triangles of the mesh
        removed_triangles: the removed triangles of the mesh"""
    triangles = np.asarray(mesh.triangles)
    components = np.array(mesh.cluster_connected_triangles()[0])
    comp_sizes = np.bincount(components)
    if max_only:
        max_comp_index = np.argmax(comp_sizes)
        large_comp_mask = components == max_comp_index

    else:
        large_comp_indices = np.where(comp_sizes >= min_size)[0]
        large_comp_mask = np.isin(components, large_comp_indices)

    preserved_triangles = triangles[large_comp_mask]
    removed_triangles = triangles[np.logical_not(large_comp_mask)]
    mesh.triangles = o3d.utility.Vector3iVector(preserved_triangles)

    return mesh, preserved_triangles, removed_triangles


def extract_up_triangles(mesh, xpos_min=10, xpos_max=-10, ypos_min=-5, ynorm_min=0.5):
    """All vertices are preserved.
    Triangles with all 3 vertices upward facing are preserved"""

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    assert (
        xpos_min > xpos_max
    ), "xpos_min must be greater than xpos_max: to avoid middle region"

    up_vert_ypos_mask = (vertices[:, 1]) > ypos_min
    up_vert_norm_mask = normals[:, 1] > ynorm_min
    up_vert_mask = up_vert_ypos_mask & up_vert_norm_mask

    up_mesh = copy.deepcopy(mesh)
    up_triangles_mask = np.all(up_vert_mask[triangles], axis=1)
    up_triangles = triangles[up_triangles_mask]
    up_mesh.triangles = o3d.utility.Vector3iVector(up_triangles)

    up_mesh, up_triangles, _ = keep_large_component(up_mesh, min_size=600)
    unsampled_triangles = set_difference(triangles, up_triangles)
    unsampled_mesh = copy.deepcopy(mesh)
    unsampled_mesh.triangles = o3d.utility.Vector3iVector(unsampled_triangles)

    unsampled_mesh, unsampled_triangles, _ = keep_large_component(
        unsampled_mesh, min_size=500
    )

    up_triangles = set_difference(triangles, unsampled_triangles)
    assert (
        abs(len(up_triangles) + len(unsampled_triangles) - len(triangles)) <= 1
    ), "Triangles are not correctly separated"

    return up_triangles, unsampled_mesh


def separate_io_triangles(mesh):
    """Separate the original mesh triangles into
    inward-facing and outward-facing triangles relative to the origin

    All vertices are preserved."""

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    viewpt1 = np.mean(vertices, axis=0)
    vertices_vp1 = vertices - viewpt1
    viewpt2 = viewpt1 / 2
    vertices_vp2 = vertices - viewpt2

    normals = np.asarray(mesh.vertex_normals)

    pos_mask1 = vertices[:, 2] > viewpt1[2]
    in_vert_mask1 = np.logical_and(
        np.sum(normals * vertices_vp1, axis=1) < 0, pos_mask1
    )
    pos_mask2 = np.logical_and(viewpt2[2] < vertices[:, 2], vertices[:, 2] < viewpt1[2])
    in_vert_mask2 = np.logical_and(
        np.sum(normals * vertices_vp2, axis=1) < 0, pos_mask2
    )
    pos_mask3 = vertices[:, 2] < viewpt2[2]
    in_vert_mask3 = np.logical_and(np.sum(normals * vertices, axis=1) < 0, pos_mask3)

    in_vert_mask = in_vert_mask1 + in_vert_mask2 + in_vert_mask3

    in_triangles_mask = np.all(in_vert_mask[triangles], axis=1)
    in_triangles = triangles[in_triangles_mask]
    in_mesh = copy.deepcopy(mesh)
    in_mesh.triangles = o3d.utility.Vector3iVector(in_triangles)

    out_triangles_mask = ~in_triangles_mask
    out_triangles = triangles[out_triangles_mask]
    out_mesh = copy.deepcopy(mesh)
    out_mesh.triangles = o3d.utility.Vector3iVector(out_triangles)

    assert len(in_triangles) + len(out_triangles) == len(
        triangles
    ), "Triangles are not correctly separated"

    in_mesh, in_triangles, in_to_out_triangles = keep_large_component(
        in_mesh, min_size=100
    )
    out_mesh, out_triangles, out_to_in_triangles = keep_large_component(
        out_mesh, min_size=100
    )

    in_triangles = np.concatenate([in_triangles, out_to_in_triangles])
    out_triangles = np.concatenate([out_triangles, in_to_out_triangles])

    return in_triangles, out_triangles


def UVmap_planar(vertices):
    """UV mapping from xyz coordinates of upward-facing vertices
    Args:
        vertices: xyz coordinates of vertices

    Return:
        uv_coords: uv coordinates of vertices (u=x, v=z)
        depth_map: projection depth = y_max - y"""

    uv_coords = vertices[:, [0, 2]]
    depth_map = vertices[:, 1]

    return uv_coords, depth_map


def UVmap_cylindrical(vertices, if_inward=True):
    """UV mapping from xyz coordinates of inward or outward facing vertices
    Args:
        vertices: xyz coordinates of vertices
        if_inward: if True, the vertices are inward facing, projection depth = xz_norm
                   if False, the vertices are outward facing = max(xz_norm) - xz_norm

    Return:
        uv_coords: uv coordinates of vertices (u=theta, v=y)
        depth_map: projection depth"""

    uv_coords = []
    for i in range(vertices.shape[0]):
        x, y, z = vertices[i]
        theta = np.arctan2(z, x)

        theta = theta + np.pi / 2
        if theta < 0:
            theta = theta + 2 * np.pi

        y = y
        uv_coords.append([theta, y])

    depth_map = np.linalg.norm(vertices[:, [0, 2]], axis=1)
    if if_inward:
        depth_map = np.max(depth_map) - depth_map

    return np.array(uv_coords), depth_map


def normalize_uv_coords(uv_coords, u_range, v_range, with_nl=False):
    """Normalize each observed UV axis to [0, 1].

    The native signature retains u_range, v_range, and with_nl, but this
    implementation ignores them. Fixed coordinate ranges are not applied.
    """
    uv_norm_coords = np.copy(uv_coords)

    u_min, u_max = np.min(uv_coords[:, 0]), np.max(uv_coords[:, 0])
    v_min, v_max = np.min(uv_coords[:, 1]), np.max(uv_coords[:, 1])

    uv_norm_coords[:, 0] = (uv_coords[:, 0] - u_min) / (u_max - u_min)
    uv_norm_coords[:, 1] = (uv_coords[:, 1] - v_min) / (v_max - v_min)

    return uv_norm_coords


def get_tri_RGB(triangles, vertex_RGB):
    """Get the RGB of each triangle face from the RGB of its 3 vertices"""

    if len(triangles) == 0:
        print("Warning: Empty triangles array in get_tri_RGB")
        return np.array([])

    if len(vertex_RGB) == 0:
        print("Warning: Empty vertex_RGB array in get_tri_RGB")
        return np.array([])

    max_vertex_idx = np.max(triangles)
    if max_vertex_idx >= len(vertex_RGB):
        print(
            f"Error: Triangle vertex index {max_vertex_idx} exceeds vertex array length {len(vertex_RGB)}"
        )
        return np.array([])

    tri_RGBs = []
    for i, triangle in enumerate(triangles):
        try:
            colors_3vert = vertex_RGB[triangle]

            tri_rgb = np.max(colors_3vert, axis=0)

            tri_RGBs.append(tri_rgb)
        except Exception as e:
            print(f"Error processing triangle {i}: {e}")
            print(f"Triangle indices: {triangle}")
            print(f"Vertex RGB array length: {len(vertex_RGB)}")

            tri_RGBs.append(np.array([0, 0, 0]))

    result = np.array(tri_RGBs)
    if result.ndim == 1:
        print(f"Warning: get_tri_RGB returned 1D array, converting to 2D")
        result = (
            result.reshape(-1, 3) if len(result) > 0 else np.array([]).reshape(0, 3)
        )

    return result


def rasterize(sorted_tri, sorted_tri_RGB, uv_pixel, img):
    """Paint in depth order; later triangles overwrite earlier pixels."""
    for i, tri in enumerate(sorted_tri):
        pts = uv_pixel[tri].reshape((-1, 1, 2)).astype(np.int32)
        tri_RGB = tuple(int(c) for c in sorted_tri_RGB[i])
        cv2.fillPoly(img, [pts], tri_RGB)
    return img


def rasterize_uv_to_image(
    uv_norm_coords,
    vert_depth,
    xx_triangles,
    vert_originRGB,
    vert_labelRGB,
    px=256,
    is_io=True,
):
    if is_io:
        px_h = px
        px_w = px_h * 8
    else:
        px_h = px * 2
        px_w = px * 3

    uv_pixel = np.copy(uv_norm_coords)
    uv_pixel[:, 0] = (uv_norm_coords[:, 0] * px_w - 1).astype(np.int32)
    uv_pixel[:, 1] = (uv_norm_coords[:, 1] * px_h - 1).astype(np.int32)

    tri_depth = np.mean(vert_depth[xx_triangles], axis=1)

    sorted_tri_idx = np.argsort(tri_depth)
    sorted_tri = xx_triangles[sorted_tri_idx]

    sorted_tri_originRGB = get_tri_RGB(sorted_tri, vert_originRGB)
    sorted_tri_labelRGB = get_tri_RGB(sorted_tri, vert_labelRGB)

    if len(sorted_tri) == 0:
        print(f"Warning: No valid triangles found for rasterization")
        print(f"sorted_tri shape: {sorted_tri.shape}")
        print(f"vert_labelRGB shape: {vert_labelRGB.shape}")

        img_origin = np.zeros((px_h, px_w, 3), dtype=np.uint8)
        img_label = np.zeros((px_h, px_w, 3), dtype=np.uint8)
        return img_origin, img_label, np.array([]), np.array([])

    if sorted_tri_labelRGB.ndim != 2 or sorted_tri_labelRGB.shape[1] != 3:
        print(
            f"Warning: sorted_tri_labelRGB has unexpected shape: {sorted_tri_labelRGB.shape}"
        )
        print(f"Expected shape: [N, 3], got: {sorted_tri_labelRGB.shape}")
        print(f"sorted_tri shape: {sorted_tri.shape}")
        print(f"vert_labelRGB shape: {vert_labelRGB.shape}")

        img_origin = np.zeros((px_h, px_w, 3), dtype=np.uint8)
        img_label = np.zeros((px_h, px_w, 3), dtype=np.uint8)
        return img_origin, img_label, np.array([]), np.array([])

    nonzero_RGB_con = np.max(sorted_tri_labelRGB, axis=1) == 255
    sorted_tri_labelbiRGB = np.where(
        nonzero_RGB_con[:, np.newaxis], np.array([255, 255, 255]), np.array([0, 0, 0])
    )

    img_origin = np.zeros((px_h, px_w, 3), dtype=np.uint8)
    img_label = np.zeros((px_h, px_w, 3), dtype=np.uint8)

    img_origin = rasterize(sorted_tri, sorted_tri_originRGB, uv_pixel, img_origin)

    img_label = rasterize(sorted_tri, sorted_tri_labelbiRGB, uv_pixel, img_label)

    assert (
        img_origin.shape == img_label.shape
    ), "Origin and label images have different shapes"

    return img_origin, img_label, uv_pixel, sorted_tri


def get_tri_center_uv(triangles, uv_pixels):
    """Return the mean projected vertex coordinates for each triangle."""
    tri_center_uv = np.mean(uv_pixels[triangles], axis=1)
    return tri_center_uv


def get_tri_pred_label(tri_uvpx, pred_img_label):
    """Sample face centers by truncation, clipping to the image boundary."""
    tri_pred_label = []
    px_h, px_w = pred_img_label.shape[:2]
    for uv in tri_uvpx:
        u, v = uv.astype(np.int32)
        u = np.clip(u, 0, px_w - 1)
        v = np.clip(v, 0, px_h - 1)
        tri_pred_label.append(pred_img_label[v, u])
    return np.array(tri_pred_label)


def compute_metrics_tri(gt_labels, pred_labels, is_plaque=True):
    """Legacy label-rasterization diagnostic, not a segmentation metric.

    The <=10-positive shortcut is retained only for source compatibility.
    Model evaluation uses dentalpsam.evaluation, without this shortcut.
    """

    gt_labels_bi = np.any(gt_labels == 1, axis=1).astype(np.int32)
    pred_labels_bi = np.any(pred_labels > 0, axis=1).astype(np.int32)

    if not is_plaque:
        gt_labels_bi = 1 - gt_labels_bi
        pred_labels_bi = 1 - pred_labels_bi

    if np.sum(pred_labels_bi) <= 10:
        print("Too little predicted label")
        return 1, 1

    intersection = np.sum(np.logical_and(gt_labels_bi, pred_labels_bi))
    union = np.sum(np.logical_or(gt_labels_bi, pred_labels_bi))
    iou = intersection / union

    intersection_bi = np.sum(np.logical_and(gt_labels_bi, pred_labels_bi))
    dice = 2 * intersection_bi / (np.sum(gt_labels_bi) + np.sum(pred_labels_bi))

    return iou, dice


# Public snake_case aliases for the native projection routines.
render_single_mesh = process_single_mesh
planar_uv_map = UVmap_planar
cylindrical_uv_map = UVmap_cylindrical
get_triangle_colours = get_tri_RGB
get_triangle_uv_centres = get_tri_center_uv
