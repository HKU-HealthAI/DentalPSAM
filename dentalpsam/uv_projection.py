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
    # create a dictionary to store the metrics
    metrics_dict = {}
    for file in os.listdir(label_dir):
        if file.endswith(".ply"):
            mesh_name = file.split(".")[0]
            print(f"Processing {mesh_name}...")
            origin_file_path = f"{origin_dir}/{mesh_name}.ply"
            label_file_path = f"{label_dir}/{mesh_name}.ply"
            raw_origin_mesh = o3d.io.read_triangle_mesh(origin_file_path)
            raw_label_mesh = o3d.io.read_triangle_mesh(label_file_path)
            # print(raw_label_mesh,raw_origin_mesh)
            iou1, dice1, iou0, dice0 = process_single_mesh(raw_origin_mesh, raw_label_mesh, save_dir, mesh_name)

            # 只记录成功的处理结果
            if iou1 is not None:
                metrics_dict[mesh_name] = [iou1, iou0, dice1, dice0]
            else:
                print(f"Warning: Failed to process {mesh_name}, skipping metrics calculation")

    # compute overall metrics (只计算成功处理的文件)
    successful_metrics = [m for m in metrics_dict.values() if m is not None and all(v is not None for v in m)]

    if successful_metrics:
        iou1_mean = np.mean([m[0] for m in successful_metrics])
        iou0_mean = np.mean([m[1] for m in successful_metrics])
        dice1_mean = np.mean([m[2] for m in successful_metrics])
        dice0_mean = np.mean([m[3] for m in successful_metrics])
        print(f"Successfully processed: {len(successful_metrics)}/{len(metrics_dict)} files")
        print(f"Plaque IoU: {iou1_mean:.3f}, Plaque Dice: {dice1_mean:.3f}")
        print(f"Non-Plaque IoU: {iou0_mean:.3f}, Non-Plaque Dice: {dice0_mean:.3f}")
    else:
        print("No files were successfully processed!")
        iou1_mean = iou0_mean = dice1_mean = dice0_mean = 0.0

    # save metrics_dict in a text file in save_dir
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Plaque IoU: {iou1_mean:.3f}, Plaque Dice: {dice1_mean:.3f}\n")
        f.write(f"Non-Plaque IoU: {iou0_mean:.3f}, Non-Plaque Dice: {dice0_mean:.3f}\n")
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")






def process_single_mesh(raw_origin_mesh, raw_label_mesh, save_dir, mesh_name):

    """"""""""""""""1. Recenter """""""""""""""
    # Remove duplicated triangles of the meshes
    raw_origin_mesh.remove_duplicated_triangles()
    raw_label_mesh.remove_duplicated_triangles()

    # Recenter the meshes
    raw_origin_mesh.translate(-raw_origin_mesh.get_center())
    origin_mesh, edp1, edp2 = recentre_mesh(raw_origin_mesh)
    # recentre the label mesh with the same transformation
    raw_label_mesh.translate(-raw_label_mesh.get_center())
    label_mesh, _, _ = recentre_mesh(raw_label_mesh)

    # print(origin_mesh)

    """""""""""""""""""""""""""""""""" 2. Mesh Separation (Upward)"""""""""""""""""""""""""""""""""""
    up_triangles, unsampled_mesh = extract_up_triangles(origin_mesh)


    """""""""""""""""""""""""""""""""" 2. Mesh Separation (Inward and Outward)"""""""""""""""""""""""""""""""""""
    in_triangles, out_triangles = separate_io_triangles(unsampled_mesh)




    """"""""""""""""""""""""""" 3. Project 3D Meshes onto Cylinders as Projection Screens"""""""""""""""""""""""""""
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Execution of UV projection (for all vertices in each of 3 cases)

    # Get the vertices of the origin mesh
    origin_mesh_c = copy.deepcopy(origin_mesh)
    origin_mesh_c.translate(-origin_mesh_c.get_center())
    vertices = np.asarray(origin_mesh_c.vertices) # origin at mesh geometric center

    # UV mapping for all vertices in 3 cases NOTE: all vertices are projected in each case!!!!!!
    uv_up, vert_depth_up = UVmap_planar(vertices)
    uv_in, vert_depth_in = UVmap_cylindrical(vertices, if_inward=True)
    uv_out, vert_depth_out = UVmap_cylindrical(vertices, if_inward=False)


    uv_norm_up = normalize_uv_coords(uv_up, (-45,45), (-35, 25)) # TODO: might require adjustment
    uv_norm_in = normalize_uv_coords(uv_in, (0, 2*np.pi), (-8, 8), with_nl=True)
    uv_norm_out = normalize_uv_coords(uv_out, (0, 2*np.pi), (-9, 7), with_nl=True)


    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Keep track of vertices which belong to each case out of all vertices
    # Find unique vertex indices for each case
    up_vert_idx = np.unique(up_triangles.flatten())
    in_vert_idx = np.unique(in_triangles.flatten())
    out_vert_idx = np.unique(out_triangles.flatten())

    # # Print lengths for verification
    # print(out_vert_idx.shape, in_vert_idx.shape)
    # print(vertices.shape)
    # assert out_vert_idx.shape[0] + in_vert_idx.shape[0] == vertices.shape[0], "Vertices are not correctly separated"

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # COLORS!!!!
    vertices_originRGB_raw = np.asarray(origin_mesh.vertex_colors) # r,g,b within (0-1)
    vertices_labelRGB_raw = np.asarray(label_mesh.vertex_colors) # black/grey/non-white = (0,0,0) for plaque, white!=(0,0,0) for non-plaque

    # Convert RGB format into 0-255
    # Flip plaque representation from dark (black or grey) to white
    vertices_originRGB = (vertices_originRGB_raw * 255).astype(np.uint8) # 0-255
    vertices_labelRGB = ((1-vertices_labelRGB_raw) * 255).astype(np.uint8) # 0-255, non-white=(1,1,1) for plaue, white!=(1,1,1) for non-plaque



    """"""""""""""""""""""""""" 4. Generate 2D Images via Rasterization """""""""""""""""""""""""""
    img_origin_up, img_label_up, uv_pixel_up, sorted_tri_up = rasterize_uv_to_image(uv_norm_up, vert_depth_up, up_triangles, vertices_originRGB, vertices_labelRGB, px=256, is_io=False)
    img_origin_in, img_label_in, uv_pixel_in, sorted_tri_in = rasterize_uv_to_image(uv_norm_in, vert_depth_in, in_triangles, vertices_originRGB, vertices_labelRGB, px=256, is_io=True)
    img_origin_out, img_label_out, uv_pixel_out, sorted_tri_out = rasterize_uv_to_image(uv_norm_out, vert_depth_out, out_triangles, vertices_originRGB, vertices_labelRGB, px=256, is_io=True)

    # 检查是否所有图像都成功生成
    if (img_origin_up is None or img_label_up is None or
        img_origin_in is None or img_label_in is None or
        img_origin_out is None or img_label_out is None):
        print(f"Error: Failed to generate images for {mesh_name}")
        return None, None, None, None

    # 检查是否所有图像都是空的（可能是由于没有有效三角形）
    if (img_origin_up is not None and img_label_up is not None and
        img_origin_in is not None and img_label_in is not None and
        img_origin_out is not None and img_label_out is not None):

        # 检查是否所有图像都是全黑的（表示没有有效三角形）
        all_empty = True
        for img in [img_origin_up, img_label_up, img_origin_in, img_label_in, img_origin_out, img_label_out]:
            if np.any(img != 0):  # 如果有任何非零像素
                all_empty = False
                break

        if all_empty:
            print(f"Warning: All generated images are empty (no valid triangles) for {mesh_name}")
            # 继续处理，但不计算IoU等指标

    # TODO: Inward / Outward projected images will be patched into 1x8
    # TODO: Upward projected image will be patched into 2x4


    """"""""""""""""""""""""""" 5. Visualize and Save """""""""""""""""""""""""""

    save_origin_path = os.path.join(save_dir, "origin")
    save_label_path = os.path.join(save_dir, "label")

    # ensure the info, origin and label folder exists, otherwise create a new folder
    if not os.path.exists(os.path.join(save_dir, "info")):
        os.makedirs(os.path.join(save_dir, "info"))
    if not os.path.exists(save_origin_path):
        os.makedirs(save_origin_path)
    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)

    for idx, (img_origin_xx, img_label_xx) in enumerate(zip([img_origin_up, img_origin_in, img_origin_out], [img_label_up, img_label_in, img_label_out])):
        origin_bgr = cv2.cvtColor(img_origin_xx, cv2.COLOR_RGB2BGR)
        label_gray = cv2.cvtColor(img_label_xx, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(save_origin_path, f"{mesh_name}_{idx}.png"), origin_bgr)
        cv2.imwrite(os.path.join(save_label_path, f"{mesh_name}_{idx}.png"), label_gray)

    # Save UV pixel coordinates and separation of triangles information in npz format
    reconst_info = {"uvpx_up": uv_pixel_up, "uvpx_in": uv_pixel_in, "uvpx_out": uv_pixel_out,
                    "tri_up": sorted_tri_up, "tri_in": sorted_tri_in, "tri_out": sorted_tri_out} # sorted_tri: sorted triangles by depth

    np.savez_compressed(os.path.join(save_dir, "info", f"{mesh_name}.npz"), **reconst_info)
    # Visaulize the rasterized images
    # Display the in- and outward large image for visualization
    # fig, ax = plt.subplots(4,1, figsize=(15, 10))

    # ax[0].imshow(img_origin_out)
    # ax[0].set_title("Outward Facing Projection - Origin")
    # ax[0].axis('off')


    # ax[1].imshow(img_label_out)
    # ax[1].set_title("Outward Facing Projection - Label")
    # ax[1].axis('off')

    # ax[2].imshow(img_origin_in)
    # ax[2].set_title("Inward Facing Projection - Origin")
    # ax[2].axis('off')

    # ax[3].imshow(img_label_in)
    # ax[3].set_title("Inward Facing Projection - Label")
    # ax[3].axis('off')

    # plt.show())

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # # Display the large image for upward-facing visualization
    # fig, ax = plt.subplots(1,2, figsize=(10, 5))

    # ax[0].imshow(img_origin_up)
    # ax[0].set_title("Upward Facing Projection - Origin")
    # ax[0].axis('off')

    # ax[1].imshow(img_label_up)
    # ax[1].set_title("Upward Facing Projection - Label")
    # ax[1].axis('off')

    # plt.show()

    """"""""""""""""""""""""""" 6. Test map back accuracy """""""""""""""""""""""""""
    pred_img_label_up = img_label_up/255
    pred_img_label_in = img_label_in/255
    pred_img_label_out = img_label_out/255


    # 检查sorted_tri是否为空数组（无有效三角形的情况）
    if len(sorted_tri_up) == 0 or len(uv_pixel_up) == 0:
        print(f"Warning: No valid triangles for Up region, skipping metrics calculation")
        return None, None, None, None

    if len(sorted_tri_in) == 0 or len(uv_pixel_in) == 0:
        print(f"Warning: No valid triangles for In region, skipping metrics calculation")
        return None, None, None, None

    if len(sorted_tri_out) == 0 or len(uv_pixel_out) == 0:
        print(f"Warning: No valid triangles for Out region, skipping metrics calculation")
        return None, None, None, None

    tri_uvpx_up = get_tri_center_uv(sorted_tri_up, uv_pixel_up) # TODO: changed from up_triangles to sorted_tri_up
    tri_uvpx_in = get_tri_center_uv(sorted_tri_in, uv_pixel_in) # TODO:
    tri_uvpx_out = get_tri_center_uv(sorted_tri_out, uv_pixel_out) # TODO:

    tri_pred_label_up = get_tri_pred_label(tri_uvpx_up, pred_img_label_up)
    tri_pred_label_in = get_tri_pred_label(tri_uvpx_in, pred_img_label_in)
    tri_pred_label_out = get_tri_pred_label(tri_uvpx_out, pred_img_label_out)

    # Get the GT label for each triangle face
    tri_GT_labelGRB_up = get_tri_RGB(sorted_tri_up, vertices_labelRGB)/255 # TODO: changed from up_triangles to sorted_tri_up
    tri_GT_labelGRB_in = get_tri_RGB(sorted_tri_in, vertices_labelRGB)/255 # TODO:
    tri_GT_labelGRB_out = get_tri_RGB(sorted_tri_out, vertices_labelRGB)/255 # TODO:

    # Compare the predicted labels with the ground truth labels for all triangle faces
    metrics_up = compute_metrics_tri(tri_GT_labelGRB_up, tri_pred_label_up)
    metrics_in = compute_metrics_tri(tri_GT_labelGRB_in, tri_pred_label_in)
    metrics_out = compute_metrics_tri(tri_GT_labelGRB_out, tri_pred_label_out)

    # print(f"Upward: IoU: {metrics_up[0]:.3f}, Dice: {metrics_up[1]:.3f}")
    # print(f"Inward: IoU: {metrics_in[0]:.3f}, Dice: {metrics_in[1]:.3f}")
    # print(f"Outward: IoU: {metrics_out[0]:.3f}, Dice: {metrics_out[1]:.3f}")

    iou_mean = (metrics_up[0] + metrics_in[0] + metrics_out[0]) / 3
    dice_mean = (metrics_up[1] + metrics_in[1] + metrics_out[1]) / 3
    print(f"Plaque IoU: {iou_mean:.3f}, Plaque Dice: {dice_mean:.3f}")


    metrics_up0 = compute_metrics_tri(tri_GT_labelGRB_up, tri_pred_label_up, is_plaque=False)
    metrics_in0 = compute_metrics_tri(tri_GT_labelGRB_in, tri_pred_label_in, is_plaque=False)
    metrics_out0 = compute_metrics_tri(tri_GT_labelGRB_out, tri_pred_label_out, is_plaque=False)

    iou0_mean = (metrics_up0[0] + metrics_in0[0] + metrics_out0[0]) / 3
    dice0_mean = (metrics_up0[1] + metrics_in0[1] + metrics_out0[1]) / 3
    print(f"Non-Plaque IoU: {iou0_mean:.3f}, Non-Plaque Dice: {dice0_mean:.3f}")


    return iou_mean, dice_mean, iou0_mean, dice0_mean



































""" 1. Recenter"""

# Find two end points of the intraoral mesh by the max and min theta values
def find_key_points(mesh):
    vertices = np.asarray(mesh.vertices)
    # print('vertices', vertices.shape) #(8754, 3)
    x,y,z = vertices[:,0], vertices[:,1], vertices[:,2]
    theta = np.arctan2(z,x) + np.pi/2
    theta = np.where(theta<0, theta+2*np.pi, theta) # if theta < 0, add 2pi to make it positive
    endpt1 = vertices[np.argmin(theta)]
    endpt2 = vertices[np.argmax(theta)]
    # shift both endpts to x-z plane (i.e.y=0)
    endpt1[1] = 0
    endpt2[1] = 0
    centre = (endpt1 + endpt2) / 2
    return endpt1, endpt2, centre

# Recentre the mesh with centre keypoint at the origin, rotate the mesh about y axis such that x-axis point to endpt1
def recentre_mesh(mesh):
    mesh_rc = copy.deepcopy(mesh)

    endpt1, endpt2, centre_keypt = find_key_points(mesh)
    vertices = np.asarray(mesh.vertices)
    vertices = vertices - centre_keypt
    endpt1 = endpt1 - centre_keypt
    endpt2 = endpt2 - centre_keypt


    # Rotate the mesh about y axis such that x-axis point to endpt1
    theta = np.arctan2(endpt1[2], endpt1[0])
    # print(f"endpt1: {endpt1}, endpt2: {endpt2}, theta: {theta}")
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    vertices = np.dot(vertices, R.T)
    endpt1 = np.dot(endpt1, R.T)
    endpt2 = np.dot(endpt2, R.T)
    # print(f"endpt1: {endpt1}, endpt2: {endpt2}")
    assert endpt1[2] < 1e-6, f"end point 1 {endpt1} is not on x-axis"
    assert endpt2[2] < 1e-6, f"end point 2 {endpt2} is not on x-axis"

    mesh_rc.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh_rc, endpt1, endpt2



""" 2. Mesh Separation (Upward, Inward and Outward)"""


# Utils function
def set_difference(A, B):
    """ Return the elements in A but not in B"""
    A_view = A.view([('', A.dtype)] * A.shape[1])
    B_view = B.view([('', B.dtype)] * B.shape[1])
    C_view = np.setdiff1d(A_view, B_view)
    C = C_view.view(A.dtype).reshape(-1, A.shape[1])
    # print(f"initial size: {len(A)}, removed size: {len(B)}, final size: {len(C)}")
    return C



# Preserve connected vertices which belong to a big enough / biggest cluster (remove small disconnecting clusters)
def keep_large_component(mesh, max_only=False, min_size=600):
    """ Remove small disconnected components from the mesh
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
    if max_only: # only keep the biggest component
        max_comp_index = np.argmax(comp_sizes)
        large_comp_mask = components == max_comp_index

    else: # keep components with size >= min_size
        large_comp_indices = np.where(comp_sizes >= min_size)[0]
        large_comp_mask = np.isin(components, large_comp_indices)

    preserved_triangles = triangles[large_comp_mask]
    removed_triangles = triangles[np.logical_not(large_comp_mask)]
    mesh.triangles = o3d.utility.Vector3iVector(preserved_triangles) # update the mesh

    return mesh, preserved_triangles, removed_triangles


""" Upward Surface Separation"""
def extract_up_triangles(mesh, xpos_min=10, xpos_max=-10,
                         ypos_min=-5,
                         ynorm_min=0.5
                         ):
    """ All vertices are preserved.
        Triangles with all 3 vertices upward facing are preserved"""

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    # print(normals.shape)

    assert xpos_min > xpos_max, "xpos_min must be greater than xpos_max: to avoid middle region"
    # up_vert_xpos_mask = np.logical_or((vertices[:,0]) > xpos_min, (vertices[:,0]) < xpos_max)
    up_vert_ypos_mask = (vertices[:,1]) > ypos_min
    up_vert_norm_mask = normals[:,1] > ynorm_min
    up_vert_mask =  up_vert_ypos_mask & up_vert_norm_mask
    # print(np.sum(up_vert_mask))

    up_mesh = copy.deepcopy(mesh)
    up_triangles_mask = np.all(up_vert_mask[triangles], axis=1) # all 3 vertices of the triangle are upward facing
    up_triangles = triangles[up_triangles_mask]
    up_mesh.triangles = o3d.utility.Vector3iVector(up_triangles)

    up_mesh, up_triangles, _ = keep_large_component(up_mesh, min_size=600)
    unsampled_triangles = set_difference(triangles,up_triangles)
    unsampled_mesh = copy.deepcopy(mesh)
    unsampled_mesh.triangles = o3d.utility.Vector3iVector(unsampled_triangles)

    unsampled_mesh, unsampled_triangles, _ = keep_large_component(unsampled_mesh, min_size=500)

    # Triangles are either in up_triangles or unsampled_triangles
    up_triangles = set_difference(triangles, unsampled_triangles)
    assert abs(len(up_triangles) + len(unsampled_triangles) - len(triangles))<=1, "Triangles are not correctly separated"
    # print(f"up_triangles: {len(up_triangles)}, unsampled_triangles: {len(unsampled_triangles)}")
    # print(f"total triangles: {len(triangles)}")

    # up_mesh.triangles = o3d.utility.Vector3iVector(up_triangles)

    return up_triangles, unsampled_mesh



""" Inner and Outer Surface Separation"""
# Separate the original mesh vertices into inward-facing and outward-facing vertices relative to the origin
# With original (non re-centered mesh)
def separate_io_triangles(mesh):
    """ Separate the original mesh triangles into
    inward-facing and outward-facing triangles relative to the origin

    All vertices are preserved."""

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    viewpt1 = np.mean(vertices, axis=0)
    vertices_vp1 = vertices - viewpt1
    viewpt2 = viewpt1 / 2
    vertices_vp2 = vertices - viewpt2

    normals = np.asarray(mesh.vertex_normals)

    # analyse the mesh in 3 parts (based on z coords)
    # each part has different viewpt for normal thresholding
    pos_mask1 = vertices[:, 2] > viewpt1[2]
    in_vert_mask1 = np.logical_and(np.sum(normals * vertices_vp1, axis=1) < 0, pos_mask1)
    pos_mask2 = np.logical_and(viewpt2[2] < vertices[:, 2], vertices[:, 2] < viewpt1[2])
    in_vert_mask2 = np.logical_and(np.sum(normals * vertices_vp2, axis=1) < 0, pos_mask2)
    pos_mask3 = vertices[:, 2] < viewpt2[2]
    in_vert_mask3 = np.logical_and(np.sum(normals * vertices, axis=1) < 0, pos_mask3)

    # merge the 3 masks
    in_vert_mask = in_vert_mask1 + in_vert_mask2 + in_vert_mask3

    # separate triangles
    in_triangles_mask = np.all(in_vert_mask[triangles], axis=1)
    in_triangles = triangles[in_triangles_mask]
    in_mesh = copy.deepcopy(mesh)
    in_mesh.triangles = o3d.utility.Vector3iVector(in_triangles)

    out_triangles_mask = ~in_triangles_mask
    out_triangles = triangles[out_triangles_mask]
    out_mesh = copy.deepcopy(mesh)
    out_mesh.triangles = o3d.utility.Vector3iVector(out_triangles)

    assert len(in_triangles) + len(out_triangles) == len(triangles), "Triangles are not correctly separated"

    # clean up
    in_mesh, in_triangles, in_to_out_triangles = keep_large_component(in_mesh, min_size=100)
    out_mesh, out_triangles, out_to_in_triangles = keep_large_component(out_mesh, min_size=100)

    # add out_to_in_triangles to in_triangles
    in_triangles = np.concatenate([in_triangles, out_to_in_triangles])
    out_triangles = np.concatenate([out_triangles, in_to_out_triangles])

    return in_triangles, out_triangles





""" 3. Project 3D Meshes onto Plane and Cylinders as Projection Screens"""


def UVmap_planar(vertices):
    """ UV mapping from xyz coordinates of upward-facing vertices
    Args:
        vertices: xyz coordinates of vertices

    Return:
        uv_coords: uv coordinates of vertices (u=x, v=z)
        depth_map: projection depth = y_max - y"""

    uv_coords = vertices[:, [0, 2]]
    depth_map = vertices[:, 1] # larger y value means closer to the camera (rasterized later)

    return uv_coords, depth_map



# Project vertices onto the respective cylinders
def UVmap_cylindrical(vertices, if_inward=True):
    """ UV mapping from xyz coordinates of inward or outward facing vertices
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

        # FIXME: only if mesh is geometrically centered at origin!!!!!!
        theta = theta + np.pi /2
        if theta < 0:
            theta = theta + 2 * np.pi

        y = y
        uv_coords.append([theta, y])

    depth_map = np.linalg.norm(vertices[:, [0, 2]], axis=1)
    if if_inward:
        depth_map = np.max(depth_map) - depth_map

    return np.array(uv_coords), depth_map



# def normalize_uv_coords(uv_coords, u_range, v_range, with_nl=False):
#     uv_norm_coords = np.copy(uv_coords)
#     u_min, u_max = u_range
#     v_min, v_max = v_range
#     uv_norm_coords[:, 0] = (uv_coords[:, 0] - u_min) / (u_max - u_min)
#     uv_norm_coords[:, 1] = (uv_coords[:, 1] - v_min) / (v_max - v_min)

#     return uv_norm_coords
def normalize_uv_coords(uv_coords,u_range, v_range, with_nl=False):
    uv_norm_coords = np.copy(uv_coords)

    # 计算 U 和 V 的范围
    u_min, u_max = np.min(uv_coords[:, 0]), np.max(uv_coords[:, 0])
    v_min, v_max = np.min(uv_coords[:, 1]), np.max(uv_coords[:, 1])

    # 打印范围
    # print(f'U range: ({u_min}, {u_max})')
    # print(f'V range: ({v_min}, {v_max})')

    # 归一化 U 和 V
    uv_norm_coords[:, 0] = (uv_coords[:, 0] - u_min) / (u_max - u_min)
    uv_norm_coords[:, 1] = (uv_coords[:, 1] - v_min) / (v_max - v_min)

    return uv_norm_coords


def get_tri_RGB(triangles, vertex_RGB):
    """ Get the RGB of each triangle face from the RGB of its 3 vertices"""
    # 检查输入参数
    if len(triangles) == 0:
        print("Warning: Empty triangles array in get_tri_RGB")
        return np.array([])

    if len(vertex_RGB) == 0:
        print("Warning: Empty vertex_RGB array in get_tri_RGB")
        return np.array([])

    max_vertex_idx = np.max(triangles)
    if max_vertex_idx >= len(vertex_RGB):
        print(f"Error: Triangle vertex index {max_vertex_idx} exceeds vertex array length {len(vertex_RGB)}")
        return np.array([])

    tri_RGBs = []
    for i, triangle in enumerate(triangles):
        try:
            colors_3vert = vertex_RGB[triangle]
            # Get the maximum color value for each channel
            tri_rgb = np.max(colors_3vert, axis=0) # FIXME: max (white [255,255,255] being the plaque) vs mean!!!!
            # tri_rgb = np.min(colors_3vert, axis=0) ###yzq:这里自己改的,还是原来更高
            tri_RGBs.append(tri_rgb)
        except Exception as e:
            print(f"Error processing triangle {i}: {e}")
            print(f"Triangle indices: {triangle}")
            print(f"Vertex RGB array length: {len(vertex_RGB)}")
            # 使用默认颜色
            tri_RGBs.append(np.array([0, 0, 0]))

    result = np.array(tri_RGBs)
    if result.ndim == 1:
        print(f"Warning: get_tri_RGB returned 1D array, converting to 2D")
        result = result.reshape(-1, 3) if len(result) > 0 else np.array([]).reshape(0, 3)

    return result


def rasterize(sorted_tri, sorted_tri_RGB, uv_pixel, img):
    for i, tri in enumerate(sorted_tri):
        pts = uv_pixel[tri].reshape((-1, 1, 2)).astype(np.int32)
        tri_RGB = tuple(int(c) for c in sorted_tri_RGB[i]) # convert to tuple for cv2.fillPoly
        cv2.fillPoly(img, [pts], tri_RGB)
    return img




def rasterize_uv_to_image(uv_norm_coords, vert_depth, xx_triangles, vert_originRGB, vert_labelRGB, px=256, is_io=True):
    if is_io:
        px_h = px # 256
        px_w = px_h*8 # 2048
    else:
        px_h = px*2 # 512
        px_w = px*3 # 768

    # Scale uv_norm_coords by width and height to get pixel coordinates
    uv_pixel = np.copy(uv_norm_coords)
    uv_pixel[:, 0] = (uv_norm_coords[:, 0] * px_w-1).astype(np.int32)
    uv_pixel[:, 1] = (uv_norm_coords[:, 1] * px_h-1).astype(np.int32)

    # Calculate projection depth for each triangle face (mean projection depth of 3 vertices)
    tri_depth = np.mean(vert_depth[xx_triangles], axis=1)

    # Sort the triangles by depth (ascending)
    sorted_tri_idx = np.argsort(tri_depth)
    sorted_tri= xx_triangles[sorted_tri_idx]


    """ Get original colors and labels for each triangle face"""
    sorted_tri_originRGB = get_tri_RGB(sorted_tri, vert_originRGB)
    sorted_tri_labelRGB = get_tri_RGB(sorted_tri, vert_labelRGB)

    # 检查是否有有效的三角形
    if len(sorted_tri) == 0:
        print(f"Warning: No valid triangles found for rasterization")
        print(f"sorted_tri shape: {sorted_tri.shape}")
        print(f"vert_labelRGB shape: {vert_labelRGB.shape}")
        # 返回空的黑色图像（表示无菌斑，保守处理）
        img_origin = np.zeros((px_h, px_w, 3), dtype=np.uint8)
        img_label = np.zeros((px_h, px_w, 3), dtype=np.uint8)
        return img_origin, img_label, np.array([]), np.array([])

    # 检查颜色数组的维度，防止轴越界错误
    if sorted_tri_labelRGB.ndim != 2 or sorted_tri_labelRGB.shape[1] != 3:
        print(f"Warning: sorted_tri_labelRGB has unexpected shape: {sorted_tri_labelRGB.shape}")
        print(f"Expected shape: [N, 3], got: {sorted_tri_labelRGB.shape}")
        print(f"sorted_tri shape: {sorted_tri.shape}")
        print(f"vert_labelRGB shape: {vert_labelRGB.shape}")
        # 返回空的黑色图像（表示无菌斑，保守处理）
        img_origin = np.zeros((px_h, px_w, 3), dtype=np.uint8)
        img_label = np.zeros((px_h, px_w, 3), dtype=np.uint8)
        return img_origin, img_label, np.array([]), np.array([])

    # FIXME: convert all non-zero rgb values to white (255,255,255)!!!! i.e. binary label (plaque or non-plaque)
    nonzero_RGB_con = np.max(sorted_tri_labelRGB, axis=1) == 255
    sorted_tri_labelbiRGB = np.where(nonzero_RGB_con[:, np.newaxis], np.array([255, 255, 255]), np.array([0, 0, 0]))


    """ Rasterize the triangles in sorted order of depth"""
    # Initialize the image with black background
    img_origin = np.zeros((px_h, px_w, 3), dtype=np.uint8)
    img_label = np.zeros((px_h, px_w, 3), dtype=np.uint8)
    # Rasterize
    img_origin = rasterize(sorted_tri, sorted_tri_originRGB, uv_pixel, img_origin)
    # img_label = rasterize(sorted_tri, sorted_tri_labelRGB, uv_pixel, img_label)
    img_label = rasterize(sorted_tri, sorted_tri_labelbiRGB, uv_pixel, img_label) # FIXME: rasterize with binary label?

    # print(f"Image shape: {img_origin.shape}")
    assert img_origin.shape == img_label.shape, "Origin and label images have different shapes"

    """ Simply return the entire complete image as a whole.
        Use Patchify function to divide the image into 8 patches later before training"""

    # # FIXME: Not sure whether to convert to binary label for training?????
    # if is_binary:
    #     img_label = np.where(img_label > 0, 1, 0).astype(np.uint8)

    return img_origin, img_label, uv_pixel, sorted_tri # img: 3 RGB channel np arrays


""""""""""""""""""""""""""" 6. Test map back accuracy """""""""""""""""""""""""""
def get_tri_center_uv(triangles, uv_pixels):
    tri_center_uv = np.mean(uv_pixels[triangles], axis=1)
    return tri_center_uv

# Extract the color of each triangle face center from the predicted label image
# Need to execute for each direction (outward, inward, upward)
def get_tri_pred_label(tri_uvpx, pred_img_label):
    tri_pred_label = []
    px_h, px_w = pred_img_label.shape[:2]
    for uv in tri_uvpx:
        u, v = uv.astype(np.int32)
        u = np.clip(u, 0, px_w-1)
        v = np.clip(v, 0, px_h-1)
        tri_pred_label.append(pred_img_label[v, u])
    return np.array(tri_pred_label)


def compute_metrics_tri(gt_labels, pred_labels, is_plaque = True):
    """ Compute the IoU and Dice scores for the triangles """
    # Convert RGB to binary scalar
    gt_labels_bi = np.any(gt_labels == 1, axis=1).astype(np.int32)
    pred_labels_bi = np.any(pred_labels > 0, axis=1).astype(np.int32)
    # print(np.sum(gt_labels_bi), np.sum(pred_labels_bi))

    if not is_plaque:
        gt_labels_bi = 1 - gt_labels_bi
        pred_labels_bi = 1 - pred_labels_bi
        # print(np.sum(gt_labels_bi), np.sum(pred_labels_bi))

    if np.sum(pred_labels_bi) <=10: # FIXME: to check
        print("Too little predicted label")
        return 1, 1

    intersection = np.sum(np.logical_and(gt_labels_bi, pred_labels_bi))
    union = np.sum(np.logical_or(gt_labels_bi, pred_labels_bi))
    iou = intersection / union

    intersection_bi = np.sum(np.logical_and(gt_labels_bi, pred_labels_bi))
    dice = 2 * intersection_bi / (np.sum(gt_labels_bi) + np.sum(pred_labels_bi))
    # if non-binary (3 channel)
    # dice = 2/3 * intersection / (np.sum(np.any(gt_labels!=0,axis=1)) + np.sum(np.any(pred_labels!=0,axis=1)))
    return iou, dice









# Standard public aliases.  The original names above are retained because
# changing this numerical module wholesale would make source-level comparison
# with the historical server implementation unnecessarily difficult.
render_single_mesh = process_single_mesh
planar_uv_map = UVmap_planar
cylindrical_uv_map = UVmap_cylindrical
get_triangle_colours = get_tri_RGB
get_triangle_uv_centres = get_tri_center_uv
