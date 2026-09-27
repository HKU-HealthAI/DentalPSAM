import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


MESH_PATCH_SHAPE = (6000, 10)
VIEW_KEYS = {
    "0": "up",
    "1": "in",
    "2": "out",
}
REQUIRED_SPLIT_DIRS = ("origin", "label", "SOTA_mesh", "label_mesh")


def require_split_dirs(base_dir):
    """Ensure the split directory contains the expected manual_2D layout."""
    for name in REQUIRED_SPLIT_DIRS:
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Missing required {name} directory: {path}")


def read_image_or_raise(path, flags, description):
    """Read an image and raise a clear error if OpenCV cannot load it."""
    image = cv2.imread(path, flags)
    if image is None:
        raise FileNotFoundError(f"Could not load {description} image: {path}")
    return image


def load_npz_or_raise(path, description):
    """Load an NPZ file and raise a clear error when it is missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required {description} NPZ: {path}")
    return np.load(path, allow_pickle=True)


class SAMDataset(Dataset):
  """Return RGB image patches, binary masks, and aligned padded mesh arrays."""
  def __init__(self, train_images, train_masks,train_mesh,label_mesh):
    self.images = train_images
    self.masks = train_masks

    self.mesh = train_mesh
    self.label_mesh = label_mesh

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):

    image_torch = torch.tensor(self.images[idx]).float()
    label_torch = torch.tensor(self.masks[idx]).float()

    image_torch = image_torch.permute(2, 0, 1)  # Input layout is [height, width, channels].
    # Add a channel dimension so the mask layout is [1, height, width].
    mask = label_torch.unsqueeze(0)  # [1, 256, 256]


    inputs = {}

    inputs['pixel_values'] = image_torch
    inputs["ground_truth_mask"] = mask

    inputs['SOTA_mesh'] = np.array(self.mesh[idx])
    inputs['label_mesh'] = np.array(self.label_mesh[idx])

    return inputs






def patchify(image,name, patch_size=256):
    """Manually divides the image into patches."""
    patches_per_dim = (image.shape[0] // patch_size, image.shape[1] // patch_size)
    patches = []
    name_patches = []
    for i in range(patches_per_dim[0]):
        for j in range(patches_per_dim[1]):
            patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)
            name_patches.append(name[:-4]+"0"+str(i)+"0"+str(j))
    return np.array(patches),np.array(name_patches)



def load_and_patchify_png(base_dir, min_positive_pixels=50):
    """Load origin and label images, then divide them into 256x256 patches.
    Args:
        base_dir: the directory containing the origin and label folders.
    Returns:
        origin_patches: a numpy array of shape (num_patches, 256, 256, 3) containing the origin images.
        label_patches: a numpy array of shape (num_patches, 256, 256, 1) containing the label images.
    """

    label_dir = os.path.join(base_dir, 'label')
    origin_dir = os.path.join(base_dir, 'origin')

    SOTA_mesh_dir = os.path.join(base_dir,'SOTA_mesh')
    label_mesh_dir = os.path.join(base_dir,'label_mesh')

    require_split_dirs(base_dir)

    label_patches = []
    origin_patches = []


    SOTA_mesh_patches = []
    label_mesh_patches = []

    file_names = sorted(os.listdir(label_dir))
    for file_name in file_names:

        label_path = os.path.join(label_dir, file_name)
        origin_path = os.path.join(origin_dir, file_name)

        SOTA_mesh_path = os.path.join(SOTA_mesh_dir,file_name[:-6]+".npz")
        label_mesh_path = os.path.join(label_mesh_dir,file_name[:-6]+".npz")


        # Load images
        label_img = read_image_or_raise(label_path, cv2.IMREAD_GRAYSCALE, "label")
        label_img = (label_img > 127).astype(np.uint8)  # label_img is binary

        origin_img = read_image_or_raise(origin_path, cv2.IMREAD_COLOR, "origin")
        # Convert BGR to RGB
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

        # Load mesh
        SOTA_npz = load_npz_or_raise(SOTA_mesh_path, "SOTA_mesh")
        label_npz = load_npz_or_raise(label_mesh_path, "label_mesh")

        try:
            # Patchify images
            label_patches_ary,_ = patchify(label_img,file_name)
            origin_patches_ary,_ = patchify(origin_img,file_name)

            SOTA_mesh_patches_ary = load_npz(SOTA_npz,file_name[-5])
            label_mesh_patches_ary = load_npz(label_npz,file_name[-5])



        except ValueError as e:
            raise ValueError(f"Invalid patches for {file_name}: {e}") from e

        counts = [len(x) for x in (label_patches_ary, origin_patches_ary,
                                   SOTA_mesh_patches_ary, label_mesh_patches_ary)]
        if len(set(counts)) != 1:
            raise ValueError(f"Image/mesh patch counts differ for {file_name}: {counts}")

        # Filter patches based on the number of positive pixels
        for label_patch, origin_patch,mesh_patch,label_mesh_patch in zip(label_patches_ary, origin_patches_ary,SOTA_mesh_patches_ary,label_mesh_patches_ary):
            if np.count_nonzero(label_patch) >= min_positive_pixels:
                label_patches.append(label_patch)
                origin_patches.append(origin_patch)
                SOTA_mesh_patches.append(mesh_patch)
                label_mesh_patches.append(label_mesh_patch)


    # Convert list to numpy array
    label_patches = np.array(label_patches)
    origin_patches = np.array(origin_patches)


    SOTA_mesh_patches = np.array(SOTA_mesh_patches)
    label_mesh_patches = np.array(label_mesh_patches)

    print(f"Origin patches shape: {origin_patches.shape}")
    print(f"Label patches shape: {label_patches.shape}")


    return origin_patches, label_patches,SOTA_mesh_patches,label_mesh_patches

def load_npz(SOTA_npz,pos):
    """Load one view from a mesh NPZ and pad patches to the public shape."""
    if pos not in VIEW_KEYS:
        raise ValueError(f"Unsupported view position {pos!r}; expected one of {sorted(VIEW_KEYS)}.")
    return pad(SOTA_npz[VIEW_KEYS[pos]], MESH_PATCH_SHAPE)

def pad(data,output_shape):
    SOTA_mesh_patches_ary = []


    # Iterate over each matrix in the list and pad it with zeros
    for i in range(len(data)):
        matrix = data[i]
        if matrix.shape[0] == 0:
            SOTA_mesh_patches_ary.append(np.zeros(output_shape))
            continue
        if matrix.ndim != 2 or matrix.shape[0] > output_shape[0] or matrix.shape[1] != output_shape[1]:
            raise ValueError(
                f"Mesh patch shape {matrix.shape} exceeds the padded output shape {output_shape}."
            )
        pad_rows = output_shape[0] - matrix.shape[0]
        pad_cols = output_shape[1] - matrix.shape[1]

        padded_matrix = np.pad(matrix, [(pad_rows, 0), (pad_cols, 0)], mode='constant', constant_values=0)
        SOTA_mesh_patches_ary.append(padded_matrix)
    return np.array(SOTA_mesh_patches_ary)

def load_and_patchify_png_permesh(base_dir, mesh_name, min_positive_pixels=-1):
    require_split_dirs(base_dir)

    origin_patches = []
    label_patches = []

    SOTA_mesh_patches = []
    label_mesh_patches = []

    preserved_patches_idx = []
    idx = 0
    for i in range(3):
        origin_path = os.path.join(base_dir, "origin", f"{mesh_name}_{i}.png")
        label_path = os.path.join(base_dir, "label", f"{mesh_name}_{i}.png")

        SOTA_mesh_path = os.path.join(base_dir,'SOTA_mesh', f"{mesh_name}.npz")
        label_mesh_path = os.path.join(base_dir,'label_mesh', f"{mesh_name}.npz")

        # Load images
        label_img = read_image_or_raise(label_path, cv2.IMREAD_GRAYSCALE, "label")
        # Use the same binary-label rule in training and prediction.  A low,
        # non-zero interpolation value must not silently become plaque here.
        label_img = (label_img > 127).astype(np.uint8)

        origin_img = read_image_or_raise(origin_path, cv2.IMREAD_COLOR, "origin")
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB


        SOTA_npz = load_npz_or_raise(SOTA_mesh_path, "SOTA_mesh")
        SOTA_mesh_patches_ary = load_npz(SOTA_npz,str(i))

        label_npz = load_npz_or_raise(label_mesh_path, "label_mesh")
        label_mesh_patches_ary = load_npz(label_npz,str(i))


        try:
            # Patchify images
            label_patches_ary,_ = patchify(label_img,f"{mesh_name}_{i}.png")
            origin_patches_ary,_ = patchify(origin_img,f"{mesh_name}_{i}.png")

        except ValueError as e:
            raise ValueError(f"Invalid patches for {mesh_name}: {e}") from e

        counts = [len(x) for x in (label_patches_ary, origin_patches_ary,
                                   SOTA_mesh_patches_ary, label_mesh_patches_ary)]
        if len(set(counts)) != 1:
            raise ValueError(f"Image/mesh patch counts differ for {mesh_name}: {counts}")

        # Filter patches based on the number of positive pixels
        for label_patch, origin_patch,mesh_patch,label_mesh_patch in zip(label_patches_ary, origin_patches_ary,SOTA_mesh_patches_ary,label_mesh_patches_ary):

            if np.count_nonzero(label_patch) >= min_positive_pixels:
                label_patches.append(label_patch)
                origin_patches.append(origin_patch)
                preserved_patches_idx.append(idx)
                # Keep the real TSGCNet prediction in the last channel.
                SOTA_mesh_patches.append(mesh_patch)
                label_mesh_patches.append(label_mesh_patch)

            idx += 1


    # Convert list to numpy array
    label_patches = np.array(label_patches) # shape (num_patches=22, 256, 256), where 22 = 6(up)+8(in)+8(out)
    origin_patches = np.array(origin_patches) # shape (num_patches=22, 256, 256, 3), where 22 = 6(up)+8(in)+8(out)

    SOTA_mesh_patches = np.array(SOTA_mesh_patches)
    label_mesh_patches = np.array(label_mesh_patches)

    return origin_patches, label_patches, preserved_patches_idx,SOTA_mesh_patches,label_mesh_patches
