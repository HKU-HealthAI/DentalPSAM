"""Adjacency feature gathering for the checkpoint-compatible TSGCNet model.

Legacy training/evaluation helpers are intentionally not duplicated here;
the public train/validate entry points own those operations.
"""

import torch


def gather_adjacent_features(coor, processed_adjacent):
    """Concatenate each face's features with its neighbors' features.

    Args:
        coor: face features, shape ``[1, channels, faces]``.
        processed_adjacent: neighbor indices, shape ``[faces, neighbors]``.

    Returns:
        Tensor of shape ``[1, 2 * channels, faces, neighbors]``. This preserves
        the historical index-select and concatenation order without allocating
        a quadratic ``faces x faces`` feature tensor. Batch size is exactly one.
    """
    if coor.shape[0] != 1:
        raise ValueError("TSGCNet adjacency gathering requires batch size one")
    processed_adjacent = processed_adjacent.to(coor.device)
    channels, faces = coor.shape[1:]
    neighbors = processed_adjacent.shape[1]
    flattened = coor.squeeze(0).unsqueeze(-1)
    adjacent = torch.index_select(flattened, dim=1, index=processed_adjacent.view(-1))
    adjacent = adjacent.view(1, channels, faces, neighbors)
    centre = coor.unsqueeze(-1).expand(1, channels, faces, neighbors)
    return torch.cat([centre, adjacent], dim=1)
