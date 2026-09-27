from typing import Any, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from .segment_anything.modeling import TwoWayTransformer
from .segment_anything.modeling.common import LayerNorm2d


class DentalPSAM(nn.Module):
    """Checkpoint-compatible baseline DentalPSAM model."""

    def __init__(self, sam, mesh_fusion="gated"):
        super().__init__()
        self.resize = torchvision.transforms.Resize(
            (1024, 1024),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        self.image_encoder = sam.image_encoder
        self.preprocess = sam.preprocess
        self.embeddings = Embeddings(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        )

        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.mesh_encoder = MeshEncoder(
            mesh_dim=(6000, 10),
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            num_mesh_point=256,
        )
        self.mesh_decoder = MeshDecoder(
            mesh_dim=(256, 256),
            num_mesh_point=6000,
            fusion=mesh_fusion,
        )
        self._freeze_encoder()
        self.embeddings.load_state_dict(sam.prompt_encoder.state_dict())

    def forward(self, image, SOTA_mesh, return_intermediate=False):
        """Run 2D mask prediction and 3D mesh prediction.

        Args:
            image: rendered IOS image tensor.
            SOTA_mesh: padded mesh tensor with shape ``[B, 6000, 10]``.
            return_intermediate: when True, also return intermediate features.
        """
        image = self.resize(image)
        input_images = self.preprocess(image)
        image_embeddings = self.image_encoder(input_images)
        sparse_embeddings, _dense_embeddings = self.embeddings(
            masks=None,
        )

        if return_intermediate:
            mesh_input, dense_mask, mesh_embeddings = self.mesh_encoder(
                SOTA_mesh,
                image_pe=self.embeddings.get_dense_pe(),
                return_intermediate=True,
            )
        else:
            mesh_input, dense_mask = self.mesh_encoder(
                SOTA_mesh,
                image_pe=self.embeddings.get_dense_pe(),
                return_intermediate=False,
            )

        low_res_masks, iou_predictions, src = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.embeddings.get_dense_pe(),
            dense_prompt_embeddings=dense_mask,
            sparse_prompt_embeddings=sparse_embeddings,
        )
        mesh_pred, _ = self.mesh_decoder(
            mesh_input,
            image_pe=self.embeddings.get_dense_pe(),
            mask_embeddings=src,
        )

        outputs = {
            'pred_masks': low_res_masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': None,
            'pred_mesh': mesh_pred,
        }

        if return_intermediate:
            intermediate_features = {
                'mesh_embeddings': mesh_embeddings,  # [B, num_mesh_point, 256]
                'dense_mask': dense_mask,            # [B, 256, 64, 64]
                'image_embeddings': image_embeddings,  # [B, 256, 64, 64]
            }
            return outputs, intermediate_features

        return outputs

    def _freeze_encoder(self):
        """Freeze the SAM image encoder except the last six parameter tensors."""
        last_layer_no = 176
        for layer_no, param in enumerate(self.image_encoder.parameters()):
            if layer_no > (last_layer_no - 6):
                param.requires_grad = True
            else:
                param.requires_grad = False


class MeshEncoder(nn.Module):
    def __init__(
        self,
        *,
        mesh_dim: tuple,
        transformer: nn.Module,
        num_mesh_point: int,
    ) -> None:
        super().__init__()
        self.mask_shape = (256, 64, 64)
        self.num_mesh_point = num_mesh_point
        self.mask_tokens = nn.Embedding(
            self.mask_shape[0],
            self.mask_shape[1] * self.mask_shape[2],
        )
        self.mesh_embedding = LSTMModel(mesh_dim[1], 256, 1, 256 * 256)
        self.transformer = transformer

    def _get_batch_size(self, sota_mesh) -> int:
        """Get the batch size from the input mesh tensor."""
        if sota_mesh is not None:
            return sota_mesh.shape[0]
        return 1

    def forward(self, sota_mesh, image_pe, return_intermediate=False):
        """Encode a mesh patch into SAM-compatible dense prompt features."""
        bs = self._get_batch_size(sota_mesh)
        mesh_embeddings = self.mesh_embedding(sota_mesh).reshape(
            -1,
            self.num_mesh_point,
            self.mask_shape[0],
        )

        mask_embeddings = self.mask_tokens.weight.reshape(
            1,
            -1,
            self.mask_shape[1],
            self.mask_shape[2],
        ).expand(bs, -1, self.mask_shape[1], self.mask_shape[2])
        _mesh_out, dense_mask = self.transformer(
            mask_embeddings,
            image_pe,
            mesh_embeddings,
        )

        dense_mask_reshaped = dense_mask.permute(0, 2, 1).reshape(
            bs,
            -1,
            self.mask_shape[1],
            self.mask_shape[2],
        )

        if return_intermediate:
            return sota_mesh, dense_mask_reshaped, mesh_embeddings

        return sota_mesh, dense_mask_reshaped

    def get_embeddings(self, sota_mesh):
        """Return LSTM mesh embeddings with shape ``[B, num_mesh_point, 256]``."""
        return self.mesh_embedding(sota_mesh).reshape(
            -1,
            self.num_mesh_point,
            self.mask_shape[0],
        )


class MeshDecoder(nn.Module):
    def __init__(
        self,
        *,
        mesh_dim: tuple,  # (256, 256)
        num_mesh_point: int,
        fusion: str = "gated",
    ) -> None:
        super().__init__()
        self.mask_shape = (256, 64, 64)
        self.mesh_dim = mesh_dim
        self.num_mesh_point = num_mesh_point
        self.cls = 1

        self.transformer_dim = 256

        self.upscaling = nn.Sequential(
            nn.Conv2d(self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(self.transformer_dim // 4, 1, kernel_size=2, stride=2),
        )

        self.mlp = MLP(9, 256, 256, 3)
        if fusion not in ("gated", "concat"):
            raise ValueError("Mesh fusion must be 'gated' or 'concat'")
        self.fusion = fusion
        if fusion == "concat":
            # The newer native model adds exactly these six checkpoint keys.
            self.mlp_fusion = MLP(257, 256, 256, 3)

    def _get_batch_size(self, sota_mesh) -> int:
        """Get the batch size from the mesh tensor."""
        if sota_mesh is not None:
            return sota_mesh.shape[0]
        return 1

    def forward(self, mesh_embeddings, image_pe, mask_embeddings):
        """Decode per-face plaque logits from mesh features and mask embeddings."""
        _bs = self._get_batch_size(mesh_embeddings)

        mesh_pos = self.mlp(mesh_embeddings[:, :, :9])
        # Channel 9 is an upstream auxiliary score in SOTA_mesh, not the
        # supervised label_mesh target. Keep the arithmetic unchanged for
        # checkpoint compatibility while making the data-flow role explicit.
        mesh_auxiliary_score = mesh_embeddings[:, :, 9:]
        if self.fusion == "concat":
            mesh_features = self.mlp_fusion(torch.cat([mesh_pos, mesh_auxiliary_score], dim=-1))
        else:
            expanded_auxiliary_score = torch.repeat_interleave(
                mesh_auxiliary_score, mesh_pos.shape[2], dim=2
            )
            mesh_features = torch.mul(expanded_auxiliary_score, mesh_pos)

        mask_embeddings = self.upscaling(mask_embeddings).flatten(2).permute(0, 2, 1)
        mesh_pred = mesh_features @ mask_embeddings
        return mesh_pred, None


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        # num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_mask_tokens = 1

        self.iou_token = nn.Embedding(1, transformer_dim)

        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=True,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,

    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of
            image_embeddings
          sparse_prompt_embeddings (torch.Tensor): sparse prompt embeddings
          dense_prompt_embeddings (torch.Tensor): dense prompt embeddings

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: decoder mask embeddings for the 3D mesh decoder
        """
        masks, iou_pred, ori_mask = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            dense_prompt_embeddings=dense_prompt_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        return masks, iou_pred, ori_mask

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = image_embeddings + dense_prompt_embeddings

        pos_src = image_pe

        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)

        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        iou_token_out = hs[:, :, :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred, src


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Embeddings(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim)
            for _ in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)

        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )  # downsample to 1/4
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        return self.mask_downscaling(masks.unsqueeze(1))

    def _get_batch_size(
        self,
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """

        if masks is not None:
            return masks.shape[0]
        return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(masks)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        return out
