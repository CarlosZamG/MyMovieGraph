"""Utilities shared by GraphSAGE offline preprocessing and runtime inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from config import DROPOUT, HIDDEN_CHANNELS
from models import Model


FORWARD_RELATION = ("user", "rates", "movie")
REVERSE_RELATION = ("movie", "rev_rates", "user")


def extract_checkpoint_parts(
    checkpoint_path: str | Path,
    user_features_path: str | Path | None = None,
    user_feature_mode: str = "training",
):
    """Return trained model weights and historical INITIAL user features.

    ``user_features_pca.pt`` contains the initial user-node feature matrix
    assigned to ``data["user"].x`` during training. It is NOT a set of
    GraphSAGE hidden representations or final user embeddings.

    A checkpoint entry named ``user_embeddings`` is deliberately ignored:
    it is not a substitute for the initial input feature matrix required by
    this decomposition.

    user_feature_mode:
        ``training``: use the exact initial feature matrix from
            ``user_features_pca.pt`` (the training-time node inputs).
        ``zeros``: reproduce the old cold-start inference fallback, which
            initialized all historical user input features to zero. The PCA
            file is still required to establish the expected feature dimension.

    Rows of ``user_features_pca.pt`` must use the same first-occurrence user
    mapping as ``inference_bundle/edge_index.npy``.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if user_feature_mode not in {"training", "zeros"}:
        raise ValueError("user_feature_mode must be 'training' or 'zeros'.")

    if user_features_path is None or not Path(user_features_path).is_file():
        raise FileNotFoundError(
            "Initial user input features were not found. Pass the exact "
            "user_features_pca.pt file used during training with --user-features. "
            "Do not substitute GraphSAGE-generated embeddings from the checkpoint."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a dictionary-like checkpoint.")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a valid model state_dict.")

    # This exact tensor is the input x_user^(0) used to train the GNN.
    user_features = torch.load(
        Path(user_features_path),
        map_location="cpu",
        weights_only=True,
    )

    if not torch.is_tensor(user_features):
        raise TypeError(
            "user_features_pca.pt must contain a torch.Tensor, matching "
            "data.py's torch.load(...).float() usage."
        )

    user_features = user_features.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()

    if user_features.ndim != 2:
        raise ValueError(
            "Initial user features must be a 2D tensor; "
            f"got {tuple(user_features.shape)}."
        )

    if user_feature_mode == "zeros":
        # Only for a controlled comparison with the previous recommender's
        # zero-filled historical-user input behavior.
        user_features = torch.zeros_like(user_features)
        feature_source = f"zeros (dimension checked using {user_features_path})"
    else:
        feature_source = str(Path(user_features_path))

    # Make a standalone state-dict copy so the loaded checkpoint can be freed.
    model_state = {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }
    del checkpoint

    print(f"Historical INITIAL user feature source: {feature_source}")
    print(f"Historical initial user feature shape: {tuple(user_features.shape)}")
    print(f"Historical user feature mode: {user_feature_mode}")

    return model_state, user_features


def build_initialized_model(
    model_state_dict: Mapping[str, torch.Tensor],
    user_feature_dim: int,
    movie_feature_dim: int,
    device: str | torch.device = "cpu",
) -> Model:
    """Instantiate the saved to_hetero GraphSAGE model and load its weights.

    A one-user/one-movie dummy graph is enough to create the same graph
    metadata (including the reverse relation) and materialize lazy SAGEConv
    parameters before loading the checkpoint.
    """
    dummy = HeteroData()
    dummy["user"].x = torch.zeros((1, user_feature_dim), dtype=torch.float32)
    dummy["movie"].x = torch.zeros((1, movie_feature_dim), dtype=torch.float32)
    dummy[FORWARD_RELATION].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    dummy = T.ToUndirected()(dummy)

    model = Model(
        hidden_channels=HIDDEN_CHANNELS,
        data=dummy,
        dropout=DROPOUT,
    )

    # Materialize LazyLinear parameters for both directions and both layers.
    model.eval()
    with torch.no_grad():
        model.encode(dummy.x_dict, dummy.edge_index_dict)

    incompatible = model.load_state_dict(model_state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model architecture mismatch. "
            f"Missing: {incompatible.missing_keys}; "
            f"unexpected: {incompatible.unexpected_keys}"
        )

    model.to(device)
    model.eval()  # Disables dropout; required for deterministic inference.
    return model


def relation_conv(model: Model, layer: str, relation: tuple[str, str, str]):
    """Return one relation-specific SAGEConv generated by PyG to_hetero()."""
    relation_key = "__".join(relation)
    path = f"encoder.{layer}.{relation_key}"
    try:
        return model.get_submodule(path)
    except (AttributeError, KeyError) as exc:
        available = [name for name, _ in model.named_modules()]
        raise RuntimeError(
            f"Could not find relation module {path!r}. "
            "Check that models.py still uses to_hetero() with relations "
            f"{FORWARD_RELATION!r} and {REVERSE_RELATION!r}. "
            f"Available module paths include: {available[:40]}"
        ) from exc


def node_norm(model: Model, node_type: str):
    """Return the node-type-specific LayerNorm generated by to_hetero()."""
    path = f"encoder.norm1.{node_type}"
    try:
        return model.get_submodule(path)
    except (AttributeError, KeyError) as exc:
        available = [name for name, _ in model.named_modules()]
        raise RuntimeError(
            f"Could not find node normalization module {path!r}. "
            f"Available module paths include: {available[:40]}"
        ) from exc


def identity_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    """One synthetic edge i -> i per target, used to apply SAGEConv to means.

    Supplying the already-computed neighborhood mean as the single source
    feature for each target is mathematically equivalent to mean aggregation
    over that target's original neighborhood.
    """
    idx = torch.arange(num_nodes, dtype=torch.long, device=device)
    return torch.stack((idx, idx), dim=0)
