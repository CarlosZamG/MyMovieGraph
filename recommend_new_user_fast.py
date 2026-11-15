"""Fast cold-start inference without loading the historical graph or ratings.csv."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from config import K
from users_features import initialize_user_features

from fast_utils import (
    FORWARD_RELATION,
    REVERSE_RELATION,
    build_initialized_model,
    identity_edge_index,
    node_norm,
    relation_conv,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_BUNDLE = BASE_DIR / "inference_bundle"
DEFAULT_FAST_BUNDLE = BASE_DIR / "fast_inference_bundle"


def load_fast_resources(
    base_bundle_dir: str | Path = DEFAULT_BASE_BUNDLE,
    fast_bundle_dir: str | Path = DEFAULT_FAST_BUNDLE,
    device: str = "cpu",
    allow_validation_bundle: bool = False,
) -> dict:
    """Load only the compact files required at application runtime.

    This function deliberately does NOT load:
        - ratings.csv
        - edge_index.npy
        - user_ids.npy
        - movie_embeddings_pca.npz
        - checkpoint['user_embeddings']

    The full historical graph is used only by the offline preprocessing step.
    """
    base_dir = Path(base_bundle_dir)
    fast_dir = Path(fast_bundle_dir)
    device = torch.device(device)

    with (base_dir / "metadata.json").open("r", encoding="utf-8") as f:
        base_metadata = json.load(f)
    with (fast_dir / "metadata.json").open("r", encoding="utf-8") as f:
        fast_metadata = json.load(f)

    if fast_metadata.get("excluded_mapped_users") and not allow_validation_bundle:
        raise ValueError(
            "The supplied fast bundle excludes users and appears to be a "
            "validation bundle. Set allow_validation_bundle=True only in evaluation."
        )

    num_movies = int(base_metadata["graph"]["num_movies"])
    if int(fast_metadata["num_movies"]) != num_movies:
        raise ValueError("Base and fast bundles contain different movie counts.")

    movie_ids = np.load(base_dir / "movie_ids.npy").astype(np.int64, copy=False)
    movies_df = pd.read_csv(
        base_dir / "movie_metadata.csv",
        dtype={"movieId": np.int64},
    )

    if len(movie_ids) != num_movies or len(movies_df) != num_movies:
        raise ValueError("Movie ID mapping / metadata size mismatch.")
    if not np.array_equal(movies_df["movieId"].to_numpy(np.int64), movie_ids):
        raise ValueError("movie_metadata.csv and movie_ids.npy are not aligned.")

    movie_features_np = np.load(base_dir / "movie_features.npy", mmap_mode="r")
    movie_features = torch.from_numpy(
        np.array(movie_features_np, dtype=np.float32, copy=True)
    ).contiguous()
    del movie_features_np

    movie_embeddings_np = np.load(fast_dir / "movie_embeddings.npy", mmap_mode="r")
    movie_embeddings = torch.from_numpy(
        np.array(movie_embeddings_np, dtype=np.float32, copy=True)
    ).contiguous()
    del movie_embeddings_np

    movie_user_feature_sum_np = np.load(
        fast_dir / "movie_user_feature_sum.npy", mmap_mode="r"
    )
    movie_user_feature_sum = torch.from_numpy(
        np.array(movie_user_feature_sum_np, dtype=np.float32, copy=True)
    ).contiguous()
    del movie_user_feature_sum_np

    movie_user_degree_np = np.load(
        fast_dir / "movie_user_degree.npy", mmap_mode="r"
    )
    movie_user_degree = torch.from_numpy(
        np.array(movie_user_degree_np, dtype=np.int64, copy=True)
    ).contiguous()
    del movie_user_degree_np

    if movie_features.size(0) != num_movies:
        raise ValueError("movie_features.npy has an unexpected number of rows.")
    if movie_embeddings.size(0) != num_movies:
        raise ValueError("movie_embeddings.npy has an unexpected number of rows.")
    if movie_user_feature_sum.size(0) != num_movies:
        raise ValueError("movie_user_feature_sum.npy has an unexpected number of rows.")
    if movie_user_degree.numel() != num_movies:
        raise ValueError("movie_user_degree.npy has an unexpected number of rows.")

    if movie_user_feature_sum.size(1) != int(fast_metadata["user_feature_dim"]):
        raise ValueError("Historical user feature aggregate dimension mismatch.")
    if movie_features.size(1) != int(fast_metadata["movie_feature_dim"]):
        raise ValueError("Movie feature dimension mismatch.")
    if movie_embeddings.size(1) != int(fast_metadata["embedding_dim"]):
        raise ValueError("Movie embedding dimension mismatch.")

    model_state_dict = torch.load(
        fast_dir / "model_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = build_initialized_model(
        model_state_dict=model_state_dict,
        user_feature_dim=int(fast_metadata["user_feature_dim"]),
        movie_feature_dim=int(fast_metadata["movie_feature_dim"]),
        device=device,
    )

    # Retain the exact trained relation-specific layers used by the decomposition.
    modules = {
        "conv1_user_to_movie": relation_conv(model, "conv1", FORWARD_RELATION),
        "conv1_movie_to_user": relation_conv(model, "conv1", REVERSE_RELATION),
        "conv2_movie_to_user": relation_conv(model, "conv2", REVERSE_RELATION),
        "norm1_user": node_norm(model, "user"),
        "norm1_movie": node_norm(model, "movie"),
    }

    if device.type != "cpu":
        movie_features = movie_features.to(device)
        movie_embeddings = movie_embeddings.to(device)
        movie_user_feature_sum = movie_user_feature_sum.to(device)
        movie_user_degree = movie_user_degree.to(device)

    movie_id_to_mapped = {
        int(movie_id): int(index)
        for index, movie_id in enumerate(movie_ids)
    }

    return {
        "movies_df": movies_df,
        "movie_ids": movie_ids,
        "movie_id_to_mapped": movie_id_to_mapped,
        "movie_features": movie_features,
        "movie_embeddings": movie_embeddings,
        "movie_user_feature_sum": movie_user_feature_sum,
        "movie_user_degree": movie_user_degree,
        "model": model,
        "modules": modules,
        "device": device,
    }


def _prepare_new_user_input(
    new_user_ratings: dict,
    resources: dict,
) -> tuple[torch.Tensor, list[int], list[float]]:
    """Map known movie IDs and compute the new user's initial feature vector.

    The current initialize_user_features() API indexes its output by the
    supplied user IDs. Since this call creates exactly one temporary user,
    all its rating rows use local user index 0 and num_users=1. This avoids
    allocating a feature matrix for all historical users on every request.

    This is equivalent to using a global appended user index when the feature
    initializer groups/accumulates features by the supplied user index (the
    contract implied by its current call site).
    """
    valid_mapped_movies: list[int] = []
    valid_ratings: list[float] = []

    for raw_movie_id, raw_rating in new_user_ratings.items():
        try:
            movie_id = int(raw_movie_id)
            rating = float(raw_rating)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(rating):
            continue

        mapped = resources["movie_id_to_mapped"].get(movie_id)
        if mapped is not None:
            valid_mapped_movies.append(mapped)
            valid_ratings.append(rating)

    if not valid_mapped_movies:
        raise ValueError(
            "None of the provided movies exist in the MovieLens mappings."
        )

    movie_tensor = torch.tensor(valid_mapped_movies, dtype=torch.long, device="cpu")
    rating_tensor = torch.tensor(valid_ratings, dtype=torch.float32, device="cpu")
    local_user_tensor = torch.zeros(len(valid_mapped_movies), dtype=torch.long, device="cpu")

    # Keep this input on CPU: initialize_user_features() in the current project
    # receives CPU tensors and the CPU movie feature matrix.
    movie_features_cpu = resources["movie_features"].detach().cpu()

    user_feature_table = initialize_user_features(
        users=local_user_tensor,
        movies=movie_tensor,
        ratings=rating_tensor,
        movie_features=movie_features_cpu,
        num_users=1,
    )

    if not torch.is_tensor(user_feature_table) or user_feature_table.ndim != 2:
        raise TypeError(
            "initialize_user_features() must return a 2D torch.Tensor."
        )
    if user_feature_table.size(0) != 1:
        raise ValueError(
            "The one-user call to initialize_user_features() returned "
            f"{user_feature_table.size(0)} rows, expected 1. Inspect that "
            "function before using the local-user-index optimization."
        )

    new_user_feature = user_feature_table[0:1].to(
        device=resources["device"],
        dtype=torch.float32,
    )

    expected_user_dim = resources["movie_user_feature_sum"].size(1)
    if new_user_feature.size(1) != expected_user_dim:
        raise ValueError(
            "New user feature dimension does not match historical user "
            f"features: got {new_user_feature.size(1)}, expected {expected_user_dim}."
        )

    return new_user_feature, valid_mapped_movies, valid_ratings


@torch.no_grad()
def score_new_user_all_movies(
    new_user_ratings: dict,
    preloaded_resources: dict,
) -> tuple[torch.Tensor, list[int]]:
    """Compute scores for all movies without loading historical edges.

    Exact for the saved two-layer GraphSAGE architecture under full-neighbor
    mean aggregation and eval mode, for candidate movies not rated by the new
    user. Scores of rated movies are set to -inf.
    """
    resources = preloaded_resources
    device = resources["device"]

    new_user_feature, rated_mapped_movies, _ = _prepare_new_user_input(
        new_user_ratings,
        resources,
    )

    rated_idx = torch.tensor(rated_mapped_movies, dtype=torch.long, device=device)
    rated_movie_features = resources["movie_features"].index_select(0, rated_idx)

    modules = resources["modules"]
    n_rated = rated_idx.numel()

    # -------------------------------------------------------------
    # 1. New user's first-layer embedding
    # -------------------------------------------------------------
    # Every rated movie sends one message to the new user. The target user
    # has local index 0; source movie indices are 0..n_rated-1.
    user_edge_index = torch.stack(
        (
            torch.arange(n_rated, dtype=torch.long, device=device),
            torch.zeros(n_rated, dtype=torch.long, device=device),
        ),
        dim=0,
    )

    h_new_user_1 = modules["conv1_movie_to_user"](
        (rated_movie_features, new_user_feature),
        user_edge_index,
    )
    h_new_user_1 = modules["norm1_user"](h_new_user_1).relu()

    # -------------------------------------------------------------
    # 2. Update first-layer movie embeddings for rated movies only
    # -------------------------------------------------------------
    # In the historical graph, each rated movie has:
    #   sum_x = sum of historical users' input features
    #   degree = number of historical user neighbors
    # Adding this new user changes the mean to (sum_x + x_new)/(degree+1).
    historical_sum = resources["movie_user_feature_sum"].index_select(0, rated_idx)
    historical_degree = resources["movie_user_degree"].index_select(0, rated_idx)

    updated_degree = historical_degree.to(torch.float32).unsqueeze(1) + 1.0
    updated_user_mean = (historical_sum + new_user_feature) / updated_degree

    rated_identity_edges = identity_edge_index(n_rated, device)

    h_rated_movies_1 = modules["conv1_user_to_movie"](
        (updated_user_mean, rated_movie_features),
        rated_identity_edges,
    )
    h_rated_movies_1 = modules["norm1_movie"](h_rated_movies_1).relu()

    # -------------------------------------------------------------
    # 3. New user's second-layer embedding
    # -------------------------------------------------------------
    z_new_user = modules["conv2_movie_to_user"](
        (h_rated_movies_1, h_new_user_1),
        user_edge_index,
    )

    # -------------------------------------------------------------
    # 4. Dot product against precomputed final movie embeddings
    # -------------------------------------------------------------
    scores = resources["movie_embeddings"] @ z_new_user.squeeze(0)
    scores = scores.clone()
    scores[rated_idx] = -float("inf")

    return scores, rated_mapped_movies


def recommend_for_new_user_fast(
    new_user_ratings: dict,
    top_k: int = K,
    preloaded_resources: Optional[dict] = None,
    base_bundle_dir: str | Path = DEFAULT_BASE_BUNDLE,
    fast_bundle_dir: str | Path = DEFAULT_FAST_BUNDLE,
    device: str = "cpu",
) -> pd.DataFrame:
    """Return top-K recommendations using compact precomputed resources."""
    resources = preloaded_resources
    if resources is None:
        resources = load_fast_resources(
            base_bundle_dir=base_bundle_dir,
            fast_bundle_dir=fast_bundle_dir,
            device=device,
        )

    scores, rated_mapped_movies = score_new_user_all_movies(
        new_user_ratings=new_user_ratings,
        preloaded_resources=resources,
    )

    num_available = scores.numel() - len(set(rated_mapped_movies))
    if top_k <= 0 or top_k > num_available:
        raise ValueError(f"top_k must be between 1 and {num_available}.")

    top_scores, top_indices = torch.topk(scores, k=top_k, sorted=True)
    top_mapped = top_indices.detach().cpu().numpy().astype(np.int64)
    top_original_ids = resources["movie_ids"][top_mapped]

    recommendations = resources["movies_df"].loc[
        resources["movies_df"]["movieId"].isin(top_original_ids)
    ].copy()

    score_by_movie_id = {
        int(resources["movie_ids"][mapped_idx]): float(score)
        for mapped_idx, score in zip(
            top_mapped,
            top_scores.detach().cpu().tolist(),
        )
    }

    rank_by_movie_id = {
        int(movie_id): rank
        for rank, movie_id in enumerate(top_original_ids.tolist())
    }

    recommendations["score"] = recommendations["movieId"].map(score_by_movie_id)
    recommendations["sort_key"] = recommendations["movieId"].map(rank_by_movie_id)
    recommendations = recommendations.sort_values("sort_key").drop(columns="sort_key")

    return recommendations.reset_index(drop=True)


if __name__ == "__main__":
    # Same Batman profile used in the original recommender.
    user_2 = {
        33794: 4,
        91529: 4,
        58559: 5,
        136864: 0,
        122898: 1,
    }

    resources = load_fast_resources()
    recommendations = recommend_for_new_user_fast(
        user_2,
        top_k=K,
        preloaded_resources=resources,
    )
    print(recommendations[["title", "score"]].to_string(index=False))
