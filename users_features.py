import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T

from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

def initialize_user_features(users, movies, ratings, movie_features,num_users, chunk_size = 100000):
    
    if num_users is None:
        num_users = users.max().item() + 1

    feat_dim = movie_features.size(1)

    # MovieLens ratings are in {0.5, 1.0, ..., 5.0}
    bins = ((ratings * 2 - 1).long()).clamp(min=0, max=9)

    hist = torch.bincount(
        users * 10 + bins,
        minlength=num_users * 10
    ).reshape(num_users, 10)

    counts = hist.sum(dim=1)

    # max(1, int(0.1 * number_of_ratings))
    min_count = torch.floor(
        counts.float() * 0.1
    ).long().clamp_min(1)

    rating_bins = torch.arange(
        10,
        dtype=torch.long,
        device=hist.device
    ).unsqueeze(0)

    valid = hist >= min_count.unsqueeze(1)

    has_valid = valid.any(dim=1)

    # First and last valid rating bins
    lower_bin = torch.where(
        valid,
        rating_bins,
        torch.full_like(rating_bins, 10)
    ).amin(dim=1)

    upper_bin = torch.where(
        valid,
        rating_bins,
        torch.full_like(rating_bins, -1)
    ).amax(dim=1)

    lower_rating = torch.full(
        (num_users,),
        3.5,
        dtype=torch.float32
    )

    upper_rating = torch.full(
        (num_users,),
        3.5,
        dtype=torch.float32
    )

    lower_rating[has_valid] = (
        lower_bin[has_valid].float() + 1
    ) / 2.0

    upper_rating[has_valid] = (
        upper_bin[has_valid].float() + 1
    ) / 2.0

    fixed_bounds = (
        (lower_rating >= 2.5) |
        (upper_rating <= 3.0)
    )

    positive_bound = torch.full(
        (num_users,),
        3.5,
        dtype=torch.float32
    )

    negative_bound = torch.full(
        (num_users,),
        2.0,
        dtype=torch.float32
    )

    flexible = has_valid & ~fixed_bounds

    mid_rating = (
        lower_rating + upper_rating
    ) / 2.0

    positive_bound[flexible] = (
        mid_rating[flexible] + 0.6
    )

    negative_bound[flexible] = (
        mid_rating[flexible] - 0.6
    )


    liked_sum = torch.zeros(
        (num_users, feat_dim),
        dtype=movie_features.dtype
    )

    disliked_sum = torch.zeros_like(liked_sum)

    liked_weight_sum = torch.zeros(
        num_users,
        dtype=movie_features.dtype
    )

    disliked_weight_sum = torch.zeros_like(
        liked_weight_sum
    )

    n = users.numel()

    for start in range(0, n, chunk_size):

        end = min(start + chunk_size, n)

        user_chunk = users[start:end]
        movie_chunk = movies[start:end]
        rating_chunk = ratings[start:end]

        user_positive_bound = positive_bound[user_chunk]
        user_negative_bound = negative_bound[user_chunk]

        liked_mask = (
            rating_chunk >= user_positive_bound
        )

        disliked_mask = (
            rating_chunk <= user_negative_bound
        )


        if liked_mask.any():

            liked_users = user_chunk[liked_mask]
            liked_movies = movie_chunk[liked_mask]
            liked_ratings = rating_chunk[liked_mask]

            liked_feats = movie_features[liked_movies]
            weighted_feats = (
                liked_feats *
                liked_ratings.unsqueeze(1)
            )

            liked_sum.index_add_(
                0,
                liked_users,
                weighted_feats
            )

            liked_weight_sum.index_add_(
                0,
                liked_users,
                liked_ratings
            )

        if disliked_mask.any():

            disliked_users = user_chunk[disliked_mask]
            disliked_movies = movie_chunk[disliked_mask]
            disliked_ratings = rating_chunk[disliked_mask]

            disliked_feats = movie_features[
                disliked_movies
            ]

            weights = 5.0 - disliked_ratings

            weighted_feats = (
                disliked_feats *
                weights.unsqueeze(1)
            )

            disliked_sum.index_add_(
                0,
                disliked_users,
                weighted_feats
            )

            disliked_weight_sum.index_add_(
                0,
                disliked_users,
                weights
            )

    liked_profiles = torch.zeros_like(liked_sum)

    disliked_profiles = torch.zeros_like(disliked_sum)

    liked_nonzero = liked_weight_sum > 0
    disliked_nonzero = disliked_weight_sum > 0

    liked_profiles[liked_nonzero] = (
        liked_sum[liked_nonzero] /
        liked_weight_sum[liked_nonzero].unsqueeze(1)
    )

    disliked_profiles[disliked_nonzero] = (
        disliked_sum[disliked_nonzero] /
        disliked_weight_sum[disliked_nonzero].unsqueeze(1)
    )

    return torch.cat(
        [liked_profiles, disliked_profiles],
        dim=1
    )


def reduced_users_features(embeddings):
    movies_df = pd.read_csv("./ml-32m/movies.csv")
    ratings_df = pd.read_csv("./ml-32m/ratings.csv")

    print(embeddings.shape)

    # Create a mapping from the userId to a unique consecutive value in
    # the range [0, num_users]:
    unique_user_id = ratings_df['userId'].unique()
    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id, 
        'mappedUserId': pd.RangeIndex(len(unique_user_id))
        })
    
    # Create a mapping from the movieId to a unique consecutive value
    # in the range [0, num_movies]:
    unique_movie_id = ratings_df['movieId'].unique()
    unique_movie_id = pd.DataFrame(data={
        'movieId': unique_movie_id,
        'mappedMovieId': pd.RangeIndex(len(unique_movie_id))
        })
    
    # Merge the mappings with the original data frame:
    ratings_df = ratings_df.merge(unique_user_id, on='userId')
    ratings_df = ratings_df.merge(unique_movie_id, on='movieId')

    # Align movie features with mappedMovieId order
    # movies_df["reduce_idx"] track original array row
    movies_df["reduce_idx"] = np.arange(len(movies_df)) 
    movies_df = movies_df.merge(unique_movie_id, on='movieId').sort_values('mappedMovieId')

    scaler = StandardScaler()

    # Align movie embeddings and genres 
    aligned_indices = movies_df["reduce_idx"].values
    aligned_embeddings = embeddings[aligned_indices]
    aligned_embeddings_scaled = scaler.fit_transform(aligned_embeddings)
    aligned_genres = movies_df["genres"].str.get_dummies('|').values.astype(np.float32)

    print(aligned_genres.shape)

    movie_features = np.concatenate([aligned_embeddings_scaled, aligned_genres], axis=1)
    movie_features = torch.from_numpy(movie_features).float()
    

    # With this, we are ready to create the edge_index representation 
    # in COO format
    # following the PyTorch Geometric semantics:
    edge_index = torch.stack([
        torch.tensor(ratings_df['mappedUserId'].values), 
        torch.tensor(ratings_df['mappedMovieId'].values)], dim=0)

    assert edge_index.shape == (2, len(ratings_df))

    data = HeteroData()

    data['user'].num_nodes = len(ratings_df['userId'].unique())
    data['movie'].x = movie_features  # [num_movies, num_features_movies]

    # Add the rating edges:
    data['user', 'rates', 'movie'].edge_index = edge_index  # [2, num_ratings]

    # Add the rating labels:
    ratings_cp = ratings_df['rating'].values.copy()
    rating = torch.from_numpy(ratings_cp).to(torch.float)
    data['user', 'rates', 'movie'].edge_label = rating  # [num_ratings]

    # Add the reverse edges from movies to users in order to let a GNN 
    # be able to pass messages in both directions.
    data = T.ToUndirected()(data)
    print()
    print(data)

    user_features = initialize_user_features(data, movie_features)
    return user_features


if __name__ == "__main__":

    movie_embeddings_data = np.load("movie_embeddings_pca.npz")
    embeddings = movie_embeddings_data["pca"]
    user_features_pca = reduced_users_features(embeddings)
    torch.save(user_features_pca, 'user_features_pca.pt')

    movie_embeddings_data = np.load("movie_embeddings_umap.npz")
    embeddings = movie_embeddings_data["umap"]
    user_features_umap = reduced_users_features(embeddings)
    torch.save(user_features_umap, 'user_features_umap.pt')

