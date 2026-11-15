"""
This files have hyperparamenters and constants values used for the code
"""

# Any movie rated with 3.5 or higher by a user is considered a positive interaction
POS_THRESHOLD = 3.5
# Any movie rated with 2 or lower by a user is considered a hard negative interaction
LOW_THRESHOLD = 2.0

"""
The dimensionality of the latent embedding space.

Both user and movie feature representations are projected into a
128-dimensional vector space across the GraphSAGE layers.
"""
HIDDEN_CHANNELS = 128

# The learning rate used by Adam Optimizer
LR = 3e-4
# L2 regularization on model weights
WEIGHT_DECAY = 1e-4
# L2 regularization specifically for BPR batch embeddings
LAMBDA_REG = 3e-4
DROPOUT = 0.10

"""
The number of positive user-movie target edges sampled per mini-batch by 
the PyG LinkNeighborLoader while training
"""
TRAIN_BATCH_SIZE = 1024

# Defines the neighborhood aggregation strategy for multi-hop graph convolutions
NUM_NEIGHBORS = [10, 5]

NEG_SAMPLING_RATIO = 3

"""
The ranking threshold used during model evaluation

Value used by Recall@K, Precision@K, F1@K, and NDCG@K
"""
K = 20

# The path used to save and reload trained model parameters, optimizer states, and training metrics.
CHECKPOINT_PATH = "graphsage_ml32m_checkpoint.pth"

# The total number of epochs during the trainning session
EPOCHS = 6
