import torch.nn as nn

class MatrixFactorizationNN(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(MatrixFactorizationNN, self).__init__()
        
        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)

    def forward(self, user, item):
        # Get user and item embeddings
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)

        # Predicted rating is the dot product of the user and item embeddings
        predicted_rating = (user_embedded * item_embedded).sum(1)
        return predicted_rating

