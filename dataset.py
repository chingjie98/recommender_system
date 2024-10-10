import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class RatingDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.users = torch.tensor(data['userId'].values - 1)  # Zero Index the userId
        
        # Use LabelEncoder to make movieId contiguous, meaning continuous
        self.movie_encoder = LabelEncoder()
        self.movies = torch.tensor(self.movie_encoder.fit_transform(data['movieId']), dtype=torch.long)
        
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]
