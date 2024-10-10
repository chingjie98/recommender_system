path = "dataset/ml-latest-small/"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RatingDataset
from model import MatrixFactorizationNN
from train import train_model
from config import Config

def main():
    df = pd.read_csv(path + "ratings.csv")

    # Initialize dataset and dataloader
    dataset = RatingDataset(df)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    # Model configuration
    num_users = df['userId'].nunique()
    num_items = df['movieId'].nunique()

    # Initialize model
    model = MatrixFactorizationNN(num_users, num_items, Config.num_factors)

    # Loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate, momentum=Config.momentum)

    # Train the model
    train_model(model, dataloader, loss_function, optimizer, Config.num_epochs)

if __name__ == "__main__":
    main()
