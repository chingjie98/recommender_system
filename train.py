import torch

def train_model(model, dataloader, loss_function, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  
        running_rmse = 0.0
        
        for batch_users, batch_items, batch_ratings in dataloader:
            optimizer.zero_grad()  # Zero gradients
            
            # Forward pass
            predictions = model(batch_users, batch_items)
            loss = loss_function(predictions, batch_ratings)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate RMSE
            running_rmse += torch.sqrt(loss).item()

        print(f"Epoch {epoch + 1}/{num_epochs}, RMSE Loss: {running_rmse / len(dataloader):.4f}")
