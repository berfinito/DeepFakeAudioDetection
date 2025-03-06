import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
from BasicCNN import BasicCNN
import torch.nn as nn
import torch.optim as optim
from dataset_preparation import train_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)


epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for input, labels in tqdm.tqdm(train_loader):
        input, labels = input.to(device), labels.to(device)


        optimizer.zero_grad()

        outputs = model(input)
        # print(labels)
        # print(outputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_preds / total_preds
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")