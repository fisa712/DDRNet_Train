import torch
import torch.nn as nn
from DDRNet_23_slim import get_seg_model
from dataloader import train_loader



cfg = None
NUM_CLASSES = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_seg_model(cfg, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Zero the gradients

        outputs = model(images)  # Forward pass

        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")




def evaluate(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # Add evaluation metrics, e.g., IoU, pixel accuracy, etc.

torch.save(model.state_dict(), 'ddrnet_custom.pth')

# To load it later:
# model.load_state_dict(torch.load('ddrnet_custom.pth'))

# eval on your test data now