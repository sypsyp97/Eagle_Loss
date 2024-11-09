import torch
import torch.nn as nn
from eagle_loss import Eagle_Loss
import torchvision.transforms as transforms
from PIL import Image

def load_and_process_image(image_path, size=(256, 256)):
    """Load and process an image for the model"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define a simple autoencoder model
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model and move to device
    model = SimpleAutoencoder().to(device)
    
    # Initialize Eagle Loss
    eagle_loss = Eagle_Loss(patch_size=3).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create some example data (random noise images)
    batch_size = 16
    input_images = torch.randn(batch_size, 3, 256, 256).to(device)
    target_images = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print("Starting training loop...")
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_images)
        
        # Calculate loss using Eagle Loss
        loss = eagle_loss(outputs, target_images)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training finished!")

    # Example of using the model for inference
    print("\nRunning inference example...")
    model.eval()
    with torch.no_grad():
        # Create a test image (you could load a real image here)
        test_image = torch.randn(1, 3, 256, 256).to(device)
        
        # Run inference
        output = model(test_image)
        
        # Calculate Eagle Loss for the output
        test_loss = eagle_loss(output, test_image)
        print(f"Test image Eagle Loss: {test_loss.item():.4f}")

if __name__ == "__main__":
    main()