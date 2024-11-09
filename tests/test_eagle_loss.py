import pytest
import torch
from eagle_loss import Eagle_Loss

@pytest.fixture
def sample_2d_data():
    batch_size = 2
    channels = 1
    height = 64
    width = 64
    
    # Create sample output and target tensors
    output = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    return output, target

@pytest.fixture
def sample_rgb_data():
    batch_size = 2
    channels = 3
    height = 64
    width = 64
    
    # Create sample RGB output and target tensors
    output = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    return output, target

def test_2d_eagle_loss_initialization():
    loss_fn = Eagle_Loss(patch_size=3)
    assert loss_fn.patch_size == 3
    assert isinstance(loss_fn.kernel_x, torch.nn.Parameter)
    assert isinstance(loss_fn.kernel_y, torch.nn.Parameter)
    assert loss_fn.cutoff == 0.5  # Test default cutoff value

def test_2d_eagle_loss_forward(sample_2d_data):
    output, target = sample_2d_data
    loss_fn = Eagle_Loss(patch_size=3)
    loss = loss_fn(output, target)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Should be a scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_rgb_to_grayscale_conversion(sample_rgb_data):
    output, target = sample_rgb_data
    loss_fn = Eagle_Loss(patch_size=3)
    
    # Test RGB to grayscale conversion
    gray_output = loss_fn.rgb_to_grayscale(output)
    gray_target = loss_fn.rgb_to_grayscale(target)
    
    assert gray_output.size(1) == 1
    assert gray_target.size(1) == 1

def test_gradient_calculation():
    loss_fn = Eagle_Loss(patch_size=3)
    # Create a simple edge pattern
    img = torch.zeros(1, 1, 64, 64)
    img[:, :, :, 32:] = 1.0  # Vertical edge
    
    gx, gy = loss_fn.calculate_gradient(img)
    
    # Check if vertical edge is detected in x gradient
    assert torch.any(gx[:, :, :, 31:33] != 0)
    # Check if y gradient is mostly zero
    assert torch.sum(torch.abs(gy)) < torch.sum(torch.abs(gx))

def test_patch_size_requirements():
    # Test different patch sizes
    for patch_size in [2, 4, 8]:
        loss_fn = Eagle_Loss(patch_size=patch_size)
        output = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)
        
        loss = loss_fn(output, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

def test_cpu_device_option():
    loss_fn = Eagle_Loss(patch_size=3, cpu=True)
    assert loss_fn.device.type == "cpu"
    
    output = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    loss = loss_fn(output, target)
    assert loss.device.type == "cpu"

def test_cutoff_parameter():
    # Test different cutoff values
    for cutoff in [0.3, 0.5, 0.7]:
        loss_fn = Eagle_Loss(patch_size=3, cutoff=cutoff)
        assert loss_fn.cutoff == cutoff
        
        output = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)
        loss = loss_fn(output, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

def test_batch_consistency():
    loss_fn = Eagle_Loss(patch_size=3)
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        output = torch.randn(batch_size, 1, 64, 64)
        target = torch.randn(batch_size, 1, 64, 64)
        
        loss = loss_fn(output, target)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

def test_training_and_gradients():
    # Create a simple model that processes images
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
            
        def forward(self, x):
            return self.conv(x)
    
    # Set up the model, loss, and optimizer
    model = SimpleModel()
    loss_fn = Eagle_Loss(patch_size=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    
    # Create sample data
    torch.manual_seed(42)
    input_image = torch.randn(2, 1, 64, 64, requires_grad=True)
    target_image = torch.randn(2, 1, 64, 64)
    
    # Run initial forward pass
    output = model(input_image)
    initial_loss = loss_fn(output, target_image)
    
    # Test forward and backward pass
    for _ in range(5):
        optimizer.zero_grad()
        output = model(input_image)
        loss = loss_fn(output, target_image)
        
        assert loss.ndim == 0
        loss.backward()
        
        assert model.conv.weight.grad is not None
        assert not torch.isnan(model.conv.weight.grad).any()
        
        optimizer.step()
    
    # Check if final loss is less than initial loss
    final_output = model(input_image)
    final_loss = loss_fn(final_output, target_image)
    assert final_loss < initial_loss, f"Loss did not decrease: initial {initial_loss.item()}, final {final_loss.item()}"

def test_fft_loss_calculation():
    loss_fn = Eagle_Loss(patch_size=3)
    
    # Create simple periodic patterns
    x = torch.linspace(0, 2*torch.pi, 64)
    y = torch.linspace(0, 2*torch.pi, 64)
    # Use indexing='ij' to fix the warning
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create patterns with different frequencies
    pattern1 = torch.sin(X) + torch.cos(Y)
    pattern2 = torch.sin(2*X) + torch.cos(2*Y)
    
    pattern1 = pattern1.unsqueeze(0).unsqueeze(0)
    pattern2 = pattern2.unsqueeze(0).unsqueeze(0)
    
    loss = loss_fn.fft_loss(pattern1, pattern2)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)