from torch.utils.data import DataLoader
from torchvision import transforms
from model import UNet
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from eagle_loss import Eagle_Loss
import matplotlib.patches as patches
from datetime import datetime
from dataset import FOVDataset


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize the pixel values of an image to the range [0, 1]."""
    min_val = image.min()
    max_val = image.max()

    return (image - min_val) / (max_val - min_val)


def compute_gradient_map(image):
    dx, dy = np.gradient(image)
    return np.sqrt(dx ** 2 + dy ** 2)


def compute_second_order_gradient(image):
    ddx, ddy = np.gradient(compute_gradient_map(image))
    return np.sqrt(ddx ** 2 + ddy ** 2)


def ssim_custom(im1, im2, data_range):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = np.mean(im1)
    mu2 = np.mean(im2)

    sigma1_sq = np.var(im1)
    sigma2_sq = np.var(im2)
    sigma12 = np.cov(im1.flatten(), im2.flatten())[0, 1]

    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_val


def psnr_custom(target, ref, data_range):
    mse = np.mean((target - ref) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(data_range / math.sqrt(mse))


def mse_custom(target, ref):
    return np.mean((target - ref) ** 2)


def metrics_with_std(model, test_loader, device):
    model.eval()
    model.to(device)

    ssim_values = []
    psnr_values = []
    mse_values = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).cpu().numpy()
            targets = targets.cpu().numpy()

            for i in range(outputs.shape[0]):
                ssim_val = ssim_custom(outputs[i], targets[i], data_range=1)
                psnr_val = psnr_custom(outputs[i], targets[i], data_range=1)
                mse_val = mse_custom(outputs[i], targets[i])

                ssim_values.append(ssim_val)
                psnr_values.append(psnr_val)
                mse_values.append(mse_val)

    # Compute the mean and standard deviation of the metrics
    avg_ssim, std_ssim = np.mean(ssim_values), np.std(ssim_values)
    avg_psnr, std_psnr = np.mean(psnr_values), np.std(psnr_values)
    avg_mse, std_mse = np.mean(mse_values), np.std(mse_values)

    return (avg_ssim, std_ssim), (avg_psnr, std_psnr), (avg_mse, std_mse)


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_dir = f"results_{timestamp}"

    os.makedirs(results_dir, exist_ok=True)

    def normalize_to_zero_to_one(tensor):
        min_val = tensor.min()
        range_val = tensor.max() - min_val
        if range_val > 0:
            normalized = (tensor - min_val) / range_val
        else:
            normalized = tensor - min_val
        return normalized

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((512, 512), antialias=True),
        transforms.Lambda(normalize_to_zero_to_one)
    ])

    train_dataset = FOVDataset('/mnt/home/sun/miccai-2024/fovct/fov_data/train', transform=transform)
    val_dataset = FOVDataset('/mnt/home/sun/miccai-2024/fovct/fov_data/val', transform=transform)
    test_dataset = FOVDataset('/mnt/home/sun/miccai-2024/fovct/fov_data/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_list = [Eagle_Loss(patch_size=3)
                 # , torch.nn.MSELoss(), SSIMLoss(window_size=11, size_average=True), CombinedTVandMSELoss(),
                 # GradientVariance(), GEE_Loss(), PerceptualLoss()
                 ]

    for criterion in loss_list:

        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')

        model = UNet().to(device)
        num_epochs = 100
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader),
                                                        epochs=num_epochs)

        loss_name = type(criterion).__name__
        model_save_path = f"{results_dir}/best_model_{loss_name}.pth"

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for batch_idx, (data, groundtruth) in enumerate(train_loader):
                data, groundtruth = data.cuda(), groundtruth.cuda()
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, groundtruth)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            # Validation phase
            if val_loader:
                model.eval()
                val_running_loss = 0.0
                with torch.no_grad():
                    for data, groundtruth in val_loader:
                        data, groundtruth = data.cuda(), groundtruth.cuda()
                        outputs = model(data)
                        loss = criterion(outputs, groundtruth)
                        val_running_loss += loss.item()

                avg_val_loss = val_running_loss / len(val_loader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss

                    torch.save(model.state_dict(), model_save_path)

                val_loss_history.append(avg_val_loss)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path))

        model.eval()
        model.to(device)

        zoom_coords = [360, 280, 80, 80]  # Example coordinates

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                if i == 41:  # choose a specific image
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    # Visualization
                    label_img = labels[0].cpu().squeeze()  # first label image
                    output_img = outputs[0].cpu().squeeze()  # first output image

                    image_names = ['Ground_Truth', f'reconstruction_{loss_name}']

                    for i, img in enumerate(
                            [normalize_image(label_img),
                             normalize_image(output_img)]):
                        circle_center = (img.shape[1] // 2, img.shape[0] // 2)  # Center of the image
                        circle_radius = 252 // 2  # The Radius is half the diameter

                        # Display and save the main image
                        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
                        ax.imshow(img, cmap='gray', interpolation='hanning')
                        ax.axis('off')
                        circle = patches.Circle(circle_center, circle_radius, linewidth=1.5, edgecolor='b',
                                                linestyle='--',
                                                facecolor='none')
                        ax.add_patch(circle)
                        rect = patches.Rectangle((zoom_coords[0], zoom_coords[1]), zoom_coords[2], zoom_coords[3],
                                                 linewidth=1.5, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        plt.tight_layout()
                        main_image_path = os.path.join(results_dir, f"{image_names[i]}_main.png")
                        plt.savefig(main_image_path)
                        plt.close(fig)  # Close the figure to free memory

                        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
                        ax.imshow(img, cmap='gray', interpolation='hanning')
                        ax.axis('off')

                        plt.tight_layout()
                        main_image_path = os.path.join(results_dir, f"{image_names[i]}_ori.png")
                        plt.savefig(main_image_path)
                        plt.close(fig)

                        # Display and save the zoomed image
                        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
                        ax.imshow(img, cmap='gray', interpolation='hanning')
                        ax.axis('off')
                        ax.set_xlim(zoom_coords[0], zoom_coords[0] + zoom_coords[2])
                        ax.set_ylim(zoom_coords[1] + zoom_coords[3], zoom_coords[1])
                        plt.tight_layout()
                        zoomed_image_path = os.path.join(results_dir, f"{image_names[i]}_zoomed.png")
                        plt.savefig(zoomed_image_path)
                        plt.close(fig)

                    break  # stop after the first batch

        # Calculate metrics
        (ssim_avg, ssim_std), (psnr_avg, psnr_std), (mse_avg, mse_std) = metrics_with_std(model, test_loader, device)
        results_str = f"""
        SSIM: Mean = {ssim_avg} ± {ssim_std}
        PSNR: Mean = {psnr_avg} ± {psnr_std}
        MSE: Mean = {mse_avg} ± {mse_std}
        """
        with open(f'{results_dir}/results_{loss_name}.txt', 'w') as file:
            file.write(results_str)


if __name__ == "__main__":
    main()
