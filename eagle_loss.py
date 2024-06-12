import torch
import torch.nn as nn
import torch.nn.functional as F
import unfoldNd


class Eagle_Loss(nn.Module):
    """
    Eagle_Loss is a custom loss function designed for image reconstruction tasks, with an emphasis on preserving
    textures and edges in the reconstructed images. It operates by analyzing the variance of image gradients within
    patches of the image, and by computing loss in the frequency domain using a high-pass filter.

    Parameters:
        patch_size (int): Defines the size of the patches used to calculate the variance in gradients.
        cutoff (float): The cutoff frequency for the high-pass filter used in gaussian high-pass filtering.
        cpu (bool): Determines whether the kernels are stored on the CPU (if True) or on CUDA (if False).

    Methods:
        forward(output, target): Calculates the Eagle loss between the output and target images.
        calculate_gradient(img): Computes the x and y gradients of an image using the Scharr filters.
        calculate_patch_loss(output_gradient, target_gradient): Measures the loss based on the variance of
            gradients within the image patches.
        gaussian_highpass_weights2d(size): Generates the weights for high-pass filtering in the frequency domain.
        fft_loss(pred, gt): Calculates the loss in the frequency domain using the high-pass filter.

    Example Usage:
        eagle_loss = Eagle_Loss(patch_size=3)
        loss = eagle_loss(output_image, target_image)
    """
    def __init__(self, patch_size, cpu=False, cutoff=0.5):
        super(Eagle_Loss, self).__init__()
        self.patch_size = patch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        self.cutoff = cutoff

        # Scharr kernel for the gradient map calculation
        kernel_values = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        self.kernel_x = nn.Parameter(
            torch.tensor(kernel_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
            requires_grad=False)
        self.kernel_y = nn.Parameter(
            torch.tensor(kernel_values, dtype=torch.float32).t().unsqueeze(0).unsqueeze(0).to(self.device),
            requires_grad=False)

        # Operation for unfolding image into non-overlapping patches
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size).to(self.device)

    def forward(self, output, target):
        output, target = output.to(self.device), target.to(self.device)
        if output.size(1) != 1 or target.size(1) != 1:
            raise ValueError("Input 'output' and 'target' should be grayscale")

        # Gradient maps calculation
        gx_output, gy_output = self.calculate_gradient(output)
        gx_target, gy_target = self.calculate_gradient(target)

        # Unfolding and variance calculation
        eagle_loss = self.calculate_patch_loss(gx_output, gx_target) + \
                     self.calculate_patch_loss(gy_output, gy_target)

        return eagle_loss

    def calculate_gradient(self, img):
        img = img.to(self.device)
        gx = F.conv2d(img, self.kernel_x, padding=1, groups=img.shape[1])
        gy = F.conv2d(img, self.kernel_y, padding=1, groups=img.shape[1])
        return gx, gy

    def calculate_patch_loss(self, output_gradient, target_gradient):
        output_gradient, target_gradient = output_gradient.to(self.device), target_gradient.to(self.device)
        batch_size = output_gradient.size(0)
        num_channels = output_gradient.size(1)
        patch_size_squared = self.patch_size * self.patch_size
        output_patches = self.unfold(output_gradient).view(batch_size, num_channels, patch_size_squared, -1)
        target_patches = self.unfold(target_gradient).view(batch_size, num_channels, patch_size_squared, -1)
        var_output = torch.var(output_patches, dim=2, keepdim=True)
        var_target = torch.var(target_patches, dim=2, keepdim=True)

        shape0, shape1 = output_gradient.shape[-2] // self.patch_size, output_gradient.shape[-1] // self.patch_size
        return self.fft_loss(var_target.view(batch_size, num_channels, shape0, shape1), var_output.view(batch_size, num_channels, shape0, shape1))

    def gaussian_highpass_weights2d(self, size):
        freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1).repeat(1, size[1]).to(self.device)
        freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1).repeat(size[0], 1).to(self.device)

        freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        weights = torch.exp(-0.5 * ((freq_mag - self.cutoff) ** 2))
        return 1 - weights  # Inverted for high pass

    def fft_loss(self, pred, gt):
        pred, gt = pred.to(self.device), gt.to(self.device)
        pred_fft = torch.fft.fft2(pred)
        gt_fft = torch.fft.fft2(gt)
        pred_mag = torch.sqrt(pred_fft.real ** 2 + pred_fft.imag ** 2)
        gt_mag = torch.sqrt(gt_fft.real ** 2 + gt_fft.imag ** 2)

        weights = self.gaussian_highpass_weights2d(pred.size()[2:]).to(pred.device)
        weighted_pred_mag = weights * pred_mag
        weighted_gt_mag = weights * gt_mag

        return F.l1_loss(weighted_pred_mag, weighted_gt_mag)


class Eagle_Loss_3D(nn.Module):
    """
    3D version of Eagle-Loss, not tested yet.

    Attributes:
        patch_size (int): The size of the patch to calculate local variance.
        device (torch.device): The device on which to perform computations. 
                               Defaults to CUDA if available, else CPU.
        cutoff (float): The cutoff frequency for the high-pass filter used in FFT.
        kernel_x, kernel_y, kernel_z (torch.FloatTensor): 3D kernels for gradient calculation in x, y, and z directions.
        unfold (unfoldNd.UnfoldNd): An UnfoldNd object to extract patches from 3D volumes.

    Methods:
        gaussian_highpass_weights_3d(size): Generates Gaussian high-pass filter weights for given size.
        fft_loss_3d(pred, gt): Calculates the loss in frequency domain using Fast Fourier Transform.
        forward(output, target): Computes the loss given the network's output and the target.

    Example:
        # Initialize loss function
        eagle_loss = Eagle_Loss_3D(patch_size=3)
        # Calculate loss
        loss = eagle_loss(output, target)
    """
    def __init__(self, patch_size, device=None, cutoff=0.5):
        super(Eagle_Loss_3D, self).__init__()
        self.patch_size = patch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cutoff = cutoff

        self.kernel_x = torch.FloatTensor([
            [[-9, 0, 9], [-30, 0, 30], [-9, 0, 9]],
            [[-30, 0, 30], [-100, 0, 100], [-30, 0, 30]],
            [[-9, 0, 9], [-30, 0, 30], [-9, 0, 9]]
        ]).unsqueeze(0).unsqueeze(0).to(self.device)

        self.kernel_y = torch.FloatTensor([
            [[-9, -30, -9], [0, 0, 0], [9, 30, 9]],
            [[-30, -100, -30], [0, 0, 0], [30, 100, 30]],
            [[-9, -30, -9], [0, 0, 0], [9, 30, 9]]
        ]).unsqueeze(0).unsqueeze(0).to(self.device)

        self.kernel_z = torch.FloatTensor([
            [[-9, -30, -9], [-30, -100, -30], [-9, -30, -9]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[9, 30, 9], [30, 100, 30], [9, 30, 9]]
        ]).unsqueeze(0).unsqueeze(0).to(self.device)

        self.unfold = unfoldNd.UnfoldNd(kernel_size=self.patch_size, stride=self.patch_size)

    def gaussian_highpass_weights_3d(self, size):
        freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1, 1).repeat(1, size[1], size[2]).to(self.device)
        freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1, 1).repeat(size[0], 1, size[2]).to(self.device)
        freq_z = torch.fft.fftfreq(size[2]).reshape(1, 1, -1).repeat(size[0], size[1], 1).to(self.device)

        freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2 + freq_z ** 2)

        weights = torch.exp(-0.5 * ((freq_mag - self.cutoff) ** 2))
        weights = 1 - weights

        return weights

    def fft_loss_3d(self, pred, gt):
        spatial_size = pred.shape[-3:]

        pred_fft = torch.fft.fftn(pred)
        gt_fft = torch.fft.fftn(gt)

        pred_mag = torch.sqrt(pred_fft.real.pow(2) + pred_fft.imag.pow(2))
        gt_mag = torch.sqrt(gt_fft.real.pow(2) + gt_fft.imag.pow(2))

        weights = self.gaussian_highpass_weights_3d(spatial_size)
        weighted_pred_mag = weights * pred_mag
        weighted_gt_mag = weights * gt_mag

        loss = F.l1_loss(weighted_pred_mag, weighted_gt_mag)
        return loss

    def forward(self, output, target):
        output = output.to(self.device)
        target = target.to(self.device)

        gx_target = F.conv3d(target, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv3d(target, self.kernel_y, stride=1, padding=1)
        gz_target = F.conv3d(target, self.kernel_z, stride=1, padding=1)
        gx_output = F.conv3d(output, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv3d(output, self.kernel_y, stride=1, padding=1)
        gz_output = F.conv3d(output, self.kernel_z, stride=1, padding=1)

        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gz_target_patches = self.unfold(gz_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)
        gz_output_patches = self.unfold(gz_output)

        var_target_x = torch.var(gx_target_patches, dim=1)
        shape0, shape1, shape2 = target.shape[-3] // self.patch_size, target.shape[-2] // self.patch_size, target.shape[
            -1] // self.patch_size
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)
        var_target_z = torch.var(gz_target_patches, dim=1)
        var_output_z = torch.var(gz_output_patches, dim=1)

        egale_loss_3d = (self.fft_loss_3d(var_target_x.reshape(1, shape0, shape1, shape2),
                                          var_output_x.reshape(1, shape0, shape1, shape2)) +
                         self.fft_loss_3d(var_target_y.reshape(1, shape0, shape1, shape2),
                                          var_output_y.reshape(1, shape0, shape1, shape2)) +
                         self.fft_loss_3d(var_target_z.reshape(1, shape0, shape1, shape2),
                                          var_output_z.reshape(1, shape0, shape1, shape2)))

        return egale_loss_3d
