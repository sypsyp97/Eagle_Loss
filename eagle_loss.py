import torch
import torch.nn as nn
import torch.nn.functional as F


class Eagle_Loss(nn.Module):
    """
    Eagle_Loss: A custom loss function for image reconstruction tasks.

    The loss function is particularly suitable for tasks where preservation of
    textures and edges is crucial.

    Attributes:
        patch_size (int): The size of the non-overlapping patches into which the
                          image is divided for loss calculation.
        kernel_x (nn.Parameter): Horizontal Scharr filter for gradient calculation.
        kernel_y (nn.Parameter): Vertical Scharr filter for gradient calculation.
        unfold (nn.Unfold): Operation for unfolding image into patches.

    Parameters:
        patch_size (int): The size of the patches for calculating variance.
        cpu (bool): If True, the kernels are kept on the CPU, otherwise on CUDA.

    Methods:
        forward(output, target): Computes the Eagle loss between the output and target.
        calculate_gradient(img): Calculates the x and y gradients of an image.
        calculate_patch_loss(output_gradient, target_gradient): Calculates the loss
                                   based on the variance of gradients in image patches.
        gaussian_highpass_weights2d(size, cutoff, strength): Generates weights for
                                                             high-pass filtering in
                                                             the frequency domain.
        fft_loss(pred, gt, cutoff, strength): Computes the loss in the frequency domain
                                              using the high-pass filter.

    Example:
        eagle_loss = Eagle_Loss(patch_size=3)
        loss = eagle_loss(output_image, target_image)
    """

    def __init__(self, patch_size=3, cpu=False):
        super(Eagle_Loss, self).__init__()
        self.patch_size = patch_size

        # Scharr kernel for the gradient map calculation
        kernel_values = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        self.kernel_x = nn.Parameter(torch.tensor(kernel_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                     requires_grad=False)
        self.kernel_y = nn.Parameter(torch.tensor(kernel_values, dtype=torch.float32).t().unsqueeze(0).unsqueeze(0),
                                     requires_grad=False)

        # Ensure kernels are on the correct device
        if not cpu:
            self.kernel_x = self.kernel_x.cuda()
            self.kernel_y = self.kernel_y.cuda()

        # Operation for unfolding image into non-overlapping patches
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, output, target):
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
        gx = F.conv2d(img, self.kernel_x, padding=1)
        gy = F.conv2d(img, self.kernel_y, padding=1)
        return gx, gy

    def calculate_patch_loss(self, output_gradient, target_gradient):
        output_patches = self.unfold(output_gradient)
        target_patches = self.unfold(target_gradient)
        var_output = torch.var(output_patches, dim=1, keepdim=True)
        var_target = torch.var(target_patches, dim=1, keepdim=True)

        shape0, shape1 = output_gradient.shape[-2] // self.patch_size, output_gradient.shape[-1] // self.patch_size
        return self.fft_loss(var_target.reshape(1, shape0, shape1), var_output.reshape(1, shape0, shape1))

    @staticmethod
    def gaussian_highpass_weights2d(size, cutoff=0.5, strength=1):
        freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1).repeat(1, size[1])
        freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1).repeat(size[0], 1)
        freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        weights = torch.exp(-0.5 * ((freq_mag - cutoff) ** 2) / (strength ** 2))
        return 1 - weights  # Inverted for high-pass

    @staticmethod
    def fft_loss(pred, gt, cutoff=0.5, strength=1):
        pred_fft = torch.fft.fft2(pred)
        gt_fft = torch.fft.fft2(gt)
        pred_mag = torch.sqrt(pred_fft.real ** 2 + pred_fft.imag ** 2)
        gt_mag = torch.sqrt(gt_fft.real ** 2 + gt_fft.imag ** 2)

        weights = Eagle_Loss.gaussian_highpass_weights2d(pred.size(), cutoff=cutoff, strength=strength).to(pred.device)
        weighted_pred_mag = weights * pred_mag
        weighted_gt_mag = weights * gt_mag

        return F.l1_loss(weighted_pred_mag, weighted_gt_mag)