import torch
import torch.nn as nn
import torch.nn.functional as F


class Eagle_Loss(nn.Module):
    def __init__(self, patch_size, cpu=False):
        super(Eagle_Loss, self).__init__()
        self.patch_size = patch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')

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
        gx = F.conv2d(img, self.kernel_x, padding=1)
        gy = F.conv2d(img, self.kernel_y, padding=1)
        return gx, gy

    def calculate_patch_loss(self, output_gradient, target_gradient):
        output_gradient, target_gradient = output_gradient.to(self.device), target_gradient.to(self.device)
        output_patches = self.unfold(output_gradient)
        target_patches = self.unfold(target_gradient)
        var_output = torch.var(output_patches, dim=1, keepdim=True)
        var_target = torch.var(target_patches, dim=1, keepdim=True)

        shape0, shape1 = output_gradient.shape[-2] // self.patch_size, output_gradient.shape[-1] // self.patch_size
        return self.fft_loss(var_target.reshape(1, shape0, shape1), var_output.reshape(1, shape0, shape1))

    def gaussian_highpass_weights2d(self, size, cutoff=0.2, strength=1):
        freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1).repeat(1, size[1]).to(self.device)
        freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1).repeat(size[0], 1).to(self.device)

        freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        weights = torch.exp(-0.5 * ((freq_mag - cutoff) ** 2) / (strength ** 2))
        return 1 - weights  # Inverted for high pass

    def fft_loss(self, pred, gt, cutoff=0.5, strength=1):
        pred, gt = pred.to(self.device), gt.to(self.device)
        pred_fft = torch.fft.fft2(pred)
        gt_fft = torch.fft.fft2(gt)
        pred_mag = torch.sqrt(pred_fft.real ** 2 + pred_fft.imag ** 2)
        gt_mag = torch.sqrt(gt_fft.real ** 2 + gt_fft.imag ** 2)

        weights = self.gaussian_highpass_weights2d(pred.size(), cutoff=cutoff, strength=strength).to(pred.device)
        weighted_pred_mag = weights * pred_mag
        weighted_gt_mag = weights * gt_mag

        return F.l1_loss(weighted_pred_mag, weighted_gt_mag)
