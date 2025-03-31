import torch
import torch.nn as nn
import torchvision


class BCEWithLabelSmoothing(nn.Module):
    def __init__(self, number_of_classes, alpha=0.1):
        super(BCEWithLabelSmoothing, self).__init__()
        self.number_of_classes = number_of_classes
        self.alpha = alpha  # label smoothing hyperparameter
        self.bce = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        # y^LS =y_k(1−α)+α/K -> from "When Does Label Smoothing Help?" - https://arxiv.org/pdf/1906.02629
        targets = torch.clamp(targets, self.alpha / 59, (1 - self.alpha) + (self.alpha / self.number_of_classes))
        return self.bce(logits, targets)


class FocalWithLogitsLoss(nn.Module):
    def __init__(self):
        super(FocalWithLogitsLoss, self).__init__()
    
    def forward(self, logits, targets):
        return torchvision.ops.sigmoid_focal_loss(logits, targets)


class AsymmetricLossOptimized(nn.Module):
    """Notice - optimized version, minimizes memory allocation and gpu uploading, favors inplace operations
    From "Asymmetric Loss For Multi-Label Classification" - https://arxiv.org/abs/2009.14119
    https://github.com/Alibaba-MIIL/ASL/tree/main
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, logits, targets):
        """"
        Parameters
        ----------
        logits: input logits
        targets: multi-label binarized vector
        """

        self.targets = targets
        self.anti_targets = 1 - targets

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(logits)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class AsymmetricLossOptimized_LabelSmoothing(nn.Module):
    """Adapted from above to include """

    def __init__(self, number_of_classes, alpha=0.1, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized_LabelSmoothing, self).__init__()

        self.number_of_classes = number_of_classes
        self.alpha = alpha  # label smoothing hyperparameter
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, logits, targets):
        """"
        Parameters
        ----------
        logits: input logits
        targets: targets (multi-label binarized vector)
        """

        # y^LS =y_k(1−α)+α/K -> from "When Does Label Smoothing Help?" - https://arxiv.org/pdf/1906.02629
        targets = torch.clamp(targets, self.alpha / self.number_of_classes, (1 - self.alpha) + (self.alpha / self.number_of_classes))

        self.targets = targets
        self.anti_targets = 1 - targets

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(logits)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
