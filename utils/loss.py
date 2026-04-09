# ============================================================
# LOSS FUNCTIONS
# ============================================================

import torch
import torch.nn as nn   

class FocalMSELoss(nn.Module):
    """
    Focal version of MSE for regression.
    Penalizes large errors more than small ones.
    """

    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        # error term
        error = preds - targets

        # standard MSE
        mse = error ** 2

        # focal modulation term
        focal_weight = (1.0 - torch.exp(-torch.abs(error))) ** self.gamma

        # final focal-MSE
        loss = focal_weight * mse

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def seasonal_weight(month, monsoon_weight=3.0, non_monsoon_weight=1.0):
    """
    Assign higher weights to monsoon months.
    month: tensor of shape (B,)
    """
    return torch.where(
        (month >= 6) & (month <= 9),
        torch.tensor(monsoon_weight, device=month.device),
        torch.tensor(non_monsoon_weight, device=month.device),
    )

class SeasonalFocalMSELoss(nn.Module):
    """
    Focal MSE with seasonal weighting.
    Penalizes large errors more during monsoon months.
    """

    def __init__(
        self,
        gamma=2.0,
        monsoon_weight=3.0,
        non_monsoon_weight=1.0,
        reduction="mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.monsoon_weight = monsoon_weight
        self.non_monsoon_weight = non_monsoon_weight
        self.reduction = reduction

    def forward(self, preds, targets, month):
        # prediction error
        error = preds - targets

        # base MSE
        mse = error ** 2

        # focal modulation (same as Phase 1)
        focal_term = (1.0 - torch.exp(-torch.abs(error))) ** self.gamma

        # seasonal weighting
        season_w = seasonal_weight(
            month,
            self.monsoon_weight,
            self.non_monsoon_weight,
        )

        # combined loss
        loss = season_w * focal_term * mse

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
