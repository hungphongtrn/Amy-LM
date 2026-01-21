# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility module to handle adversarial losses without requiring to mess up the main training loop.
"""

import contextlib
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


def broadcast_model(model: nn.Module):
    """Broadcast model parameters from rank 0 to other ranks."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for p in model.parameters():
            torch.distributed.broadcast(p, src=0)
        for b in model.buffers():
            torch.distributed.broadcast(b, src=0)


@contextlib.contextmanager
def readonly(model: nn.Module):
    """Context manager to temporarily disable gradient calculation for a module."""
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad = False
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            param.requires_grad = requires_grad[name]


ADVERSARIAL_LOSSES = ["mse", "hinge", "hinge2"]


AdvLossType = tp.Union[nn.Module, tp.Callable[[torch.Tensor], torch.Tensor]]
FeatLossType = tp.Union[
    nn.Module, tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
]


class AdversarialLoss(nn.Module):
    """Adversary training wrapper.

    Args:
        adversary (nn.Module): The adversary module will be used to estimate the logits given the fake and real samples.
            We assume here the adversary output is ``Tuple[List[torch.Tensor], List[List[torch.Tensor]]]``
            where the first item is a list of logits and the second item is a list of feature maps.
        optimizer (torch.optim.Optimizer): Optimizer used for training the given module.
        loss (AdvLossType): Loss function for generator training.
        loss_real (AdvLossType): Loss function for adversarial training on logits from real samples.
        loss_fake (AdvLossType): Loss function for adversarial training on logits from fake samples.
        loss_feat (FeatLossType): Feature matching loss function for generator training.
        normalize (bool): Whether to normalize by number of sub-discriminators.

    Example of usage:
        adv_loss = AdversarialLoss(adversaries, optimizer, loss, loss_real, loss_fake)
        for real in loader:
            noise = torch.randn(...)
            fake = model(noise)
            adv_loss.train_adv(fake, real)
            loss, _ = adv_loss(fake, real)
            loss.backward()
    """

    def __init__(
        self,
        adversary: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: AdvLossType,
        loss_real: AdvLossType,
        loss_fake: AdvLossType,
        loss_feat: tp.Optional[FeatLossType] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.adversary: nn.Module = adversary
        broadcast_model(self.adversary)
        self.optimizer = optimizer
        self.loss = loss
        self.loss_real = loss_real
        self.loss_fake = loss_fake
        self.loss_feat = loss_feat
        self.normalize = normalize

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Add the optimizer state dict inside our own.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.optimizer is not None:
            destination[prefix + "optimizer"] = self.optimizer.state_dict()
        return destination

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Load optimizer state.
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict.pop(prefix + "optimizer"))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_adversary_pred(self, x):
        """Run adversary model, validating expected output format."""
        logits, fmaps = self.adversary(x)
        assert isinstance(logits, list) and all(
            [isinstance(t, torch.Tensor) for t in logits]
        ), f"Expecting a list of tensors as logits but {type(logits)} found."
        assert isinstance(fmaps, list), (
            f"Expecting a list of features maps but {type(fmaps)} found."
        )
        for fmap in fmaps:
            assert isinstance(fmap, list) and all(
                [isinstance(f, torch.Tensor) for f in fmap]
            ), f"Expecting a list of tensors as feature maps but {type(fmap)} found."
        return logits, fmaps

    def train_adv(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Train the adversary with the given fake and real example.

        We assume the adversary output is the following format: Tuple[List[torch.Tensor], List[List[torch.Tensor]]].
        The first item being the logits and second item being a list of feature maps for each sub-discriminator.

        This will automatically synchronize gradients (with `flashy.distrib.eager_sync_model`)
        and call the optimizer.
        """
        loss = torch.tensor(0.0, device=fake.device)
        all_logits_fake_is_fake, _ = self.get_adversary_pred(fake.detach())
        all_logits_real_is_fake, _ = self.get_adversary_pred(real.detach())
        n_sub_adversaries = len(all_logits_fake_is_fake)
        for logit_fake_is_fake, logit_real_is_fake in zip(
            all_logits_fake_is_fake, all_logits_real_is_fake
        ):
            loss += self.loss_fake(logit_fake_is_fake) + self.loss_real(
                logit_real_is_fake
            )

        if self.normalize:
            loss /= n_sub_adversaries

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def forward(
        self, fake: torch.Tensor, real: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Return the loss for the generator, i.e. trying to fool the adversary,
        and feature matching loss if provided.
        """
        adv = torch.tensor(0.0, device=fake.device)
        feat = torch.tensor(0.0, device=fake.device)
        with readonly(self.adversary):
            all_logits_fake_is_fake, all_fmap_fake = self.get_adversary_pred(fake)
            n_sub_adversaries = len(all_logits_fake_is_fake)
            for logit_fake_is_fake in all_logits_fake_is_fake:
                adv += self.loss(logit_fake_is_fake)
            if self.loss_feat:
                # Compute real features without graph for feature matching
                with torch.no_grad():
                    _, all_fmap_real = self.get_adversary_pred(real)
                for fmap_fake, fmap_real in zip(all_fmap_fake, all_fmap_real):
                    feat += self.loss_feat(fmap_fake, fmap_real)

        if self.normalize:
            adv /= n_sub_adversaries
            feat /= n_sub_adversaries

        return adv, feat


def get_adv_criterion(loss_type: str) -> tp.Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == "mse":
        return mse_loss
    elif loss_type == "hinge":
        return hinge_loss
    elif loss_type == "hinge2":
        return hinge2_loss
    raise ValueError("Unsupported loss")


def get_fake_criterion(loss_type: str) -> tp.Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == "mse":
        return mse_fake_loss
    elif loss_type in ["hinge", "hinge2"]:
        return hinge_fake_loss
    raise ValueError("Unsupported loss")


def get_real_criterion(loss_type: str) -> tp.Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == "mse":
        return mse_real_loss
    elif loss_type in ["hinge", "hinge2"]:
        return hinge_real_loss
    raise ValueError("Unsupported loss")


def mse_real_loss(x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, torch.tensor(1.0, device=x.device).expand_as(x))


def mse_fake_loss(x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, torch.tensor(0.0, device=x.device).expand_as(x))


def hinge_real_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(
        torch.min(x - 1, torch.tensor(0.0, device=x.device).expand_as(x))
    )


def hinge_fake_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(
        torch.min(-x - 1, torch.tensor(0.0, device=x.device).expand_as(x))
    )


def mse_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0], device=x.device)
    return F.mse_loss(x, torch.tensor(1.0, device=x.device).expand_as(x))


def hinge_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0], device=x.device)
    return -x.mean()


def hinge2_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0])
    return -torch.mean(
        torch.min(x - 1, torch.tensor(0.0, device=x.device).expand_as(x))
    )


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for adversarial training.

    Args:
        loss (nn.Module): Loss to use for feature matching (default=torch.nn.L1).
        normalize (bool): Whether to normalize the loss.
            by number of feature maps.
    """

    def __init__(self, loss: nn.Module = torch.nn.L1Loss(), normalize: bool = True):
        super().__init__()
        self.loss = loss
        self.normalize = normalize

    def forward(
        self, fmap_fake: tp.List[torch.Tensor], fmap_real: tp.List[torch.Tensor]
    ) -> torch.Tensor:
        assert len(fmap_fake) == len(fmap_real) and len(fmap_fake) > 0
        feat_loss = torch.tensor(0.0, device=fmap_fake[0].device)
        feat_scale = torch.tensor(0.0, device=fmap_fake[0].device)
        n_fmaps = 0
        for feat_fake, feat_real in zip(fmap_fake, fmap_real):
            assert feat_fake.shape == feat_real.shape
            n_fmaps += 1
            feat_loss += self.loss(feat_fake, feat_real)
            feat_scale += torch.mean(torch.abs(feat_real))

        if self.normalize:
            feat_loss /= n_fmaps

        return feat_loss


def test_adversarial_losses():
    print("Running tests for adversarial_losses...")

    # Test readonly
    model = nn.Linear(10, 10)
    for p in model.parameters():
        p.requires_grad = True

    with readonly(model):
        for p in model.parameters():
            assert not p.requires_grad, "Parameters should not require grad inside readonly"

    for p in model.parameters():
        assert p.requires_grad, "Parameters should require grad after readonly"
    print("  readonly: OK")

    # Test Loss Getters
    for name in ADVERSARIAL_LOSSES:
        assert callable(get_adv_criterion(name))
        assert callable(get_fake_criterion(name))
        assert callable(get_real_criterion(name))
    print("  Loss getters: OK")

    # Test FeatureMatchingLoss
    fm_loss = FeatureMatchingLoss()
    fmap_fake = [torch.randn(2, 4, 4), torch.randn(2, 4, 4)]
    fmap_real = [torch.randn(2, 4, 4), torch.randn(2, 4, 4)]
    loss = fm_loss(fmap_fake, fmap_real)
    assert loss.dim() == 0
    print("  FeatureMatchingLoss: OK")

    # Test AdversarialLoss
    class MockAdversary(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            # Returns logits (list of tensors) and fmaps (list of list of tensors)
            # Simulating 2 sub-discriminators
            logits = [torch.randn(x.size(0), 1, requires_grad=True), torch.randn(x.size(0), 1, requires_grad=True)]
            fmaps = [
                [torch.randn(x.size(0), 10, requires_grad=True), torch.randn(x.size(0), 10, requires_grad=True)],
                [torch.randn(x.size(0), 10, requires_grad=True), torch.randn(x.size(0), 10, requires_grad=True)]
            ]
            return logits, fmaps

    adversary = MockAdversary()
    optimizer = torch.optim.SGD(adversary.parameters(), lr=0.01)

    adv_loss = AdversarialLoss(
        adversary=adversary,
        optimizer=optimizer,
        loss=get_adv_criterion("hinge"),
        loss_real=get_real_criterion("hinge"),
        loss_fake=get_fake_criterion("hinge"),
        loss_feat=FeatureMatchingLoss(),
        normalize=True
    )

    fake = torch.randn(4, 10)
    real = torch.randn(4, 10)

    # Test train_adv
    loss_val = adv_loss.train_adv(fake, real)
    assert isinstance(loss_val, torch.Tensor)
    print("  AdversarialLoss.train_adv: OK")

    # Test forward (generator loss)
    adv, feat = adv_loss(fake, real)
    assert isinstance(adv, torch.Tensor)
    assert isinstance(feat, torch.Tensor)
    print("  AdversarialLoss.forward: OK")

    # Test with optimizer=None (for manual optimization cases like Lightning)
    adv_loss_no_opt = AdversarialLoss(
        adversary=adversary,
        optimizer=None,
        loss=get_adv_criterion("hinge"),
        loss_real=get_real_criterion("hinge"),
        loss_fake=get_fake_criterion("hinge"),
        loss_feat=FeatureMatchingLoss(),
        normalize=True
    )
    
    # Test state_dict with None optimizer
    state_dict = adv_loss_no_opt.state_dict()
    assert "optimizer" not in state_dict or state_dict.get("optimizer") is None
    
    # Test load_state_dict with None optimizer (should not fail)
    adv_loss_no_opt.load_state_dict(state_dict)
    print("  AdversarialLoss with optimizer=None: OK")

    print("All tests passed!")


if __name__ == "__main__":
    test_adversarial_losses()
