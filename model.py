"""
model.py — Architecture AlexNet adaptée pour la classification de pièces
           Modifications vs paper original :
           - BatchNorm2d remplace LocalResponseNorm (convergence ×10 plus rapide)
           - bias=False sur Conv et Linear (BatchNorm intègre déjà le biais)
           - Nombre de classes adapté (315 au lieu de 1000)
"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet modifié pour la classification de pièces de monnaie.

    Architecture :
        Conv1(96, 11×11, s=4) → BN → ReLU → MaxPool
        Conv2(256, 5×5, p=2)  → BN → ReLU → MaxPool
        Conv3(384, 3×3, p=1)  → BN → ReLU
        Conv4(384, 3×3, p=1)  → BN → ReLU
        Conv5(256, 3×3, p=1)  → BN → ReLU → MaxPool
        FC1(9216→4096) → BN → ReLU → Dropout(0.5)
        FC2(4096→4096) → BN → ReLU → Dropout(0.5)
        FC3(4096→num_classes)
    """

    def __init__(self, num_classes: int = 315):
        super().__init__()

        # ── Couches convolutives ──────────────────────────────────
        self.features = nn.Sequential(
            # [0] Conv1 — bias=False (BN prend en charge)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # [4] Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # [8] Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            # [11] Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            # [14] Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # ── Classifieur entièrement connecté ─────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialisation des poids selon le paper AlexNet (Gaussienne std=0.01)."""
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return self.classifier(x)


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Instancie et déplace le modèle sur le device."""
    model = AlexNet(num_classes=num_classes).to(device)

    # Vérification des dimensions
    dummy = torch.zeros(2, 3, 227, 227).to(device)
    out = model(dummy)
    assert out.shape == (2, num_classes), f"Sortie inattendue : {out.shape}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ AlexNet — {num_classes} classes — {n_params:,} paramètres")
    return model
