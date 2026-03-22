# Coin Classification — AlexNet

Classification de pièces de monnaie par CNN (AlexNet) sur le dataset Kaggle [DL4CV Coin Classification](https://www.kaggle.com/competitions/dl4cv-coin-classification/overview).

**Entrée** : image d'une pièce  
**Sortie** : `"1 Dollar, Australian dollar, australia"`

---

## Résultats

| Métrique | Valeur |
|---|---|
| Meilleur epoch | 28 / 30 |
| Val Accuracy | **71.46%** |
| Train Accuracy | 94.20% |
| Nombre de classes | 315 |
| Référence hasard | 0.32% |
| GPU utilisé | NVIDIA T4 (Google Colab) |

---

## Architecture

AlexNet modifié — différences vs paper original (2012) :

| Élément | Paper | Notre version |
|---|---|---|
| Normalisation | LocalResponseNorm | **BatchNorm2d** (convergence ×10) |
| Biais Conv/Linear | bias=True | **bias=False** (redondant avec BN) |
| Nb classes | 1000 | **315** |
| Initialisation | Gaussienne std=0.01 | Identique |

---

## Structure du projet

```
coin_classification/
├── main.py        # Point d'entrée principal
├── dataset.py     # Chargement, scan, split, DataLoaders
├── model.py       # Architecture AlexNet
├── train.py       # Boucle d'entraînement
├── evaluate.py    # Courbes, matrice de confusion, rapport
├── predict.py     # Inférence sur une image
└── README.md
```

---

## Installation

```bash
pip install torch torchvision matplotlib seaborn scikit-learn pandas pillow
```

---

## Utilisation

### Entraînement complet

```bash
python main.py
```

Modifie les chemins dans `CONFIG` dans `main.py` :

```python
CONFIG = {
    'csv_path': '/content/data/kaggle/train.csv',
    'img_dir':  '/content/data/kaggle/train',
    ...
}
```

### Prédiction sur une image

```bash
python main.py --predict chemin/vers/image.jpg
```

### Évaluation seule (charge `best_model.pth`)

```bash
python main.py --eval-only
```

---

## Hyperparamètres (fidèles au paper AlexNet)

```python
optimizer    = SGD(lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler    = ReduceLROnPlateau(factor=0.1, patience=5)
criterion    = CrossEntropyLoss()
batch_size   = 64
num_epochs   = 30
input_size   = 227 × 227
```

---

## Pipeline de données

```
train.csv
    ↓
scan_valid_images()     ← filtre les images corrompues
    ↓
train_test_split()      ← 80% train / 10% val / 10% test (stratifié)
    ↓
CoinDataset
    ↓
DataLoader (batch_size=64)
```

### Augmentation (train uniquement)

```
Resize(256×256)
→ RandomRotation(180)   ← rotation AVANT crop (absorbe les coins noirs)
→ RandomCrop(227)
→ RandomHorizontalFlip
→ ColorJitter
→ Normalize (ImageNet stats)
```

---

## Classes difficiles

Les pièces d'Euro de différents pays partagent le même design européen — elles ne se distinguent que par la face nationale, ce qui les rend très difficiles à classifier (0% accuracy sur plusieurs sous-classes Euro).

```
50 Cents, Euro, france   → 0%
50 Cents, Euro, germany  → 0%
50 Cents, Euro, cyprus   → 0%
```

---

## Axes d'amélioration

- **Transfer learning** : AlexNet ou ResNet-50 pré-entraîné sur ImageNet → +20-30% attendu
- **Weighted CrossEntropyLoss** : corriger le déséquilibre entre classes
- **Test Time Augmentation** : moyenne sur 10 crops comme dans le paper original
- **Architecture moderne** : EfficientNet-B0, MobileNetV3
- **Fine-grained classification** : attention maps pour les pièces Euro similaires

---

## Référence

> Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).  
> *ImageNet classification with deep convolutional neural networks.*  
> Advances in neural information processing systems, 25.
