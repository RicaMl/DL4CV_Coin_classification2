"""
evaluate.py — Évaluation finale sur le test set
              Génère : courbes, matrice de confusion, rapport sklearn
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_curves(history: dict, best_epoch: int,
                         save_path: str = 'courbes_entrainement.png'):
    """Affiche et sauvegarde les courbes loss et accuracy."""
    num_epochs = len(history['train_loss'])
    epochs_range = range(1, num_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Courbes entraînement — AlexNet', fontsize=13, fontweight='bold')

    for ax, (k_train, k_val), title in zip(
        axes,
        [('train_loss', 'val_loss'), ('train_acc', 'val_acc')],
        ['Loss', 'Accuracy (%)']
    ):
        ax.plot(epochs_range, history[k_train], 'b-o', markersize=4, label='Train')
        ax.plot(epochs_range, history[k_val],   'r-o', markersize=4, label='Val')
        ax.axvline(x=best_epoch, color='green', linestyle='--',
                   linewidth=1.5, label=f'Best epoch ({best_epoch})')
        ax.fill_between(epochs_range, history[k_train], history[k_val],
                        alpha=0.08, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Courbes sauvegardées → {save_path}")


def evaluate_test_set(model, test_loader, classes: list, device,
                      best_val_acc: float,
                      save_path: str = 'confusion_matrix_test.png'):
    """
    Évaluation finale sur le test set.
    Affiche : accuracy, rapport sklearn, matrice de confusion,
              top-10 meilleures et difficiles classes.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            all_preds.extend(model(imgs).argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    print("=" * 60)
    print(f"  Val  Accuracy (meilleur epoch) : {best_val_acc:.2f}%")
    print(f"  Test Accuracy finale           : {test_acc:.2f}%")
    print("=" * 60)

    # Rapport
    print("\n── Rapport de classification ──")
    print(classification_report(all_labels, all_preds,
                                target_names=classes, zero_division=0))

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(24, 22))
    sns.heatmap(cm, cmap='Blues', annot=False,
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Matrice de confusion — Test Acc : {test_acc:.2f}%', fontsize=13)
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0,  fontsize=5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\n── Top 10 meilleures classes ──")
    for i in np.argsort(per_class_acc)[-10:][::-1]:
        print(f"  {classes[i][:45]:<45} {per_class_acc[i]*100:.1f}%")

    print("\n── Top 10 classes difficiles ──")
    for i in np.argsort(per_class_acc)[:10]:
        print(f"  {classes[i][:45]:<45} {per_class_acc[i]*100:.1f}%")

    return test_acc
