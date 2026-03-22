"""
main.py — Point d'entrée principal
          Lance l'entraînement, l'évaluation et la prédiction.

Usage :
    python main.py                        # entraînement complet
    python main.py --predict image.jpg    # prédiction seule
    python main.py --eval-only            # évaluation seule (charge best_model.pth)
"""
import argparse
import os
import torch

from dataset  import get_loaders
from model    import build_model
from train    import train
from evaluate import plot_training_curves, evaluate_test_set
from predict  import predict_image

# ── Configuration ─────────────────────────────────────────────
CONFIG = {
    'csv_path':   '/content/data/kaggle/train.csv',
    'img_dir':    '/content/data/kaggle/train',
    'batch_size': 64,
    'seed':       42,
    'num_epochs': 30,
    'lr':         0.01,
    'momentum':   0.9,
    'weight_decay': 0.0005,
    'save_path':  'best_model.pth',
}


def main():
    parser = argparse.ArgumentParser(description='AlexNet — Coin Classification')
    parser.add_argument('--predict',   type=str, default=None,
                        help='Chemin vers une image pour la prédiction')
    parser.add_argument('--eval-only', action='store_true',
                        help='Évaluation seule (charge best_model.pth)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── Chargement des données ────────────────────────────────
    train_loader, val_loader, test_loader, classes = get_loaders(
        csv_path=CONFIG['csv_path'],
        img_dir=CONFIG['img_dir'],
        batch_size=CONFIG['batch_size'],
        seed=CONFIG['seed']
    )
    num_classes = len(classes)

    # ── Modèle ───────────────────────────────────────────────
    model = build_model(num_classes=num_classes, device=device)

    # ── Mode prédiction seule ─────────────────────────────────
    if args.predict:
        if not os.path.exists(CONFIG['save_path']):
            print(f"✗ Modèle non trouvé : {CONFIG['save_path']}")
            return
        model.load_state_dict(torch.load(CONFIG['save_path'], map_location=device))
        predict_image(args.predict, model, classes, device)
        return

    # ── Mode évaluation seule ─────────────────────────────────
    if args.eval_only:
        if not os.path.exists(CONFIG['save_path']):
            print(f"✗ Modèle non trouvé : {CONFIG['save_path']}")
            return
        model.load_state_dict(torch.load(CONFIG['save_path'], map_location=device))
        evaluate_test_set(model, test_loader, classes, device,
                          best_val_acc=0.0)
        return

    # ── Entraînement complet ──────────────────────────────────
    history, best_epoch, best_val_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=CONFIG['num_epochs'],
        lr=CONFIG['lr'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay'],
        save_path=CONFIG['save_path']
    )

    # ── Courbes ───────────────────────────────────────────────
    plot_training_curves(history, best_epoch)

    # ── Évaluation finale sur le test set ─────────────────────
    print("\n── Évaluation finale sur le TEST SET ──")
    model.load_state_dict(torch.load(CONFIG['save_path'], map_location=device))
    evaluate_test_set(model, test_loader, classes, device, best_val_acc)


if __name__ == '__main__':
    main()
