"""
train.py — Boucle d'entraînement AlexNet
          Hyperparamètres du paper AlexNet (Krizhevsky et al. 2012) :
          SGD, lr=0.01, momentum=0.9, weight_decay=0.0005
          Scheduler : ReduceLROnPlateau (÷10 si val stagne)
"""
import time
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Une epoch d'entraînement. Retourne (loss, accuracy)."""
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, 100 * correct / total


def evaluate(model, loader, criterion, device):
    """Évaluation sur val ou test. Retourne (loss, accuracy)."""
    model.eval()
    total_loss = correct = total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, 100 * correct / total


def train(model, train_loader, val_loader, device,
          num_epochs: int = 30,
          lr: float = 0.01,
          momentum: float = 0.9,
          weight_decay: float = 0.0005,
          save_path: str = 'best_model.pth'):
    """
    Entraînement complet avec :
    - SGD + momentum (paper AlexNet)
    - ReduceLROnPlateau (÷10 si val_acc stagne 5 epochs)
    - Sauvegarde automatique du meilleur modèle
    Retourne : history dict
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }
    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(v_acc)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        flag = ''
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), save_path)
            flag = '  ★ sauvegardé'

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}% | "
              f"LR: {current_lr:.6f} | "
              f"{time.time()-t0:.1f}s{flag}")

    print(f"\n✓ Meilleur epoch : {best_epoch}  —  Val Acc : {best_val_acc:.2f}%")
    return history, best_epoch, best_val_acc
