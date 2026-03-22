"""
predict.py — Inférence sur une image avec Top-5 et visualisation
"""
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def predict_image(img_path: str, model, classes: list, device,
                  save_path: str = 'prediction.png') -> str:
    """
    Prédit la classe d'une image et affiche le Top-5.
    Retourne le nom de la classe prédite.
    """
    tf = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Erreur lecture image : {e}")
        return None

    tensor = tf(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        top5_p, top5_i = torch.topk(probs, 5)

    labels_top5 = [classes[i][:30] for i in top5_i.cpu().numpy()]
    probs_top5  = top5_p.cpu().numpy() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title(f"→ {classes[top5_i[0]]}", fontsize=9)
    axes[0].axis('off')

    axes[1].barh(labels_top5[::-1], probs_top5[::-1],
                 color=['green'] + ['steelblue'] * 4)
    axes[1].set_xlabel('Probabilité (%)')
    axes[1].set_title('Top 5 prédictions')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Résultat  : {classes[top5_i[0]]}")
    print(f"Confiance : {probs_top5[0]:.2f}%")
    return classes[top5_i[0]]
