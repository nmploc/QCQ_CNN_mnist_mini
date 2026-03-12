# step6_evaluate.py — Đánh giá và trực quan hóa kết quả
# Chạy độc lập: python step6_evaluate.py

import matplotlib.pyplot as plt
import numpy as np

import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_baseline_curves(histories: dict, save_path="baseline_curves.png"):
    """
    Vẽ đường cong loss và accuracy của 4 mô hình baseline.

    Tham số:
        histories : dict — {'classical_mlp': History, 'quanvo_mlp': History, ...}
        save_path : str — đường dẫn lưu ảnh PNG

    Output:
        Lưu ảnh 2×4 subplots (loss hàng trên, accuracy hàng dưới) vào save_path
    """
    baseline_data = [
        (histories.get('classical_mlp'),  "Classical MLP",  0),
        (histories.get('quanvo_mlp'),     "Quanvo + MLP",   1),
        (histories.get('classical_cnn'),  "Classical CNN",  2),
        (histories.get('qccnn'),          "QCCNN",          3),
    ]
    # Lọc bỏ những model không có trong dict
    baseline_data = [(h, n, i) for h, n, i in baseline_data if h is not None]

    n_models = len(baseline_data)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 9))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle('So sánh Quá trình Huấn luyện — Các Mô hình Baseline',
                 fontsize=14, fontweight='bold')

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for hist, name, col_idx in baseline_data:
        epochs_range = range(1, len(hist.history['loss']) + 1)
        c = colors[col_idx % len(colors)]

        ax_loss = axes[0, col_idx] if n_models > 1 else axes[0, 0]
        ax_acc  = axes[1, col_idx] if n_models > 1 else axes[1, 0]

        # Hàng 1: Loss
        ax_loss.plot(epochs_range, hist.history['loss'],     color=c, lw=2,       label='Train Loss')
        ax_loss.plot(epochs_range, hist.history['val_loss'], color=c, lw=2, ls='--', label='Val Loss')
        ax_loss.set_title(f'{name}\nLoss', fontsize=11)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(fontsize=8)
        ax_loss.grid(alpha=0.3)

        # Hàng 2: Accuracy
        ax_acc.plot(epochs_range, hist.history['accuracy'],     color=c, lw=2,       label='Train Acc')
        ax_acc.plot(epochs_range, hist.history['val_accuracy'], color=c, lw=2, ls='--', label='Val Acc')
        best_val = max(hist.history['val_accuracy'])
        ax_acc.axhline(best_val, color='gray', ls=':', lw=1,
                       label=f"Best: {best_val*100:.1f}%")
        ax_acc.set_title(f'{name}\nAccuracy', fontsize=11)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_ylim([0, 1.05])
        ax_acc.legend(fontsize=8)
        ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Đã lưu: {save_path}")


def plot_qcq_curves(loss_list, acc_list, save_path="qcq_curves.png"):
    """
    Vẽ đường cong loss và accuracy của QCQ-CNN.

    Tham số:
        loss_list : list[float] — loss mỗi epoch
        acc_list  : list[float] — accuracy % mỗi epoch
        save_path : str — đường dẫn lưu ảnh PNG
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('QCQ-CNN — Quá trình Huấn luyện (PyTorch + Qiskit)',
                 fontsize=13, fontweight='bold')

    epochs_range = range(1, len(loss_list) + 1)

    # Loss
    axes[0].plot(epochs_range, loss_list, color='#9b59b6', lw=2.5, label='Training Loss')
    axes[0].set_title('QCQ-CNN Training Loss', fontsize=12)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Cross-Entropy Loss', fontsize=11)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Accuracy
    acc_normalized = [a / 100 for a in acc_list]
    axes[1].plot(epochs_range, acc_normalized, color='#9b59b6', lw=2.5, label='Training Accuracy')
    axes[1].axhline(max(acc_list) / 100, color='gray', ls=':', lw=1.5,
                    label=f"Peak: {max(acc_list):.1f}%")
    axes[1].set_title('QCQ-CNN Training Accuracy', fontsize=12)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Đã lưu: {save_path}")
    print(f"📊 Loss cuối: {loss_list[-1]:.4f} | Accuracy cuối: {acc_list[-1]:.1f}%")


def plot_confusion_matrix(
    model,
    test_loader,
    class_names=None,
    model_name="QCQ-CNN",
    device='cpu',
    save_path="confusion_matrix.png"
):
    """
    Tính và vẽ confusion matrix (dạng phần trăm) cho mô hình PyTorch.

    Tham số:
        model       : QCQ_CNN_Model — mô hình đã train
        test_loader : DataLoader — dữ liệu test
        class_names : list[str] — tên hiển thị từng lớp
        model_name  : str — tên hiển thị trên tiêu đề
        device      : str/device — thiết bị PyTorch
        save_path   : str — đường dẫn lưu ảnh PNG
    """
    if class_names is None:
        class_names = ["Số 3", "Số 5"]

    model.eval()   # Tắt Dropout khi đánh giá
    all_preds  = []
    all_labels = []

    with torch.no_grad():   # Không tính gradient → tiết kiệm bộ nhớ
        for data, target in test_loader:
            output = model(data.to(device))
            preds  = output.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(target.numpy())

    # Tính confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Chuẩn hóa về phần trăm theo từng hàng (theo lớp thực tế)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    overall_acc = np.trace(cm) / np.sum(cm) * 100
    print(f"\n📊 Kết quả đánh giá {model_name} trên test set:")
    print(f"   Ma trận nhầm lẫn (số mẫu):\n{cm}")
    print(f"   Accuracy tổng thể: {overall_acc:.1f}%")

    # Vẽ heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor='white',
        vmin=0, vmax=100
    )
    ax.set_ylabel("Nhãn Thực tế", fontsize=12)
    ax.set_xlabel("Nhãn Dự đoán", fontsize=12)
    ax.set_title(f"Ma trận Nhầm lẫn — {model_name}\n(Đơn vị: %)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"💾 Đã lưu: {save_path}")


if __name__ == "__main__":
    # Test với dữ liệu giả
    import numpy as np

    n_epochs = 10

    class FakeHistory:
        def __init__(self):
            self.history = {
                'loss':         np.linspace(0.7, 0.1, n_epochs).tolist(),
                'val_loss':     np.linspace(0.8, 0.2, n_epochs).tolist(),
                'accuracy':     np.linspace(0.5, 0.95, n_epochs).tolist(),
                'val_accuracy': np.linspace(0.5, 0.9, n_epochs).tolist(),
            }

    fake_histories = {
        'classical_mlp': FakeHistory(),
        'quanvo_mlp'   : FakeHistory(),
        'classical_cnn': FakeHistory(),
        'qccnn'        : FakeHistory(),
    }

    plot_baseline_curves(fake_histories, save_path="test_baseline_curves.png")
    plot_qcq_curves(
        np.linspace(0.7, 0.1, 20).tolist(),
        np.linspace(50.0, 95.0, 20).tolist(),
        save_path="test_qcq_curves.png"
    )
    print("✅ Biểu đồ đã lưu thành công")
