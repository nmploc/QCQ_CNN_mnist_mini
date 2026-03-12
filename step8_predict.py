# step8_predict.py — Dự đoán trên ảnh ngẫu nhiên từ MNIST
# Chạy độc lập: python step8_predict.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # TF chỉ dùng CPU

import numpy as np
import matplotlib.pyplot as plt
import torch

from config import TARGET_LABELS, MODEL_SAVE_DIR
from step3a_quanvo_layer import Quanvolutional_Layer


def predict_random_samples_balanced(
    model,
    all_train_images,
    all_train_labels,
    all_test_images,
    all_test_labels,
    class_names=None,
    n_per_class=5,
    device='cpu',
    seed=None,
    save_path="random_predictions.png"
):
    """
    Lấy ngẫu nhiên n_per_class ảnh mỗi lớp từ toàn bộ MNIST gốc,
    trích xuất đặc trưng lượng tử và chạy inference qua QCQ-CNN.

    Tham số:
        model             : QCQ_CNN_Model — đã train
        all_train_images  : numpy (N_train_full, 28, 28) — MNIST gốc train (để pool)
        all_train_labels  : numpy (N_train_full,)
        all_test_images   : numpy (N_test_full, 28, 28)
        all_test_labels   : numpy (N_test_full,)
        class_names       : list[str] — nhãn hiển thị
        n_per_class       : int — số mẫu mỗi lớp (tổng = 2 * n_per_class)
        device            : str — thiết bị PyTorch
        seed              : int hoặc None — seed ngẫu nhiên
        save_path         : str — đường dẫn lưu ảnh PNG

    Output:
        Lưu ảnh lưới 2×n_per_class vào save_path, in bảng kết quả chi tiết
    """
    if class_names is None:
        class_names = ["Số 3", "Số 5"]
    if seed is not None:
        np.random.seed(seed)

    n_total = n_per_class * 2

    # ----------------------------------------------------------
    # 1. Ghép pool toàn bộ MNIST train + test
    # ----------------------------------------------------------
    all_imgs = np.concatenate([all_train_images, all_test_images], axis=0)
    all_lbls = np.concatenate([all_train_labels, all_test_labels], axis=0)

    # Chuẩn hóa pixel về [0,1] — bắt buộc cho Angle Encoding
    all_imgs = all_imgs.astype(np.float32) / 255.0

    # ----------------------------------------------------------
    # 2. Lọc chỉ lấy TARGET_LABELS, ánh xạ {3,5} → {0,1}
    # ----------------------------------------------------------
    mask      = np.isin(all_lbls, TARGET_LABELS)
    pool_imgs = all_imgs[mask]
    pool_lbls = (all_lbls[mask] == TARGET_LABELS[1]).astype(np.int32)

    # ----------------------------------------------------------
    # 3. Chọn ngẫu nhiên đúng n_per_class mẫu mỗi lớp
    # ----------------------------------------------------------
    selected_imgs = []
    selected_lbls = []

    for class_idx in [0, 1]:
        candidate_indices = np.where(pool_lbls == class_idx)[0]
        chosen = np.random.choice(candidate_indices, size=n_per_class, replace=False)
        selected_imgs.append(pool_imgs[chosen])
        selected_lbls.append(pool_lbls[chosen])

    # Xáo trộn để 2 lớp không đứng riêng thành 2 khối
    selected_imgs = np.concatenate(selected_imgs, axis=0)
    selected_lbls = np.concatenate(selected_lbls, axis=0)
    shuffle_order = np.random.permutation(n_total)
    selected_imgs = selected_imgs[shuffle_order]
    selected_lbls = selected_lbls[shuffle_order]

    # ----------------------------------------------------------
    # 4. Trích xuất đặc trưng lượng tử cho n_total ảnh mới
    # ----------------------------------------------------------
    print(f"⚛️  Đang trích xuất đặc trưng lượng tử cho {n_total} ảnh mới...")
    imgs_with_channel = selected_imgs[:, :, :, np.newaxis]  # (n_total, 28, 28, 1)

    Quan2D_infer = Quanvolutional_Layer(
        size_filter=2, layers_filter=1, stride_filter=2, padding=0
    )
    features_list = Quan2D_infer.call(imgs_with_channel)
    features_np   = np.asarray(features_list, dtype=np.float32)  # (n_total, 14, 14, 4)

    # ----------------------------------------------------------
    # 5. Inference qua QCQ-CNN
    # ----------------------------------------------------------
    model.eval()
    results_detail = []

    for i in range(n_total):
        # (14, 14, 4) → (1, 4, 14, 14): channels-first cho PyTorch
        feat_tensor = torch.tensor(
            features_np[i].transpose(2, 0, 1)[np.newaxis, :],
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            output = model(feat_tensor)
            probs  = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred   = output.argmax(dim=1).item()

        label      = selected_lbls[i]
        is_correct = (pred == label)
        results_detail.append({
            'image'    : selected_imgs[i],
            'true'     : int(label),
            'pred'     : int(pred),
            'prob_cls0': float(probs[0]),
            'prob_cls1': float(probs[1]),
            'correct'  : is_correct,
        })

    # ----------------------------------------------------------
    # 6. Vẽ lưới 2 hàng × n_per_class cột
    # ----------------------------------------------------------
    fig, axes = plt.subplots(2, n_per_class, figsize=(16, 7))
    if n_per_class == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        f'Kiểm tra QCQ-CNN — {n_total} ảnh ngẫu nhiên từ MNIST\n'
        f'({n_per_class} mẫu {class_names[0]}  •  {n_per_class} mẫu {class_names[1]}'
        f'  |  Xanh = Đúng, Đỏ = Sai)',
        fontsize=13, fontweight='bold', y=1.01
    )

    for plot_idx, r in enumerate(results_detail):
        row = plot_idx // n_per_class
        col = plot_idx %  n_per_class
        ax  = axes[row, col]

        ax.imshow(r['image'], cmap='gray', interpolation='nearest')

        color  = '#27ae60' if r['correct'] else '#e74c3c'
        status = '✓' if r['correct'] else '✗'
        prob_shown = r['prob_cls1'] if r['pred'] == 1 else r['prob_cls0']

        ax.set_title(
            f"Thực tế : {class_names[r['true']]}\n"
            f"Dự đoán : {class_names[r['pred']]} {status}\n"
            f"Độ tin cậy : {prob_shown*100:.1f}%",
            color=color, fontsize=9, fontweight='bold', pad=6
        )
        ax.axis('off')

        # Viền màu quanh ảnh
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------------
    # 7. Bảng kết quả chi tiết
    # ----------------------------------------------------------
    n_correct = sum(r['correct'] for r in results_detail)
    print(f"\n{'='*62}")
    print(f"📊 KẾT QUẢ CHI TIẾT — {n_total} ảnh ({n_per_class} mỗi lớp)")
    print(f"{'='*62}")
    print(f"{'#':>3} | {'Thực tế':>8} | {'Dự đoán':>8} | "
          f"{'P(Số 3)':>8} | {'P(Số 5)':>8} | {'Kết quả':>8}")
    print(f"{'─'*62}")

    for i, r in enumerate(results_detail):
        kq = "✓ Đúng" if r['correct'] else "✗ Sai"
        print(f"{i+1:>3} | {class_names[r['true']]:>8} | "
              f"{class_names[r['pred']]:>8} | "
              f"{r['prob_cls0']*100:>6.1f}%  | "
              f"{r['prob_cls1']*100:>6.1f}%  | {kq}")

    print(f"{'─'*62}")
    n_c3 = sum(r['correct'] for r in results_detail if r['true'] == 0)
    n_c5 = sum(r['correct'] for r in results_detail if r['true'] == 1)
    print(f"{class_names[0]} : {n_c3}/{n_per_class} đúng  |  "
          f"{class_names[1]} : {n_c5}/{n_per_class} đúng  |  "
          f"Tổng : {n_correct}/{n_total} = {n_correct/n_total*100:.1f}%")
    print(f"{'='*62}")
    print(f"\n💾 Đã lưu: {save_path}")


if __name__ == "__main__":
    from step2_data import load_mnist_binary
    from step3b_feature_extract import extract_and_save_features
    from step5bc_train_qcq import prepare_dataloaders, train_qcq_cnn
    from step7_save_load import load_qcq_model

    weight_path = os.path.join(MODEL_SAVE_DIR, "qcq_cnn_weights.pt")

    data = load_mnist_binary()
    train_images, train_labels, test_images, test_labels, \
        all_train_images, all_train_labels, \
        all_test_images, all_test_labels = data

    op_train, op_test = extract_and_save_features(train_images, test_images)
    train_loader, _ = prepare_dataloaders(op_train, train_labels, op_test, test_labels)

    if os.path.exists(weight_path):
        model = load_qcq_model(weight_path)
        print("✅ Đã tải model từ file")
    else:
        model, _, _ = train_qcq_cnn(train_loader, epochs=5)
        print("⚡ Đã train nhanh 5 epoch để test")

    predict_random_samples_balanced(
        model,
        all_train_images, all_train_labels,
        all_test_images, all_test_labels,
        seed=42
    )
    print("✅ Dự đoán hoàn thành — xem random_predictions.png")
