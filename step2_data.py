# step2_data.py — Pipeline tải và tiền xử lý dữ liệu MNIST
# Chạy độc lập: python step2_data.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # TF chỉ dùng CPU

import numpy as np

from config import (
    TARGET_LABELS, N_TRAIN_PER_CLASS, N_TEST_PER_CLASS, RANDOM_SEED
)


def load_mnist_binary(
    target_labels=None,
    n_train_per_class=None,
    n_test_per_class=None,
    seed=None
) -> tuple:
    """
    Tải MNIST, lọc hai nhãn target_labels, chuẩn hóa và chọn tập con cân bằng.

    Tham số:
        target_labels      : list[int] — hai nhãn cần phân loại (mặc định từ config)
        n_train_per_class  : int — số mẫu mỗi lớp cho train
        n_test_per_class   : int — số mẫu mỗi lớp cho test
        seed               : int — random seed

    Trả về: tuple 8 phần tử
        (train_images, train_labels, test_images, test_labels,
         all_train_images, all_train_labels,
         all_test_images,  all_test_labels)
        - train/test images: float32, shape (N, 28, 28, 1), giá trị [0,1]
        - train/test labels: int32, 0 hoặc 1 (remap từ target_labels[0]→0, target_labels[1]→1)
        - all_*: toàn bộ MNIST gốc (dùng cho step8 predict)
    """
    if target_labels is None:
        target_labels = TARGET_LABELS
    if n_train_per_class is None:
        n_train_per_class = N_TRAIN_PER_CLASS
    if n_test_per_class is None:
        n_test_per_class = N_TEST_PER_CLASS
    if seed is None:
        seed = RANDOM_SEED

    # Dùng tensorflow.keras để tải MNIST (giữ nguyên từ notebook)
    from tensorflow import keras

    # --- 2.1: Tải toàn bộ MNIST ---
    (all_train_images, all_train_labels), (all_test_images, all_test_labels) = \
        keras.datasets.mnist.load_data()

    # --- 2.2: Lọc chỉ lấy hai nhãn target ---
    train_mask = np.isin(all_train_labels, target_labels)
    test_mask  = np.isin(all_test_labels,  target_labels)

    train_images = all_train_images[train_mask]
    train_labels = all_train_labels[train_mask]
    test_images  = all_test_images[test_mask]
    test_labels  = all_test_labels[test_mask]

    # --- 2.3: Ánh xạ nhãn {target_labels[0], target_labels[1]} → {0, 1} ---
    # Nhãn đầu tiên → 0 (lớp âm), nhãn thứ hai → 1 (lớp dương)
    train_labels = (train_labels == target_labels[1]).astype(np.int32)
    test_labels  = (test_labels  == target_labels[1]).astype(np.int32)

    # --- 2.4: Chọn tập con cân bằng ---
    train_images, train_labels, test_images, test_labels = _select_balanced_subset(
        train_images, train_labels,
        test_images,  test_labels,
        per_class_train=n_train_per_class,
        per_class_test=n_test_per_class
    )

    # --- 2.5: Chuẩn hóa về [0,1] và thêm chiều kênh ---
    train_images = train_images.astype(np.float32) / 255.0
    test_images  = test_images.astype(np.float32)  / 255.0
    train_images = train_images[..., np.newaxis]   # (N, 28, 28) → (N, 28, 28, 1)
    test_images  = test_images[..., np.newaxis]

    return (
        train_images, train_labels,
        test_images,  test_labels,
        all_train_images, all_train_labels,
        all_test_images,  all_test_labels
    )


def _select_balanced_subset(tr_imgs, tr_lbls, te_imgs, te_lbls,
                             per_class_train, per_class_test):
    """
    Chọn đúng per_class_train mẫu mỗi lớp cho train
    và per_class_test mẫu mỗi lớp cho test.
    Đảm bảo dữ liệu cân bằng giữa 2 lớp.
    """
    sel_tr_imgs, sel_tr_lbls = [], []
    sel_te_imgs, sel_te_lbls = [], []

    for label_val in [0, 1]:
        # Tìm chỉ số của các mẫu thuộc lớp label_val
        tr_idx = np.where(tr_lbls == label_val)[0][:per_class_train]
        te_idx = np.where(te_lbls == label_val)[0][:per_class_test]

        sel_tr_imgs.append(tr_imgs[tr_idx])
        sel_tr_lbls.append(tr_lbls[tr_idx])
        sel_te_imgs.append(te_imgs[te_idx])
        sel_te_lbls.append(te_lbls[te_idx])

    return (
        np.concatenate(sel_tr_imgs, axis=0),
        np.concatenate(sel_tr_lbls, axis=0),
        np.concatenate(sel_te_imgs, axis=0),
        np.concatenate(sel_te_lbls, axis=0)
    )


if __name__ == "__main__":
    data = load_mnist_binary()
    train_images, train_labels, test_images, test_labels, *_ = data
    print(f"Train: {train_images.shape}, labels: {set(train_labels)}")
    print(f"Test : {test_images.shape},  labels: {set(test_labels)}")
    print(f"Phân bố Train - Lớp 0: {(train_labels==0).sum()}, Lớp 1: {(train_labels==1).sum()}")

    # Vẽ 5 ảnh mẫu để xác nhận dữ liệu đúng
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    fig.suptitle("5 mẫu train — kiểm tra dữ liệu")
    for i, ax in enumerate(axes):
        ax.imshow(train_images[i, :, :, 0], cmap='gray')
        ax.set_title(f"Label: {TARGET_LABELS[train_labels[i]]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("test_data_samples.png")
    print("✅ Đã lưu test_data_samples.png")
