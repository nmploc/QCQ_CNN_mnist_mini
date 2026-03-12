# step3b_feature_extract.py — Trích xuất và lưu đặc trưng lượng tử
# Chạy độc lập: python step3b_feature_extract.py

import os
import time
import numpy as np

from config import (
    SIZE_FILTER, LAYERS_FILTER, STRIDE_FILTER,
    FEATURE_SAVE_DIR, USE_GDRIVE,
    N_TRAIN_PER_CLASS, N_TEST_PER_CLASS
)
from step3a_quanvo_layer import Quanvolutional_Layer


def extract_and_save_features(
    train_images,
    test_images,
    save_dir=None,
    use_gdrive=None,
    force_recompute=False
) -> tuple:
    """
    Trích xuất đặc trưng lượng tử từ tập train và test, lưu ra file .npy.

    Tham số:
        train_images   : numpy array shape (N_train, 28, 28, 1), float32 [0,1]
        test_images    : numpy array shape (N_test, 28, 28, 1),  float32 [0,1]
        save_dir       : str — thư mục lưu file .npy
        use_gdrive     : bool — True nếu mount Google Drive (chỉ dùng trên Colab)
        force_recompute: bool — True để tính lại dù file đã tồn tại

    Trả về: tuple (op_train, op_test)
        - op_train: shape (N_train, 14, 14, 4), dtype float32
        - op_test : shape (N_test,  14, 14, 4), dtype float32
    """
    if save_dir is None:
        save_dir = FEATURE_SAVE_DIR
    if use_gdrive is None:
        use_gdrive = USE_GDRIVE

    # --- Xử lý Google Drive (chỉ có tác dụng trên Colab) ---
    if use_gdrive:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            save_dir = '/content/drive/MyDrive/QCQ_CNN_Features'
            print(f"📁 Lưu vào Google Drive: {save_dir}")
        except ImportError:
            # Không phải Colab → bỏ qua, dùng save_dir mặc định
            print("⚠️  USE_GDRIVE=True nhưng không phải Colab — bỏ qua, lưu local")

    os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(save_dir, "MNIST_op_train.npy")
    test_path  = os.path.join(save_dir, "MNIST_op_test.npy")

    # --- Kiểm tra file đã tồn tại ---
    if os.path.exists(train_path) and os.path.exists(test_path) and not force_recompute:
        print("📂 Tìm thấy file đặc trưng đã lưu. Đang tải...")
        op_train = np.load(train_path).astype(np.float32)
        op_test  = np.load(test_path).astype(np.float32)
        print(f"✅ Đã tải xong!")
        print(f"   op_train: {op_train.shape}")
        print(f"   op_test : {op_test.shape}")
        return op_train, op_test

    # --- Tính mới ---
    print("🚀 Bắt đầu trích xuất đặc trưng lượng tử...")
    print(f"   Cấu hình: size_filter={SIZE_FILTER}, layers_filter={LAYERS_FILTER}, stride={STRIDE_FILTER}")
    n_circuits = (14 * 14) * (len(train_images) + len(test_images))
    print(f"   Ước tính: ~{n_circuits:,} lần chạy mạch")
    print(f"   (Với lightning.qubit nhanh hơn ~5-10x so với default.qubit)\n")

    # Khởi tạo layer: bộ lọc 2×2, LAYERS_FILTER layer, stride 2, không padding
    Quan2D = Quanvolutional_Layer(
        size_filter=SIZE_FILTER,
        layers_filter=LAYERS_FILTER,
        stride_filter=STRIDE_FILTER,
        padding=0
    )

    print("\n[TRAIN SET]")
    t_start = time.time()
    op_train_list = Quan2D.call(train_images)
    op_train = np.asarray(op_train_list, dtype=np.float32)
    t_train = time.time() - t_start

    print("\n[TEST SET]")
    op_test_list = Quan2D.call(test_images)
    op_test = np.asarray(op_test_list, dtype=np.float32)

    # Lưu kết quả
    np.save(train_path, op_train)
    np.save(test_path,  op_test)
    print(f"\n💾 Đã lưu đặc trưng:")
    print(f"   Train → {train_path}")
    print(f"   Test  → {test_path}")
    print(f"⏱️  Thời gian xử lý train set: {t_train:.1f} giây")
    print(f"📐 op_train: {op_train.shape}  |  op_test: {op_test.shape}")
    print(f"   Giá trị min/max: [{op_train.min():.3f}, {op_train.max():.3f}]")

    return op_train, op_test


if __name__ == "__main__":
    from step2_data import load_mnist_binary

    data = load_mnist_binary()
    train_images, train_labels, test_images, test_labels = data[:4]

    op_train, op_test = extract_and_save_features(
        train_images, test_images, force_recompute=False
    )

    print(f"\nop_train shape: {op_train.shape}")   # (100, 14, 14, 4) khi LAYERS_FILTER=1
    print(f"op_test  shape: {op_test.shape}")      # (30,  14, 14, 4)
    print(f"Value range: [{op_train.min():.3f}, {op_train.max():.3f}]")

    expected_channels = LAYERS_FILTER * SIZE_FILTER * SIZE_FILTER  # 1*2*2=4
    assert op_train.shape == (N_TRAIN_PER_CLASS * 2, 14, 14, expected_channels), \
        f"❌ Shape sai! Nhận {op_train.shape}"
    assert op_test.shape == (N_TEST_PER_CLASS * 2, 14, 14, expected_channels), \
        f"❌ Shape test sai! Nhận {op_test.shape}"
    print("✅ Feature extraction thành công")
