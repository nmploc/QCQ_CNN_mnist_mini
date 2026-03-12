# step5bc_train_qcq.py — DataLoader + Vòng lặp huấn luyện QCQ-CNN
# Chạy độc lập: python step5bc_train_qcq.py

import time
import random
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, N_EPOCHS, RANDOM_SEED
from step4b_model_qcq_cnn import QCQ_CNN_Model, create_qnn


def prepare_dataloaders(
    op_train,
    train_labels,
    op_test,
    test_labels,
    batch_size=None
) -> tuple:
    """
    Tạo PyTorch DataLoader từ đặc trưng lượng tử.

    Chuyển đổi (N, H, W, C) → (N, C, H, W): PyTorch dùng channels-first.

    Tham số:
        op_train     : numpy array (N_train, 14, 14, C)
        train_labels : numpy array (N_train,)
        op_test      : numpy array (N_test, 14, 14, C)
        test_labels  : numpy array (N_test,)
        batch_size   : int — kích thước batch

    Trả về: tuple (train_loader, test_loader)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    # (N, H, W, C) → (N, C, H, W): PyTorch dùng channels-first
    op_train_pt = np.transpose(op_train, (0, 3, 1, 2))   # (N_train, 4, 14, 14)
    op_test_pt  = np.transpose(op_test,  (0, 3, 1, 2))   # (N_test,  4, 14, 14)

    X_train_tensor = torch.tensor(op_train_pt,  dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
    X_test_tensor  = torch.tensor(op_test_pt,   dtype=torch.float32)
    y_test_tensor  = torch.tensor(test_labels,  dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)

    # pin_memory=True: Dữ liệu được "ghim" vào RAM không swap → transfer CPU→GPU nhanh hơn
    # Chỉ có tác dụng khi có CUDA, tự tắt khi dùng CPU
    # num_workers=0: Giữ nguyên 0 để tránh lỗi multiprocessing với CUDA
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin,
        num_workers=0,
    )

    print(f"📦 Train Tensor: {X_train_tensor.shape}  |  Test Tensor: {X_test_tensor.shape}")
    print(f"   Batch size: {batch_size}  |  Train batches: {len(train_loader)}  |  pin_memory: {use_pin}")

    return train_loader, test_loader


def train_qcq_cnn(
    train_loader,
    epochs=None,
    seed=None,
    device=None
) -> tuple:
    """
    Huấn luyện mô hình QCQ-CNN end-to-end.

    Gradient được backprop qua: fc3 → QNN (CPU) → fc2 → CNN (GPU/CPU).
    TorchConnector cho phép tính gradient của QNN bằng parameter-shift rule.
    Dùng AMP (torch.cuda.amp) nếu CUDA available.

    Tham số:
        train_loader : DataLoader — vòng lặp dữ liệu train
        epochs       : int — số epoch
        seed         : int — seed để tái lập kết quả
        device       : str hoặc torch.device (mặc định: tự detect GPU/CPU)

    Trả về: tuple (model, loss_list, acc_list)
        - model    : QCQ_CNN_Model đã train
        - loss_list: list float — loss mỗi epoch
        - acc_list : list float — accuracy % mỗi epoch
    """
    if epochs is None:
        epochs = N_EPOCHS
    if seed is None:
        seed = RANDOM_SEED

    # Tự động detect device nếu không truyền vào
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # --- Đặt seed cho tái lập kết quả ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Khởi tạo mô hình ---
    qnn   = create_qnn()
    model = QCQ_CNN_Model(qnn).to(device)

    # Warm-up forward pass để khởi tạo lazy layer fc2
    # (fc2 cần thấy kích thước flatten thực tế trước khi optimizer nhận params)
    with torch.no_grad():
        _dummy = torch.zeros(1, 4, 14, 14).to(device)
        _ = model(_dummy)
        del _dummy

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Tổng tham số có thể học: {total_params:,}")
    print(f"   Device CNN/fc: {device}")
    print(f"   Device QNN   : CPU (Qiskit không hỗ trợ CUDA)")

    # Adam optimizer — hoạt động tốt với cả tham số cổ điển và lượng tử
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    # --- Mixed Precision (AMP) ---
    # autocast + GradScaler: Tính forward/backward bằng float16 trên GPU → ~1.5-2x nhanh hơn
    # Tự động tắt khi không có CUDA (enabled=False → hoạt động như code thường)
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print(f"   ⚡ Mixed Precision (AMP): Bật")
    else:
        print(f"   ℹ️  Mixed Precision (AMP): Tắt (CPU mode)")

    loss_list, acc_list = [], []
    start_time = time.time()

    model.train()  # Bật Dropout

    for epoch in range(epochs):
        correct    = 0
        total_loss = []

        for data, target in train_loader:
            # Chuyển data và target lên GPU (nếu có)
            # QCQ_CNN_Model.forward() sẽ tự xử lý việc đưa x xuống CPU cho QNN
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad(set_to_none=True)  # Hiệu quả hơn zero_grad()

            # autocast: Tự động chọn float16 cho các op phù hợp trên GPU
            # Phần QNN chạy trên CPU không bị ảnh hưởng bởi autocast
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(data)
                loss   = loss_func(output, target.long())

            # scaler.scale(loss).backward() tương đương loss.backward()
            # nhưng scale loss để tránh underflow với float16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss.append(loss.item())
            pred     = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

        epoch_loss = sum(total_loss) / len(total_loss)
        epoch_acc  = 100.0 * correct / len(train_loader.dataset)
        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{epoch+1:3d}/{epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.1f}%")

    elapsed = time.time() - start_time
    print(f"\n✅ Tổng thời gian huấn luyện: {elapsed:.1f} giây")

    return model, loss_list, acc_list


if __name__ == "__main__":
    from step2_data import load_mnist_binary
    from step3b_feature_extract import extract_and_save_features

    data = load_mnist_binary()
    train_images, train_labels, test_images, test_labels = data[:4]
    op_train, op_test = extract_and_save_features(train_images, test_images)

    train_loader, test_loader = prepare_dataloaders(
        op_train, train_labels, op_test, test_labels
    )

    # Smoke test: 3 epoch
    print("\n🚀 Smoke test: 3 epoch")
    model, losses, accs = train_qcq_cnn(train_loader, epochs=3)
    print(f"Epoch 3 — loss: {losses[-1]:.4f}, acc: {accs[-1]:.1f}%")
    print("✅ QCQ-CNN training loop hoạt động")
