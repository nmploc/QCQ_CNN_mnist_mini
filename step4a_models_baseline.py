# step4a_models_baseline.py — Các mô hình Keras baseline
# Chạy độc lập: python step4a_models_baseline.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Keras chạy CPU

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Resizing


def build_classical_mlp() -> keras.Model:
    """
    MLP thuần cổ điển nhận ảnh gốc 28×28×1 làm đầu vào.
    Baseline không dùng đặc trưng lượng tử.

    Input : (28, 28, 1)
    Output: (2,) — xác suất 2 lớp (softmax)
    """
    # Dùng keras.Input() thay cho input_shape= trong layer đầu tiên
    # → Keras 3.x không build graph ngay, tránh crash CUDA khi chưa có GPU context
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2,  activation='softmax'),
    ], name="Classical_MLP")
    return model


def build_mlp_on_quantum_features() -> keras.Model:
    """
    MLP đơn giản nhận đặc trưng lượng tử (14×14×4) làm đầu vào.
    Đây là baseline Quanvo + MLP.

    Input : (14, 14, 4)
    Output: (2,) — xác suất 2 lớp (softmax)
    """
    model = keras.Sequential([
        keras.Input(shape=(14, 14, 4)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2,  activation='softmax'),
    ], name="MLP_on_QuantumFeatures")
    return model


def build_classical_cnn() -> keras.Model:
    """
    CNN cổ điển 2 lớp trên ảnh gốc 28×28×1.
    Baseline CNN không có tầng lượng tử.
    Resize về 14×14 để cùng scale với quantum features.

    Input : (28, 28, 1)
    Output: (2,) — xác suất 2 lớp (softmax)
    """
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        Resizing(14, 14),                      # Resize về 14×14 để so sánh công bằng
        layers.Conv2D(4, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(2, activation='softmax'),
    ], name="Classical_CNN")
    return model


def build_qccnn() -> keras.Model:
    """
    CNN Keras xử lý đặc trưng lượng tử (14×14×4).
    = Quanvolutional + Classical CNN (không có QNN ở cuối).

    Input : (14, 14, 4)
    Output: (2,) — xác suất 2 lớp (softmax)
    """
    model = keras.Sequential([
        keras.Input(shape=(14, 14, 4)),
        layers.Conv2D(4, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(2, activation='softmax'),
    ], name="QCCNN")
    return model


if __name__ == "__main__":
    # Kiểm tra build và đếm tham số — chưa compile
    models = {
        "Classical MLP"  : build_classical_mlp(),
        "Quanvo MLP"     : build_mlp_on_quantum_features(),
        "Classical CNN"  : build_classical_cnn(),
        "QCCNN"          : build_qccnn(),
    }
    print("=" * 50)
    print("📋 KIẾN TRÚC CÁC MÔ HÌNH BASELINE (Keras, chưa compile):")
    print("=" * 50)
    for name, model in models.items():
        n_params = model.count_params()
        print(f"  {name:20s}: {n_params:,} params")
    print("✅ Tất cả 4 baseline models build thành công (chưa compile)")
