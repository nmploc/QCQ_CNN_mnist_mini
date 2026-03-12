# step1_setup.py — Kiểm tra môi trường và import thư viện
# Chạy độc lập: python step1_setup.py

import os

# ============================================================
# Ép TensorFlow/Keras chạy hoàn toàn trên CPU
# Lý do: TF và PyTorch KHÔNG chia sẻ CUDA context tốt trên Colab
# → TF chiếm GPU trước sẽ gây lỗi cho PyTorch và ngược lại
# → Keras chỉ dùng cho 4 baseline nhỏ (100 mẫu), CPU là đủ nhanh
# → PyTorch độc quyền GPU cho CNN + QCQ-CNN training
# ============================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Ẩn GPU khỏi TensorFlow hoàn toàn


def check_environment():
    """
    Kiểm tra và in trạng thái môi trường: phiên bản thư viện, device, backend.
    
    Thứ tự import quan trọng:
    os.environ → tensorflow → keras → torch → pennylane → qiskit
    """
    print("=" * 55)
    print("  KIỂM TRA MÔI TRƯỜNG")
    print("=" * 55)

    # --- TensorFlow / Keras ---
    try:
        import tensorflow as tf
        from tensorflow import keras
        tf_devices = [d.name for d in tf.config.list_logical_devices()]
        print(f"✅ TensorFlow {tf.__version__}  — devices: {tf_devices}")
        print(f"   (CPU only — GPU dành riêng cho PyTorch)")
    except ImportError:
        raise ImportError("Chạy: pip install tensorflow")

    # ============================================================
    # SAU KHI import TF xong, bật lại GPU cho PyTorch
    # PyTorch dùng CUDA trực tiếp qua driver, không bị ảnh hưởng
    # bởi CUDA_VISIBLE_DEVICES đã được đọc bởi TF ở trên
    # ============================================================
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # --- PyTorch ---
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ PyTorch {torch.__version__}  — device: {device}")
        if torch.cuda.is_available():
            print(f"   GPU  : {torch.cuda.get_device_name(0)}")
            print(f"   VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        raise ImportError("Chạy: pip install torch")

    # --- PennyLane ---
    try:
        import pennylane as qml
        # Kiểm tra backend lightning.qubit
        try:
            _dev = qml.device("lightning.qubit", wires=2)
            backend_name = "lightning.qubit (C++ — nhanh)"
            del _dev
        except Exception:
            backend_name = "default.qubit (Python — fallback)"
        print(f"✅ PennyLane {qml.__version__}  — backend: {backend_name}")
    except ImportError:
        raise ImportError("Chạy: pip install pennylane pennylane-lightning")

    # --- Qiskit ---
    try:
        import qiskit
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        print(f"✅ Qiskit {qiskit.__version__}")
    except ImportError:
        raise ImportError("Chạy: pip install qiskit>=1.0 qiskit-machine-learning>=0.7")

    # --- Thư viện phụ trợ ---
    try:
        import numpy as np
        import matplotlib
        from sklearn import __version__ as sk_ver
        from tqdm import __version__ as tqdm_ver
        import seaborn
        print(f"✅ NumPy {np.__version__}  |  Matplotlib {matplotlib.__version__}")
        print(f"✅ scikit-learn {sk_ver}  |  tqdm {tqdm_ver}  |  seaborn {seaborn.__version__}")
    except ImportError as e:
        raise ImportError(f"Thiếu thư viện phụ trợ: {e}")

    print("\n✅ Tất cả thư viện đã sẵn sàng!")
    print("=" * 55)


if __name__ == "__main__":
    check_environment()
    # Expected output:
    # ✅ TensorFlow 2.x.x  — devices: ['/device:CPU:0']
    # ✅ PyTorch 2.x.x  — device: cuda / cpu
    # ✅ PennyLane 0.x.x  — backend: lightning.qubit / default.qubit
    # ✅ Qiskit 1.x.x
    # ✅ NumPy ...  |  Matplotlib ...
    # ✅ Tất cả thư viện đã sẵn sàng!
