# step7_save_load.py — Lưu và tải trọng số mô hình
# Chạy độc lập: python step7_save_load.py

import os
import json
import torch

from config import MODEL_SAVE_DIR
from step4b_model_qcq_cnn import QCQ_CNN_Model, create_qnn


def save_qcq_model(model, save_dir=None):
    """
    Lưu mô hình QCQ-CNN: state_dict và full model.

    Tham số:
        model    : QCQ_CNN_Model — mô hình đã train
        save_dir : str — thư mục lưu file

    Lưu:
        <save_dir>/qcq_cnn_weights.pt  — state_dict (khuyến nghị)
        <save_dir>/qcq_cnn_full.pt     — full model
    """
    if save_dir is None:
        save_dir = MODEL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Cách 1: Lưu state_dict (khuyến nghị)
    weights_path = os.path.join(save_dir, "qcq_cnn_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Đã lưu QCQ-CNN weights: {weights_path}")

    # Cách 2: Lưu toàn bộ mô hình
    full_path = os.path.join(save_dir, "qcq_cnn_full.pt")
    torch.save(model, full_path)
    print(f"✅ Đã lưu QCQ-CNN full model: {full_path}")


def load_qcq_model(path: str) -> QCQ_CNN_Model:
    """
    Tải state_dict vào model mới, trả về model ở eval mode.

    Tham số:
        path : str — đường dẫn đến file .pt chứa state_dict

    Trả về:
        QCQ_CNN_Model đã tải trọng số, ở eval mode
    """
    qnn   = create_qnn()
    model = QCQ_CNN_Model(qnn)

    # Cần warm-up để khởi tạo fc2 (lazy layer) trước khi load state_dict
    with torch.no_grad():
        dummy = torch.zeros(1, 4, 14, 14)
        _ = model(dummy)

    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    print(f"✅ Đã tải QCQ-CNN model từ: {path}")
    return model


def save_keras_models(models_dict: dict, save_dir=None):
    """
    Lưu các mô hình Keras với định dạng .keras.

    Tham số:
        models_dict : dict — {'classical_mlp': model, 'quanvo_mlp': model, ...}
        save_dir    : str — thư mục lưu file
    """
    if save_dir is None:
        save_dir = MODEL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    for name, model in models_dict.items():
        k_path = os.path.join(save_dir, f"{name}.keras")
        model.save(k_path)
        print(f"✅ Đã lưu Keras model: {k_path}")


def save_training_history(history_dict: dict, save_dir=None):
    """
    Lưu lịch sử huấn luyện QCQ-CNN dưới dạng JSON.

    Tham số:
        history_dict : dict — {'loss_qcq': [...], 'acc_qcq': [...], 'epochs': N}
        save_dir     : str — thư mục lưu file
    """
    if save_dir is None:
        save_dir = MODEL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    hist_path = os.path.join(save_dir, "training_history.json")
    with open(hist_path, 'w', encoding='utf-8') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✅ Đã lưu lịch sử huấn luyện: {hist_path}")


if __name__ == "__main__":
    import torch, os

    # Test save/load roundtrip
    model_orig = QCQ_CNN_Model()

    # Warm-up để khởi tạo fc2
    with torch.no_grad():
        _ = model_orig(torch.zeros(1, 4, 14, 14))

    save_qcq_model(model_orig)

    weights_path = os.path.join(MODEL_SAVE_DIR, "qcq_cnn_weights.pt")
    model_loaded = load_qcq_model(weights_path)

    # Kiểm tra weights giống nhau
    orig_state  = model_orig.state_dict()
    load_state  = model_loaded.state_dict()
    for k1, k2 in zip(orig_state.keys(), load_state.keys()):
        v1, v2 = orig_state[k1], load_state[k2]
        assert torch.allclose(v1, v2), f"❌ Weights khác nhau tại {k1}"

    print("✅ Save/load roundtrip thành công — weights khớp hoàn toàn")
