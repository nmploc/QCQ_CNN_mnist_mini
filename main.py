"""
main.py — Pipeline QCQ-CNN end-to-end
Pipeline điều phối toàn bộ các bước từ đầu đến cuối.

Cách chạy:
    python main.py                         # Chạy đầy đủ 100 epoch
    python main.py --epochs 3              # Test nhanh 3 epoch
    python main.py --skip-baseline         # Chỉ train QCQ-CNN
    python main.py --skip-qcq             # Chỉ train baseline
    python main.py --epochs 3 --skip-baseline  # QCQ-CNN, 3 epoch
"""
import argparse
from config import N_EPOCHS


def main(epochs=N_EPOCHS, skip_baseline=False, skip_qcq=False):
    """
    Chạy pipeline QCQ-CNN đầy đủ.

    Tham số:
        epochs         : int — số epoch huấn luyện
        skip_baseline  : bool — bỏ qua huấn luyện 4 baseline Keras
        skip_qcq       : bool — bỏ qua huấn luyện QCQ-CNN (PyTorch)
    """
    print("=" * 60)
    print("  QCQ-CNN MNIST Pipeline")
    print("=" * 60)
    print(f"  Epochs       : {epochs}")
    print(f"  Skip Baseline: {skip_baseline}")
    print(f"  Skip QCQ     : {skip_qcq}")
    print("=" * 60)

    # Bước 1: Kiểm tra môi trường
    from step1_setup import check_environment
    check_environment()

    # Bước 2: Tải dữ liệu
    print("\n[Bước 2] Tải và tiền xử lý dữ liệu MNIST...")
    from step2_data import load_mnist_binary
    data = load_mnist_binary()
    train_images, train_labels, test_images, test_labels, *rest_data = data
    all_train_images, all_train_labels, all_test_images, all_test_labels = rest_data
    print(f"   Train: {train_images.shape} | Test: {test_images.shape}")

    # Bước 3: Trích xuất đặc trưng lượng tử
    print("\n[Bước 3] Trích xuất đặc trưng lượng tử (Quanvolutional)...")
    from step3b_feature_extract import extract_and_save_features
    op_train, op_test = extract_and_save_features(train_images, test_images)

    histories = None

    # Bước 4+5: Baseline models (Keras)
    if not skip_baseline:
        print("\n[Bước 4+5] Huấn luyện các mô hình baseline (Keras)...")
        from step5a_train_baseline import train_all_baselines
        histories = train_all_baselines(
            train_images, train_labels,
            test_images,  test_labels,
            op_train, op_test,
            epochs=epochs
        )

    model_qcq = None
    test_loader = None

    # Bước 5bc: QCQ-CNN (PyTorch + Qiskit)
    if not skip_qcq:
        print("\n[Bước 5bc] Huấn luyện QCQ-CNN (PyTorch + Qiskit)...")
        from step5bc_train_qcq import prepare_dataloaders, train_qcq_cnn
        train_loader, test_loader = prepare_dataloaders(
            op_train, train_labels, op_test, test_labels
        )
        model_qcq, loss_qcq, acc_qcq = train_qcq_cnn(
            train_loader, epochs=epochs
        )

    # Bước 6: Đánh giá và trực quan hóa
    print("\n[Bước 6] Vẽ biểu đồ đánh giá...")
    from step6_evaluate import plot_baseline_curves, plot_qcq_curves, plot_confusion_matrix
    if not skip_baseline and histories:
        plot_baseline_curves(histories)
    if not skip_qcq and model_qcq:
        plot_qcq_curves(loss_qcq, acc_qcq)
        plot_confusion_matrix(model_qcq, test_loader)

    # Bước 7: Lưu model
    if not skip_qcq and model_qcq:
        print("\n[Bước 7] Lưu trọng số mô hình...")
        from step7_save_load import save_qcq_model, save_keras_models, save_training_history
        save_qcq_model(model_qcq)
        save_training_history({'loss_qcq': loss_qcq, 'acc_qcq': acc_qcq, 'epochs': epochs})
        if not skip_baseline and histories:
            # Rebuild Keras models với weights đã train (nếu cần lưu)
            pass  # histories chứa History objects, không lưu model weights ở đây

    # Bước 8: Dự đoán ảnh ngẫu nhiên
    if not skip_qcq and model_qcq:
        print("\n[Bước 8] Dự đoán trên ảnh ngẫu nhiên...")
        from step8_predict import predict_random_samples_balanced
        predict_random_samples_balanced(
            model_qcq,
            all_train_images, all_train_labels,
            all_test_images,  all_test_labels,
            seed=42
        )

    print("\n" + "=" * 60)
    print("  ✅ Pipeline hoàn thành!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QCQ-CNN MNIST Pipeline"
    )
    parser.add_argument(
        "--epochs", type=int, default=N_EPOCHS,
        help=f"Số epoch huấn luyện (mặc định: {N_EPOCHS})"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Bỏ qua huấn luyện 4 mô hình baseline Keras"
    )
    parser.add_argument(
        "--skip-qcq", action="store_true",
        help="Bỏ qua huấn luyện QCQ-CNN"
    )
    args = parser.parse_args()
    main(args.epochs, args.skip_baseline, args.skip_qcq)
