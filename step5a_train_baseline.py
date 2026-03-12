# step5a_train_baseline.py — Huấn luyện các mô hình Keras baseline
# Chạy độc lập: python step5a_train_baseline.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Keras chạy CPU

from tensorflow import keras
from tensorflow.keras import optimizers

from config import N_EPOCHS
from step4a_models_baseline import (
    build_classical_mlp,
    build_mlp_on_quantum_features,
    build_classical_cnn,
    build_qccnn,
)


def train_all_baselines(
    train_images,
    train_labels,
    test_images,
    test_labels,
    op_train,
    op_test,
    epochs=None
) -> dict:
    """
    Compile và huấn luyện 4 mô hình Keras baseline.

    Tham số:
        train_images : numpy array (N, 28, 28, 1) — ảnh gốc normalized
        train_labels : numpy array (N,) — nhãn 0/1
        test_images  : numpy array (M, 28, 28, 1)
        test_labels  : numpy array (M,)
        op_train     : numpy array (N, 14, 14, C) — đặc trưng lượng tử train
        op_test      : numpy array (M, 14, 14, C) — đặc trưng lượng tử test
        epochs       : int — số epoch huấn luyện

    Trả về:
        dict với keys: 'classical_mlp', 'quanvo_mlp', 'classical_cnn', 'qccnn'
        Mỗi value là keras History object.
    """
    if epochs is None:
        epochs = N_EPOCHS

    compile_cfg = dict(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    histories = {}

    # --- Model 1: Classical MLP (trên ảnh gốc) ---
    print("\n[1/4] Classical MLP (trên ảnh gốc 28×28)")
    model_mlp = build_classical_mlp()
    model_mlp.compile(**compile_cfg)
    hist = model_mlp.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        batch_size=4, epochs=epochs, verbose=0
    )
    val_acc = max(hist.history['val_accuracy'])
    print(f"   ✅ Tốt nhất Val Accuracy: {val_acc*100:.1f}%")
    histories['classical_mlp'] = hist

    # --- Model 2: Quanvo MLP (trên đặc trưng lượng tử) ---
    print("\n[2/4] Quanvo + MLP (đặc trưng lượng tử 14×14×C)")
    model_qmlp = build_mlp_on_quantum_features()
    model_qmlp.compile(**compile_cfg)
    hist = model_qmlp.fit(
        op_train, train_labels,
        validation_data=(op_test, test_labels),
        batch_size=4, epochs=epochs, verbose=0
    )
    val_acc = max(hist.history['val_accuracy'])
    print(f"   ✅ Tốt nhất Val Accuracy: {val_acc*100:.1f}%")
    histories['quanvo_mlp'] = hist

    # --- Model 3: Classical CNN (trên ảnh gốc, resize 14×14 bên trong model) ---
    print("\n[3/4] Classical CNN (trên ảnh gốc, resize 14×14)")
    model_cnn = build_classical_cnn()
    model_cnn.compile(**compile_cfg)
    hist = model_cnn.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        batch_size=4, epochs=epochs, verbose=0
    )
    val_acc = max(hist.history['val_accuracy'])
    print(f"   ✅ Tốt nhất Val Accuracy: {val_acc*100:.1f}%")
    histories['classical_cnn'] = hist

    # --- Model 4: QCCNN (CNN Keras trên đặc trưng lượng tử) ---
    print("\n[4/4] QCCNN (CNN Keras + đặc trưng lượng tử)")
    model_qccnn = build_qccnn()
    model_qccnn.compile(**compile_cfg)
    hist = model_qccnn.fit(
        op_train, train_labels,
        validation_data=(op_test, test_labels),
        batch_size=4, epochs=epochs, verbose=0
    )
    val_acc = max(hist.history['val_accuracy'])
    print(f"   ✅ Tốt nhất Val Accuracy: {val_acc*100:.1f}%")
    histories['qccnn'] = hist

    print("\n✅ Huấn luyện tất cả baseline hoàn tất!")
    return histories


if __name__ == "__main__":
    from step2_data import load_mnist_binary
    from step3b_feature_extract import extract_and_save_features

    data = load_mnist_binary()
    train_images, train_labels, test_images, test_labels = data[:4]
    op_train, op_test = extract_and_save_features(train_images, test_images)

    histories = train_all_baselines(
        train_images, train_labels,
        test_images, test_labels,
        op_train, op_test,
        epochs=10  # Dùng 10 epoch khi test nhanh
    )

    print("\n📊 Kết quả:")
    for name, hist in histories.items():
        val_acc = max(hist.history['val_accuracy']) * 100
        print(f"  {name:20s}: best val_acc = {val_acc:.1f}%")
    print("✅ Tất cả baseline đã train xong")
