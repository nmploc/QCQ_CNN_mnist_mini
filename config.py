# config.py — Trung tâm cấu hình cho toàn bộ project QCQ-CNN
# Mọi file khác đều import từ đây, không được hardcode giá trị ở nơi khác.

# --- Dữ liệu ---
TARGET_LABELS       = [3, 5]      # Hai chữ số cần phân loại (số 3 và số 5)
N_TRAIN_PER_CLASS   = 50          # Số mẫu mỗi lớp cho tập train
N_TEST_PER_CLASS    = 15          # Số mẫu mỗi lớp cho tập test
N_EPOCHS            = 100         # Số epoch huấn luyện mặc định

# --- Quanvolutional Layer ---
SIZE_FILTER    = 2    # Kích thước bộ lọc vuông (2×2)
LAYERS_FILTER  = 1    # Số lớp RandomLayers (= số 'kênh' đầu ra thực tế từ notebook)
STRIDE_FILTER  = 2    # Bước trượt của bộ lọc
RANDOM_SEED    = 42   # Seed ngẫu nhiên để đảm bảo tái lập kết quả

# --- QNN (Qiskit) ---
QNN_NUM_QUBITS     = 2   # Số qubit trong mạch ZZFeatureMap
QNN_REPS           = 1   # Số lần lặp RealAmplitudes
QNN_FEATURE_DIM    = 2   # Số đặc trưng đầu vào cho QNN

# --- PyTorch Training ---
BATCH_SIZE    = 4       # Kích thước batch — phù hợp dataset nhỏ 100 mẫu
LEARNING_RATE = 0.001   # Tốc độ học cho Adam optimizer

# --- Đường dẫn lưu file ---
FEATURE_SAVE_DIR  = "saved_features"   # Thư mục lưu đặc trưng lượng tử (.npy)
MODEL_SAVE_DIR    = "saved_models"     # Thư mục lưu trọng số mô hình
USE_GDRIVE        = False              # Đặt True khi chạy Colab + Google Drive

# --- Device (tự động detect GPU/CPU) ---
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[config] Device: {DEVICE}")
