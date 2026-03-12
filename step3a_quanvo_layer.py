# step3a_quanvo_layer.py — Định nghĩa class Quanvolutional_Layer
# Chạy độc lập: python step3a_quanvo_layer.py

import math

from config import (
    SIZE_FILTER, LAYERS_FILTER, STRIDE_FILTER, RANDOM_SEED
)

# --- PennyLane ---
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers


class Quanvolutional_Layer:
    """
    Tầng Quanvolutional: Thay bộ lọc CNN cổ điển bằng mạch lượng tử ngẫu nhiên.

    Tham số:
        size_filter  : int — Kích thước bộ lọc vuông (ví dụ: 2 → bộ lọc 2×2)
        layers_filter: int — Số lớp RandomLayers (= số 'kênh' đầu ra)
        stride_filter: int — Bước trượt của bộ lọc
        padding      : int — Đệm (chưa cài đặt, để 0)

    Quá trình xử lý:
        1. Angle Encoding: θ = π·x, mỗi pixel mã hóa bằng RY(θ)
        2. RandomLayers: Mạch ngẫu nhiên với tham số cố định
        3. Đo lường: Kỳ vọng PauliZ ⟨Z_j⟩ = P(|0⟩) - P(|1⟩) ∈ [-1, 1]
    """

    def __init__(self, size_filter, layers_filter, stride_filter, padding=0):
        self.size_filter   = size_filter
        self.layers_filter = layers_filter
        self.stride_filter = stride_filter
        self.padding       = padding

        # Số qubit = size_filter² (bộ lọc 2×2 → 4 qubit)
        n_qubits = self.size_filter * self.size_filter

        # --- Chọn backend simulator nhanh nhất có thể ---
        # lightning.qubit: Được viết bằng C++/C với BLAS, nhanh hơn default.qubit ~5-10x
        # default.qubit: Thuần Python, dùng làm fallback nếu lightning chưa cài
        try:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
            self._backend_name = "lightning.qubit (C++ — nhanh)"
        except Exception:
            self.dev = qml.device("default.qubit", wires=n_qubits)
            self._backend_name = "default.qubit (Python — fallback)"

        # Tham số ngẫu nhiên cho RandomLayers — cố định để đảm bảo tái lập
        # Shape: (layers_filter, n_qubits) — mỗi layer có n_qubits tham số
        self.rand_params = np.random.uniform(
            high=2 * np.pi,
            size=(self.layers_filter, n_qubits)
        )

        print(f"✅ Quanvolutional_Layer khởi tạo thành công!")
        print(f"   Backend   : {self._backend_name}")
        print(f"   Bộ lọc   : {size_filter}×{size_filter}, Stride: {stride_filter}")
        print(f"   Số qubit  : {n_qubits}, Số kênh đầu ra: {layers_filter}")

    def circuit(self, phi):
        """
        Tạo và chạy mạch lượng tử cho một patch.

        Tham số:
            phi: vector pixel của patch, shape=(n_qubits,) — giá trị [0,1]

        Trả về:
            list shape=(layers_filter * n_qubits,) — các giá trị kỳ vọng PauliZ
        """
        n_qubits = self.size_filter * self.size_filter

        # interface='numpy': Tắt autograd của PennyLane vì rand_params không cần gradient
        # → nhanh hơn so với interface='auto' mặc định khi chỉ cần inference
        @qml.qnode(self.dev, interface='numpy')
        def _circuit(phi):
            op_list = []

            # === ANGLE ENCODING ===
            # Mỗi pixel phi[j] ∈ [0,1] được mã hóa bằng góc θ = π * phi[j]
            # RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
            for j in range(n_qubits):
                qml.RY(np.pi * phi[j], wires=j)

            # === RANDOM QUANTUM CIRCUIT ===
            # Áp dụng RandomLayers với mỗi hàng của rand_params
            for k in self.rand_params:
                tmp = np.expand_dims(k, axis=0)  # Shape: (1, n_qubits)
                RandomLayers(tmp, wires=list(range(n_qubits)))

                # === ĐO LƯỜNG: Kỳ vọng Pauli-Z trên mỗi qubit ===
                # ⟨Z_j⟩ = P(|0⟩) - P(|1⟩) ∈ [-1, 1]
                for j in range(n_qubits):
                    op_list.append(qml.expval(qml.PauliZ(j)))

            return op_list

        op_list = _circuit(phi)
        op_list = np.array(op_list)
        # Reshape về (layers_filter, n_qubits)
        return list(np.reshape(op_list, (self.layers_filter, n_qubits)))

    def process_image(self, image_batch):
        """
        Áp dụng Quanvolutional layer lên toàn bộ batch ảnh.

        Tham số:
            image_batch: numpy array shape (N, H, W, 1)

        Trả về:
            list N phần tử, mỗi phần tử numpy array shape (H_out, W_out, C_out)
        """
        from tqdm.auto import tqdm

        H, W = image_batch[0].shape[0], image_batch[0].shape[1]

        # Tính kích thước đầu ra theo công thức CNN:
        # H_out = floor((H + 2*padding - size_filter) / stride) + 1
        H_out = math.floor((H + 2 * self.padding - self.size_filter) / self.stride_filter) + 1
        C_out = self.layers_filter * (self.size_filter * self.size_filter)

        print(f"   Kích thước input mỗi ảnh : {H}×{W}")
        print(f"   Kích thước output mỗi ảnh: {H_out}×{H_out}×{C_out}")

        if H_out < 1:
            raise ValueError("Kích thước bộ lọc quá lớn so với ảnh!")

        collection = []

        # Dùng tqdm để hiển thị thanh tiến trình + tốc độ + ETA
        for img in tqdm(image_batch, desc="⚛️  Quanvolution"):
            # Khởi tạo tensor đầu ra cho ảnh hiện tại
            out = np.zeros((H_out, H_out, C_out))

            # Trượt bộ lọc với stride
            for j in range(0, H, self.stride_filter):
                for k in range(0, W, self.stride_filter):
                    # Bỏ qua patch vượt ra ngoài biên ảnh
                    if (j + self.size_filter > H) or (k + self.size_filter > W):
                        continue

                    # Lấy pixel trong patch 2×2
                    phi = []
                    for l in range(j, j + self.size_filter):
                        for m in range(k, k + self.size_filter):
                            phi.append(img[l][m][0])  # Kênh 0 (grayscale)

                    # Chạy mạch lượng tử và lấy kết quả đo
                    q_results = self.circuit(phi)

                    # Ghi kết quả vào tensor đầu ra
                    ctr = 0
                    for x in q_results:
                        for y_val in x:
                            out[j // self.stride_filter, k // self.stride_filter, ctr] = y_val
                            ctr += 1

            collection.append(out)

        print(f"✅ Xử lý xong {len(image_batch)} ảnh!")
        return collection

    # Alias để giữ tương thích với notebook (notebook dùng .call())
    call = process_image


if __name__ == "__main__":
    import numpy as test_np

    # Test với một ảnh giả 28×28
    fake_image = test_np.random.rand(28, 28).astype(test_np.float32)

    ql = Quanvolutional_Layer(
        size_filter=SIZE_FILTER,
        layers_filter=LAYERS_FILTER,
        stride_filter=STRIDE_FILTER
    )

    # Chạy thử một patch 2×2
    patch = fake_image[0:2, 0:2].flatten()
    result = ql.circuit(patch)

    print(f"\nInput patch  : {patch}")
    print(f"Circuit output ({LAYERS_FILTER} layer × 4 qubit): {result}")
    total_outputs = sum(len(r) for r in result)
    print(f"Tổng số giá trị đầu ra: {total_outputs}")
    assert len(result) == LAYERS_FILTER, f"❌ Số layer output sai! Kỳ vọng {LAYERS_FILTER}, nhận {len(result)}"
    print("✅ Quanvolutional_Layer hoạt động đúng")
