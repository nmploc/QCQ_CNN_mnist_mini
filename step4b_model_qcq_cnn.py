# step4b_model_qcq_cnn.py — Mô hình QCQ-CNN (PyTorch + Qiskit)
# Chạy độc lập: python step4b_model_qcq_cnn.py

import torch
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear, Dropout2d

# QUAN TRỌNG: EstimatorQNN không hỗ trợ CUDA
# TorchConnector wrap QNN thành PyTorch layer nhưng tính toán vẫn trên CPU
# Chỉ CNN backbone (Conv2d, Linear) mới được .to(device) lên GPU
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from config import QNN_NUM_QUBITS, QNN_REPS, QNN_FEATURE_DIM


def create_qnn() -> EstimatorQNN:
    """
    Tạo Mạng Nơ-ron Lượng tử (QNN) dùng Qiskit.

    Cấu trúc mạch:
    [ZZFeatureMap(2)] → [RealAmplitudes(2, reps=1)]

    - ZZFeatureMap: Mã hóa vector 2D vào trạng thái lượng tử
      Dùng cổng H và cổng ZZ-interaction: e^{i(π-x₁)(π-x₂)Z⊗Z}

    - RealAmplitudes: Ansatz với RY và CNOT gates
      Tham số: 4 tham số (2 qubits × 2 layers = 2×reps+2 = 4)

    - EstimatorQNN: Tính kỳ vọng ⟨H⟩ = ⟨ψ|Z⊗I|ψ⟩ (mặc định)
      Đầu ra: scalar ∈ [-1, 1]

    ⚠️ Lưu ý phần cứng:
      EstimatorQNN của Qiskit KHÔNG hỗ trợ CUDA — luôn chạy trên CPU.
      QCQ_CNN_Model.forward() sẽ tự xử lý việc chuyển tensor CPU↔GPU.

    Trả về:
        EstimatorQNN với num_inputs=2, num_weights=4
    """
    # Luôn chạy trên CPU (Qiskit không hỗ trợ CUDA)
    feature_map = ZZFeatureMap(QNN_NUM_QUBITS)             # 2 qubit, mã hóa 2 đặc trưng
    ansatz      = RealAmplitudes(QNN_NUM_QUBITS, reps=QNN_REPS)  # 2 qubit, 1 lớp lặp, 4 tham số

    # Ghép feature map + ansatz thành một mạch hoàn chỉnh
    qc = QuantumCircuit(QNN_NUM_QUBITS)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz,      inplace=True)

    # Tạo EstimatorQNN — hỗ trợ tính gradient tự động qua parameter-shift rule
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,   # Tham số đầu vào (x₁, x₂)
        weight_params=ansatz.parameters,        # Tham số có thể học (θ₀..θ₃)
        input_gradients=True,                   # Bật tính gradient theo đầu vào
    )
    return qnn


class QCQ_CNN_Model(Module):
    """
    Mô hình QCQ-CNN đề xuất: Kết hợp CNN cổ điển và QNN.

    Luồng dữ liệu:
    Input (N, 4, 14, 14)
        → conv1: (N, 8, 12, 12)    [Conv2d(4→8, 3×3) + ReLU]
        → MaxPool: (N, 8, 6, 6)
        → conv2: (N, 16, 4, 4)     [Conv2d(8→16, 3×3) + ReLU]
        → MaxPool: (N, 16, 2, 2)
        → Dropout2d
        → Flatten: (N, 64)
        → fc2: (N, 2)              [Linear(64→2), lazy init]
        → [CPU] QNN: (N, 1)        [TorchConnector — luôn CPU]
        → [GPU] fc3: (N, 2)        [Linear(1→2)]

    Ghi chú device:
        - conv1, conv2, dropout, fc2, fc3: Chạy trên device của input (GPU nếu có)
        - qnn (TorchConnector): Luôn chạy trên CPU (giới hạn của Qiskit)
        - forward() tự động chuyển tensor CPU↔GPU tại đúng điểm cần thiết

    Input : (batch, 4, 14, 14) — đặc trưng lượng tử channels-first
    Output: (batch, 2)          — logit 2 lớp
    """

    def __init__(self, qnn=None):
        super().__init__()
        if qnn is None:
            qnn = create_qnn()
        # Lớp Conv2d thứ nhất: 4 kênh vào → 8 kênh ra, kernel 3×3
        self.conv1   = Conv2d(4, 8, 3)
        # Lớp Conv2d thứ hai: 8 kênh vào → 16 kênh ra, kernel 3×3
        self.conv2   = Conv2d(8, 16, 3)
        # Dropout ngẫu nhiên các kênh (chống overfitting)
        self.dropout = Dropout2d()
        # fc2 được khởi tạo động ở forward() vì phụ thuộc kích thước flatten
        self.fc2     = None
        # TorchConnector: Kết nối QNN Qiskit với PyTorch autograd
        # KHÔNG gọi .to(device) cho self.qnn — Qiskit không hỗ trợ CUDA
        self.qnn     = TorchConnector(qnn)
        # Lớp cuối: chuyển scalar QNN sang 2 logit lớp
        self.fc3     = Linear(1, 2)

    def forward(self, x):
        """
        Forward pass của QCQ-CNN.

        Tham số:
            x: tensor shape (batch, 4, 14, 14) — có thể ở GPU

        Trả về:
            tensor shape (batch, 2) — logit 2 lớp
        """
        # x nằm trên GPU nếu được chuyển bằng .to(device) từ train loop

        # --- Phần CNN: Chạy trên GPU ---
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = self.dropout(x)

        # Flatten về vector 1D (giữ nguyên batch dimension)
        x = x.view(x.shape[0], -1)

        # Khởi tạo fc2 động dựa trên kích thước thực tế sau flatten
        if self.fc2 is None:
            self.fc2 = Linear(x.shape[1], 2).to(x.device)

        # FC2: Nén xuống 2 đặc trưng để đưa vào QNN (2 qubit cần 2 input)
        x = self.fc2(x)   # x vẫn trên GPU ở đây

        # --- Chuyển CPU cho QNN (Qiskit không hỗ trợ CUDA) ---
        # .cpu() tách tensor khỏi GPU và tạo bản sao trên RAM
        # Gradient vẫn được giữ nguyên qua phép chuyển này
        x_cpu = x.cpu()

        # QNN: Xử lý 2 đặc trưng qua mạch lượng tử trên CPU
        # Đầu ra: (batch, 1) — scalar kỳ vọng lượng tử ∈ [-1, 1]
        x_qnn = self.qnn(x_cpu)

        # --- Chuyển trở lại device gốc (GPU) để fc3 và loss tính trên GPU ---
        # Đảm bảo gradient flow thông suốt từ fc3 → QNN → fc2 → CNN
        x_qnn = x_qnn.to(x.device)

        # FC3: Chuyển scalar QNN → 2 logit lớp
        return self.fc3(x_qnn)


if __name__ == "__main__":
    import torch

    # Test QNN
    qnn = create_qnn()
    print(f"QNN num_inputs  : {qnn.num_inputs}")    # 2
    print(f"QNN num_weights : {qnn.num_weights}")   # 4

    # Test model forward pass
    model = QCQ_CNN_Model(qnn)

    # Warm-up để khởi tạo fc2 (lazy layer)
    dummy = torch.randn(2, 4, 14, 14)  # batch=2
    out = model(dummy)

    print(f"Output shape: {out.shape}")  # (2, 2)
    assert out.shape == (2, 2), f"❌ Output shape sai! Nhận {out.shape}"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng tham số có thể học: {total_params:,}")
    print("✅ QCQ_CNN_Model forward pass thành công")
