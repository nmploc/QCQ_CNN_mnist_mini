@echo off
REM activate.bat — Khởi động môi trường venv cho project QCQ-CNN
REM Cách dùng: double-click hoặc chạy trong terminal: activate.bat

echo ==============================================
echo   QCQ-CNN Project — Virtual Environment
echo ==============================================
call "%~dp0venv\Scripts\activate.bat"
echo.
echo [OK] venv đã kích hoạt. Python: & python --version
echo [OK] Thư mục làm việc: %~dp0
echo.
echo Lệnh nhanh:
echo   python config.py                          -- Kiểm tra config
echo   python step1_setup.py                     -- Kiểm tra môi trường
echo   python step3a_quanvo_layer.py             -- Test Quanvolutional Layer
echo   python step4b_model_qcq_cnn.py            -- Test QCQ-CNN model
echo   python main.py --epochs 3                 -- Pipeline nhanh 3 epoch
echo   python main.py --epochs 100 --skip-baseline  -- Train QCQ-CNN đầy đủ
echo.
cmd /k
