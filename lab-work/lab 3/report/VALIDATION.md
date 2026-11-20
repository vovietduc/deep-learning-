# Kiểm chứng source

Ngày chạy: 07/11/2026.

## Môi trường chạy thật

- CPython 3.14.6, Linux x86_64, CPU, 4 luồng PyTorch.
- torch 2.15.0+cpu; transformers 5.22.2; datasets 5.1.0; evaluate 0.4.6; accelerate 1.15.0; numpy 2.5.4.
- check_env.py: tất cả import và Trainer thành công.
- exercise1.py: suy luận hoàn tất, xuất tokens và nhãn cho hai câu tiếng Anh.
- exercise2.py --cpu: 315 bước, 5 epoch, 1.000 mẫu train/200 validation/200 test; fine-tune pretrained BERT thật.
- Accuracy test 0.63; F1 lớp positive 0.5595238095. Không phải kết quả smoke test.
- verify_results.py: tính lại accuracy/F1 từ CSV, nạp model offline và đối chiếu dự đoán toàn bộ 200 mẫu, kiểm tra không trùng văn bản giữa các split.
- predict.py: nạp lại model đã fine-tune và suy luận.
- Các file Python được kiểm tra cú pháp bằng compileall.

## Windows

requirements-windows.lock được phân giải cho Windows x64 / Python 3.14 bằng uv.
Đã kiểm tra giải phụ thuộc với only-binary; wheel cho CPython 3.14 x64 của các gói lớn được ghi tại windows_wheels.json.
**Chưa thực thi trên Windows hoặc trong giao diện VS Code.** Các file CMD và setup_env.py được chuẩn bị cho môi trường này; không khẳng định đã kiểm thử Windows.

## Tái hiện

Chạy check_env.py, src/exercise1.py, src/exercise2.py --cpu và verify_results.py bằng cùng interpreter .venv.
Source thiết lập seed 42; khác phần cứng/hệ điều hành vẫn có thể gây sai khác số học.
Chạy lại mặc định cập nhật outputs/exercise2; dùng --output-dir outputs/my_run nếu muốn giữ kết quả mẫu.
Báo cáo PDF mô tả lần chạy đã đóng gói, không tự sửa sau mỗi lần huấn luyện.

Các log cuối cùng được giữ lại trong outputs/ và report/; bản đóng gói bỏ checkpoint trung gian và môi trường .venv, nhưng giữ model safetensors đã fine-tune.
