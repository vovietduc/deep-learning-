# Gỡ lỗi

| Hiện tượng | Cách xử lý |
|---|---|
| `No module named transformers/torch` | Đang dùng sai Python hoặc chưa cài xong. Chạy `py -3.14 setup_env.py`; dùng `.\.venv\Scripts\python.exe` thay `python`. |
| `py` không được nhận diện | Cài Python Launcher hoặc dùng đường dẫn đầy đủ đến python.exe 3.14.6. |
| Không thấy file script | Mở thư mục chứa setup_env.py bằng Open Folder. Gõ `dir` để kiểm tra thư mục terminal. |
| Nhấp đúp CMD không chạy | Giải nén ZIP trước. Chạy CMD từ terminal để xem lỗi. |
| `No matching distribution` | Kiểm tra Python 3.14 x64 tiêu chuẩn, kết nối PyPI và phiên bản pip. Dự án không kiểm chứng Windows ARM64 hoặc Python free-threaded. |
| `DLL load failed` của torch | Kiểm tra Python 64-bit; cài Microsoft Visual C++ Redistributable x64 từ Microsoft, mở lại terminal và chạy check_env.py. |
| `Can't load configuration`, timeout, SSL | Kiểm tra Internet/VPN/proxy, đồng hồ máy và truy cập huggingface.co; chạy lại. Không tắt xác minh SSL. |
| Cảnh báo chưa có HF_TOKEN | Các mô hình dùng trong bài công khai; không bắt buộc đăng nhập. Nếu gặp giới hạn tải, chờ rồi thử lại. |
| `classifier.weight/bias ... newly initialized` | Bình thường ở Bài 2: đầu phân loại mới cần được huấn luyện. |
| Cảnh báo symlink Windows | Không làm thay đổi kết quả; chương trình đã tắt cảnh báo này. |
| CPU chạy chậm | Dùng `--smoke --output-dir outputs/smoke` kiểm tra trước; đóng ứng dụng nặng. Không dùng smoke metrics làm kết quả thực nghiệm. |
| Hết RAM | Giảm `--batch-size 8` hoặc `--batch-size 4`. |
| Dự đoán tiếng Việt sai | Các mô hình và dataset bài này dành cho tiếng Anh. |
| Accuracy khác báo cáo | Kiểm tra seed, dữ liệu, phiên bản thư viện và tham số. Phần cứng/hệ điều hành vẫn có thể gây sai khác số học. |

Khi cần hỗ trợ, gửi đầy đủ phần lỗi cuối terminal và kết quả `check_env.py`, không chỉ ảnh thanh tiến trình.
