# Fashion-MNIST Classification Project

## 1. Giới thiệu

Dự án thực hiện bài toán phân loại hình ảnh thời trang (Fashion-MNIST) bằng Deep Learning sử dụng PyTorch.

Mục tiêu:
- Xây dựng quy trình xử lý dữ liệu (Data Pipeline).
- Xây dựng và huấn luyện mô hình MLP.
- Thử nghiệm các Hyperparameters để lựa chọn mô hình phù hợp.
- Đánh giá mô hình trên tập Test.
- So sánh kết quả giữa MLP và CNN.

## 2. Dataset Description

- **Dataset:** Fashion-MNIST
- **Source:** Zalando Research
- **Total images:** 70,000
- **Image size:** 28 × 28 pixels
- **Color:** Grayscale (1 channel)
- **Number of classes:** 10
- **Task:** Multi-class Image Classification

### 10 classes

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## 3. Dataset Splitting

Dataset được chia thành ba tập:

| Dataset | Number of images | Purpose |
|---|---:|---|
| Train | 54,000 | Huấn luyện mô hình |
| Validation | 6,000 | Chọn Hyperparameters và theo dõi Overfitting |
| Test | 10,000 | Đánh giá cuối cùng |

Validation được tách từ tập Training ban đầu với random seed cố định.

## 4. Data Preprocessing

Các bước tiền xử lý:
- Kiểm tra nhãn và dữ liệu ảnh.
- Kiểm tra ảnh trống, ảnh trùng lặp và Data Leakage.
- Phân tích phân bố các lớp.
- Chuyển đổi ảnh sang Tensor.
- Chuẩn hóa dữ liệu bằng Mean và Standard Deviation tính từ tập Train.
- Đóng gói dữ liệu bằng DataLoader.

## 5. Models

Dự án triển khai và thử nghiệm:

- **MLP (Multi-Layer Perceptron):** mô hình phân loại cơ bản.
- **CNN (Convolutional Neural Network):** mô hình bổ sung để so sánh khả năng học đặc trưng không gian của ảnh.

## 6. Experiments

Thử nghiệm các Hyperparameters:
- Learning Rate.
- Batch Size.
- Hidden Layers.
- Optimizer.
- Dropout.
- Weight Decay.

Mô hình được lựa chọn dựa trên Validation Accuracy.

## 7. Evaluation

Các chỉ số đánh giá:
- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion Matrix.

Ngoài ra, dự án trực quan hóa Training Loss, Validation Loss và kết quả dự đoán.

## 8. Project Structure

```text
FashionMNIST_Project/
│
├── fashionmnist_practice1_full.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   └── FashionMNIST/
│
├── models/
│   └── best_mlp_fashionmnist.pth
│
└── results/
    ├── confusion_matrix.png
    ├── training_loss.png
    ├── accuracy_comparison.png
    └── experiment_results.csv
```

## 9. Installation

Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

## 10. Run Project

Mở Jupyter Notebook:

```bash
jupyter notebook
```

Mở file `fashionmnist_practice1_full.ipynb` và chạy lần lượt các cell từ đầu đến cuối.

Dataset sẽ được tải tự động thông qua `torchvision.datasets.FashionMNIST` trong lần chạy đầu tiên.

## 11. Results

Các kết quả thực nghiệm, biểu đồ và mô hình được lưu trong thư mục `results/` và `models/` sau khi thực hiện chạy và xuất kết quả từ notebook.

**Note:** Các chỉ số đánh giá thực tế được cập nhật sau khi chạy hoàn chỉnh notebook.
