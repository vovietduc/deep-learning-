# Nguồn dữ liệu

Dataset: https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
Nguồn gốc: Bo Pang và Lillian Lee, Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales, ACL 2005.
Homepage: https://www.cs.cornell.edu/people/pabo/movie-review-data/

Các CSV là tập con phục vụ bài thực hành, giữ split gốc và chọn với seed 42. Xem provenance.json để biết số mẫu và fingerprint. Không tự gán giấy phép mới cho nội dung dataset; tham khảo điều kiện sử dụng ở nguồn gốc.

Mô hình Bài 2: https://huggingface.co/prajjwal1/bert-tiny
Mô hình Bài 1: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
Trọng số BERT đã fine-tune đi kèm chỉ phục vụ tái hiện bài thực hành. Model card gốc của BERT-Tiny ghi license Apache-2.0.
