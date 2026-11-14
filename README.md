# Fake News Detection ML

Dự án môn Machine Learning dùng để phân loại bài báo tiếng Anh thành:

- `REAL` (`0`): có xu hướng là tin thật
- `FAKE` (`1`): có xu hướng là tin giả

Hệ thống gồm 4 mô hình:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVC
3. BiLSTM
4. DistilBERT

Giao diện được xây dựng bằng Streamlit và có chức năng so sánh dự đoán giữa các mô hình.

> Lưu ý: kết quả chỉ mang tính thử nghiệm học máy. Hệ thống phân tích đặc trưng ngôn ngữ, không tự kiểm chứng sự kiện hoặc nguồn báo.

---

## 1. Yêu cầu môi trường

Khuyến nghị:

- Windows 10 hoặc Windows 11
- Python 3.11 hoặc 3.12
- RAM tối thiểu 8 GB
- Dung lượng trống tối thiểu 5 GB
- Git hoặc GitHub Desktop

Có thể chạy bằng CPU. DistilBERT sẽ chạy chậm nếu máy không có GPU CUDA.

---

## 2. Tải source code

Repository:

https://github.com/ThaiAnhh-2507/FakeNewsDetection-ML

### Dùng GitHub Desktop

1. Mở GitHub Desktop.
2. Chọn `File` → `Clone repository`.
3. Chọn tab `URL`.
4. Nhập URL repository ở trên.
5. Chọn thư mục lưu và nhấn `Clone`.
6. Chọn `Repository` → `Open in Visual Studio Code`.

---

## 3. Tạo môi trường Python

Mở Terminal trong Visual Studio Code tại thư mục gốc của dự án.

### PowerShell trên Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu máy không có Python 3.12 nhưng đã có Python phù hợp:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Kiểm tra môi trường:

```powershell
python --version
python -c "import pandas, sklearn, streamlit, torch; print('Môi trường đã sẵn sàng')"
```

Nếu dùng DistilBERT, kiểm tra thêm:

```powershell
python -c "import transformers; print(transformers.__version__)"
```

---

## 4. Chuẩn bị dữ liệu

Dataset không được lưu trong GitHub vì dung lượng lớn. Mỗi thành viên tải dữ liệu từ các nguồn dưới đây rồi đặt đúng thư mục.

### Link tải dataset

| Dataset | Mục đích | Link tải |
|---|---|---|
| ISOT Fake News Dataset | Bắt buộc để chuẩn bị dữ liệu và huấn luyện các model | [Tải trực tiếp file ZIP từ ISOT Research Lab](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip) |
| ISOT Dataset Description | Tài liệu mô tả chính thức | [Xem tài liệu PDF](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf) |
| WELFake Dataset | Không bắt buộc, chỉ dùng cho đánh giá external | [Mở trang tải WELFake trên Zenodo](https://zenodo.org/records/4561253) |
| WELFake CSV | Tải trực tiếp file CSV, dung lượng khoảng 245 MB | [Tải trực tiếp WELFake_Dataset.csv](https://zenodo.org/api/records/4561253/files/WELFake_Dataset.csv/content) |

> Nếu link tải trực tiếp ISOT không hoạt động, mở [trang dataset chính thức của ISOT Research Lab](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) rồi chọn **Click here to download the ISOT Fake News Dataset**.

### 4.1. ISOT Fake News Dataset

Sau khi tải và giải nén file ZIP, lấy hai file:

```text
data/raw/Fake.csv
data/raw/True.csv
```

Sau khi chép dữ liệu, cấu trúc phải là:

```text
FakeNewsDetection-ML/
├── data/
│   └── raw/
│       ├── Fake.csv
│       └── True.csv
├── src/
├── app.py
└── requirements.txt
```

ISOT được dùng để chuẩn bị tập train, validation và test.

### 4.2. WELFake Dataset, không bắt buộc

WELFake chỉ dùng để đánh giá khả năng tổng quát hóa ngoài bộ ISOT. Đặt file tại:

```text
data/external/WELFake_Dataset.csv
```

File cần có các cột:

```text
Unnamed: 0, title, text, label
```

Quy ước nhãn của file đang dùng trong dự án:

```text
0 = REAL
1 = FAKE
```

Không cần WELFake nếu chỉ muốn train model và chạy giao diện web.

---

## 5. Chuẩn bị dữ liệu ISOT

Sau khi đã có `Fake.csv` và `True.csv`, chạy:

```powershell
python src/prepare_data.py
```

Script sẽ:

- Đọc hai file ISOT
- Làm sạch tiêu đề và nội dung
- Bỏ các bài quá ngắn
- Bỏ bài trùng hoặc xung đột nhãn
- Chia dữ liệu theo tỷ lệ 70% train, 15% validation, 15% test

Các file được tạo tại:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/split_stats.json
```

---

## 6. Huấn luyện mô hình

Không bắt buộc train cả 4 model. Chỉ cần baseline là giao diện đã có thể chạy.

### 6.1. Logistic Regression và Linear SVC

```powershell
python src/train_baseline.py
```

Kết quả được lưu tại:

```text
models/baseline/tfidf.joblib
models/baseline/logistic.joblib
models/baseline/svc.joblib
```

Đây là bước nên chạy đầu tiên vì nhanh và chỉ cần CPU.

### 6.2. BiLSTM

```powershell
python src/train_lstm.py
```

Cấu hình nhẹ hơn cho máy RAM thấp:

```powershell
python src/train_lstm.py --batch-size 32 --max-len 200
```

Kết quả được lưu tại:

```text
models/lstm/lstm.pt
models/lstm/vocab.json
```

### 6.3. DistilBERT

Cấu hình thử nghiệm trên máy CPU:

```powershell
python src/train_bert.py --epochs 1 --batch-size 4 --max-len 128
```

Cấu hình đầy đủ hơn:

```powershell
python src/train_bert.py --epochs 3 --batch-size 4 --max-len 256
```

Nếu máy hết RAM, giảm xuống:

```powershell
python src/train_bert.py --epochs 1 --batch-size 2 --max-len 128
```

Kết quả được lưu tại:

```text
models/bert/
```

Lần chạy đầu tiên cần kết nối Internet để tải `distilbert-base-uncased` từ Hugging Face.

### 6.4. Kết quả đánh giá nội bộ

Sau khi train, các chỉ số được cập nhật tại:

```text
models/metrics.json
```

Các chỉ số chính gồm Accuracy, Precision, Recall, F1-score và confusion matrix.

---

## 7. Đánh giá trên WELFake

Chỉ thực hiện khi đã có:

- Ít nhất một baseline trong `models/baseline/`
- File `data/external/WELFake_Dataset.csv`

Chạy thử trên 10.000 bài:

```powershell
python src/eval_external.py --sample 10000
```

Chạy toàn bộ dữ liệu:

```powershell
python src/eval_external.py
```

Kết quả được lưu tại:

```text
data/external/eval_results/metrics.json
data/external/eval_results/logistic_errors.csv
data/external/eval_results/svc_errors.csv
```

---

## 8. Chạy giao diện web

Sau khi đã train ít nhất baseline, chạy:

```powershell
python -m streamlit run app.py
```

Nếu trình duyệt không tự mở, truy cập địa chỉ Streamlit hiển thị trong Terminal, thường là:

```text
http://localhost:8501
```

Giao diện chỉ hiển thị các model đã có đủ file trong thư mục `models`.

Ví dụ:

- Có `models/baseline/` → hiện Logistic Regression và Linear SVC
- Có `models/lstm/` → hiện thêm BiLSTM
- Có `models/bert/` → hiện thêm DistilBERT

Để dừng ứng dụng, quay lại Terminal và nhấn `Ctrl + C`.

---

## 9. Chạy nhanh bằng model được chia sẻ sẵn

Nếu một thành viên trong nhóm chia sẻ `models.zip`:

1. Clone repository.
2. Cài thư viện bằng `requirements.txt`.
3. Giải nén `models.zip` vào thư mục gốc.
4. Kiểm tra cấu trúc:

```text
FakeNewsDetection-ML/
├── models/
│   ├── baseline/
│   ├── lstm/
│   └── bert/
├── src/
└── app.py
```

5. Chạy:

```powershell
python -m streamlit run app.py
```

Trong trường hợp này không cần tải dataset hoặc train lại model để sử dụng giao diện.

---

## 10. Cấu trúc dự án

```text
FakeNewsDetection-ML/
├── data/
│   ├── raw/                 # Fake.csv và True.csv
│   ├── processed/           # train.csv, val.csv, test.csv
│   └── external/            # WELFake_Dataset.csv
├── models/
│   ├── baseline/            # TF-IDF, Logistic Regression, Linear SVC
│   ├── lstm/                # BiLSTM và vocabulary
│   ├── bert/                # DistilBERT
│   └── metrics.json
├── src/
│   ├── common.py
│   ├── eval_external.py
│   ├── lstm_model.py
│   ├── predictor.py
│   ├── prepare_data.py
│   ├── text_utils.py
│   ├── train_baseline.py
│   ├── train_bert.py
│   └── train_lstm.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 11. Lỗi thường gặp

### `ModuleNotFoundError`

Đảm bảo `.venv` đã được kích hoạt rồi chạy:

```powershell
python -m pip install -r requirements.txt
```

### `Fatal error in launcher` khi dùng pip

Không dùng trực tiếp `pip`. Dùng:

```powershell
python -m pip install ten-thu-vien
```

Nếu vẫn lỗi, xóa `.venv` và tạo lại môi trường trong đúng thư mục dự án.

### Không thấy model trên giao diện

Kiểm tra model đã được train và có đủ file. Sau đó dừng Streamlit bằng `Ctrl + C` rồi mở lại.

### Không thấy `Fake.csv` hoặc `True.csv`

Đặt đúng hai file tại:

```text
data/raw/Fake.csv
data/raw/True.csv
```

### BERT chạy rất chậm

Máy không có CUDA sẽ train BERT bằng CPU. Dùng cấu hình nhẹ:

```powershell
python src/train_bert.py --epochs 1 --batch-size 2 --max-len 128
```

### PowerShell không cho kích hoạt `.venv`

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## 12. Quy trình tối thiểu để chạy từ đầu

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Đặt `Fake.csv` và `True.csv` vào `data/raw/`, sau đó:

```powershell
python src/prepare_data.py
python src/train_baseline.py
python -m streamlit run app.py
```

Ba lệnh cuối là quy trình tối thiểu để có dữ liệu, train baseline và mở giao diện.

---

## 13. Làm việc nhóm bằng GitHub Desktop

Mỗi thành viên nên làm trên branch riêng:

1. Clone repository bằng GitHub Desktop.
2. Chọn `Current branch` → `New branch`.
3. Đặt tên theo chức năng, ví dụ:
   - `feature-ui`
   - `feature-bert`
   - `fix-data-processing`
4. Sửa code và kiểm tra trên máy.
5. Nhập Summary trong GitHub Desktop.
6. Chọn `Commit to <branch>`.
7. Chọn `Publish branch` hoặc `Push origin`.
8. Tạo Pull Request vào nhánh `main` trên GitHub.

Không commit các mục sau:

```text
.venv/
data/raw/*.csv
data/external/*.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
models/baseline/
models/lstm/
models/bert/
src/__pycache__/
```

---

## 14. Ghi chú về kết quả

Điểm test nội bộ cao không đồng nghĩa model có thể kiểm chứng mọi tin tức ngoài thực tế. Dataset ISOT có đặc trưng nguồn và văn phong riêng. WELFake được dùng để quan sát hiện tượng domain shift, tức hiệu năng giảm khi model gặp dữ liệu từ miền khác.
