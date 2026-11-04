# Robust Multi-View 3D Human Pose Fusion

**Ray-Casting Occlusion Awareness and Zero-GT Kinematic Quality Gating**

## 1. Mục tiêu nghiên cứu (Research Objective)

Dự án nghiên cứu phương pháp kết hợp thông tin tư thế người 3D từ nhiều camera đơn (monocular cameras) được đồng bộ thời gian, nhằm cải thiện độ chính xác và độ tin cậy của kết quả ước lượng tư thế người trong các tình huống có che khuất (occlusion).

Đầu vào của hệ thống là các kết quả ước lượng tư thế 3D từ những phương pháp monocular Human Pose Estimation (HPE), chẳng hạn như WHAM.

Mục tiêu chính là khai thác thông tin bổ sung giữa các góc nhìn camera để cải thiện chất lượng tái tạo tư thế 3D, đồng thời hạn chế sự phụ thuộc vào dữ liệu ground truth trong quá trình đánh giá chất lượng và kết hợp tư thế.

### Các hướng nghiên cứu chính

**1.1. Ray-Casting Occlusion Awareness**

Sử dụng thông tin hình học camera và kỹ thuật ray-casting để phân tích khả năng che khuất giữa các đối tượng, các khớp cơ thể và các góc nhìn camera khác nhau.

Mục tiêu là xác định những quan sát có khả năng không đáng tin cậy do hiện tượng che khuất.

**1.2. Zero-GT Kinematic Quality Gating**

Đánh giá độ tin cậy của các dự đoán tư thế dựa trên những đặc trưng hình học và ràng buộc động học của cơ thể người, không yêu cầu nhãn ground truth 3D trong quá trình đánh giá chất lượng.

Các thông tin về chất lượng tư thế được sử dụng để điều chỉnh mức độ đóng góp của từng quan sát vào kết quả kết hợp cuối cùng.

**1.3. Multi-View 3D Pose Fusion**

Kết hợp các kết quả ước lượng tư thế 3D từ nhiều camera nhằm khai thác những thông tin bổ sung giữa các góc nhìn và giảm ảnh hưởng của các dự đoán sai lệch.

### Mục tiêu đánh giá

Các phương pháp được nghiên cứu và đánh giá thông qua những tiêu chí như:

- MPJPE (Mean Per Joint Position Error).
- PA-MPJPE (Procrustes Aligned MPJPE).
- Độ ổn định hình học của tư thế.
- Khả năng xử lý các tình huống che khuất.
- Hiệu năng tính toán và thời gian xử lý.

Lưu ý: Không sử dụng ground truth trong quá trình quality gating không đồng nghĩa với việc không cần ground truth khi đánh giá thực nghiệm.

---

## 2. Cấu trúc dự án (Project Structure)

Cấu trúc mã nguồn chính:

```text
ray_casting/
│
├── src/
│   └── ray_casting/
│       │
│       ├── main.py
│       ├── config.py
│       ├── __init__.py
│       │
│       ├── belief_calculator/
│       │   ├── config.py
│       │   ├── correction.py
│       │   ├── detector.py
│       │   ├── executor.py
│       │   └── optimization.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── keypoints_map.py
│       │   └── smpl/
│       │
│       ├── pose_pipeline/
│       │   ├── executor.py
│       │   └── smpl_runner.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── compat.py
│           └── json_io.py
│
├── inputs/
│   └── datasets/
│       └── mpi_inf_3dhpe/
│
├── outputs/
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Mô tả các thành phần

| Thành phần | Chức năng |
|---|---|
| `main.py` | Điểm bắt đầu thực thi chương trình, tiếp nhận tham số đầu vào |
| `config.py` | Cấu hình chung của hệ thống |
| `belief_calculator/` | Các module phát hiện, đánh giá độ tin cậy, hiệu chỉnh và tối ưu hóa tư thế |
| `belief_calculator/detector.py` | Phát hiện và phân tích các thông tin liên quan đến độ tin cậy |
| `belief_calculator/correction.py` | Hiệu chỉnh các dự đoán tư thế |
| `belief_calculator/optimization.py` | Các thuật toán tối ưu hóa |
| `belief_calculator/executor.py` | Điều phối quá trình tính toán trong belief calculator |
| `models/` | Các thành phần liên quan đến mô hình và biểu diễn dữ liệu tư thế |
| `models/keypoints_map.py` | Ánh xạ và xử lý dữ liệu keypoints |
| `pose_pipeline/` | Pipeline xử lý dữ liệu tư thế đầu vào |
| `pose_pipeline/executor.py` | Điều phối pipeline xử lý và xuất kết quả |
| `pose_pipeline/smpl_runner.py` | Các thành phần thực thi liên quan đến SMPL |
| `utils/` | Các tiện ích hỗ trợ xử lý dữ liệu |

Thư mục `inputs/datasets/` lưu trữ dataset và các kết quả trung gian từ WHAM.

Thư mục `outputs/` lưu trữ kết quả thực nghiệm và các dữ liệu được xuất ra trong quá trình chạy chương trình.

---

## 3. Cài đặt và thư viện phụ thuộc (Installation and Dependencies)

### 3.1. Yêu cầu môi trường

- Python 3.
- pip.
- Git.
- Các thư viện khoa học và tính toán cần thiết.

Phiên bản Python và các thư viện nên được kiểm tra tương thích với môi trường thực nghiệm.

### 3.2. Clone repository

```bash
git clone git@github.com:bkhnk48/ray_casting.git

cd ray_casting
```

### 3.3. Tạo môi trường ảo

Trên macOS hoặc Linux:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Nâng cấp pip:

```bash
python -m pip install --upgrade pip
```

### 3.4. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Các thư viện cơ bản được sử dụng:

| Thư viện | Mục đích |
|---|---|
| NumPy | Tính toán và xử lý dữ liệu dạng mảng |
| SciPy | Các thuật toán tính toán khoa học và hình học |
| Joblib | Lưu trữ và tải dữ liệu |
| tqdm | Hiển thị tiến trình xử lý |

Ngoài ra, tùy theo các module và mô hình được sử dụng, môi trường có thể cần thêm các thư viện như PyTorch, OpenCV, SMPL-X và những dependencies của WHAM.

Các dependencies bổ sung cần được cài đặt theo phiên bản tương thích với từng module.

---

## 4. Chạy chương trình (Running the Program)

### 4.1. Chạy với dữ liệu mặc định

Từ thư mục gốc của repository, thực hiện:

```bash
python src/ray_casting/main.py
```

Theo cấu hình hiện tại trong `main.py`, chương trình sử dụng file PKL mặc định:

```text
inputs/datasets/mpi_inf_3dhpe/S8/Seq1/imageSequences/wham_video_0.pkl
```

File PKL này chứa kết quả ước lượng tư thế 3D được tạo ra từ WHAM.

Sau khi nhận được đường dẫn đầu vào, chương trình gọi:

```python
run_pose_export(args.pkl)
```

để thực thi pipeline xử lý và xuất dữ liệu tư thế.

### 4.2. Chỉ định file PKL khác

Người dùng có thể chỉ định file PKL thông qua tham số `--pkl`.

Cú pháp:

```bash
python src/ray_casting/main.py --pkl PATH_TO_PKL_FILE
```

Ví dụ:

```bash
python src/ray_casting/main.py \
    --pkl inputs/datasets/mpi_inf_3dhpe/S8/Seq1/imageSequences/wham_video_0.pkl
```

Hoặc sử dụng đường dẫn tuyệt đối:

```bash
python src/ray_casting/main.py \
    --pkl /path/to/your/wham_output.pkl
```

### 4.3. Xem hướng dẫn tham số

```bash
python src/ray_casting/main.py --help
```

Tham số hiện tại:

| Tham số | Ý nghĩa | Giá trị mặc định |
|---|---|---|
| `--pkl` | Đường dẫn đến file PKL đầu vào từ WHAM | `inputs/datasets/mpi_inf_3dhpe/S8/Seq1/imageSequences/wham_video_0.pkl` |

**Lưu ý:** Đường dẫn tương đối được tính từ thư mục làm việc hiện tại (current working directory). Vì vậy, nên thực thi lệnh từ thư mục gốc `ray_casting/`.

---

## 5. Cài đặt SMPL, Pretrained Weights và Dataset

Các file model, pretrained weights và dataset có dung lượng lớn không được quản lý trực tiếp trong Git repository.

Người sử dụng cần tải và cấu hình các tài nguyên này riêng biệt.

### 5.1. SMPL Model

Dự án sử dụng mô hình SMPL để biểu diễn cơ thể người dưới dạng mô hình tham số 3D.

Trang chủ SMPL:

https://smpl.is.tue.mpg.de/

Các bước thực hiện:

1. Truy cập trang chủ SMPL.
2. Đăng ký tài khoản và chấp nhận các điều khoản sử dụng.
3. Tải model phù hợp.
4. Đặt các file model vào đúng vị trí mà chương trình yêu cầu.

Một số file SMPL và dữ liệu liên quan được sử dụng trong quá trình phát triển:

```text
src/ray_casting/
│
├── SMPL_NEUTRAL.pkl
│
└── models/
    ├── SMPL_NEUTRAL.pkl
    ├── J_regressor_body25_extended.npy
    ├── J_regressor_body25_plus_palm27.npy
    ├── torso_indices.npy
    │
    └── smpl/
        └── J_regressor_body25.npy
```

Lưu ý: Đường dẫn và tên file thực tế phải phù hợp với cấu hình trong mã nguồn.

Các file SMPL được bảo vệ bởi điều khoản cấp phép của nhà cung cấp. Người sử dụng phải tự đảm bảo quyền sử dụng hợp lệ.

### 5.2. Pretrained Weights

Một số pretrained weights được sử dụng hoặc tham chiếu trong quá trình phát triển:

| File | Mô tả |
|---|---|
| `best_ckpt.pth.tar` | Checkpoint của mô hình được sử dụng trong dự án |
| `cpn_101_384x288.pth.tar` | Pretrained weights của mô hình CPN |

Các file này cần được đặt trong:

```text
src/ray_casting/models/
```

Người sử dụng cần tải các weights từ nguồn phát hành tương ứng hoặc liên hệ nhóm phát triển để nhận checkpoint.

Các file pretrained weights không được đính kèm trong Git repository.

### 5.3. Dataset MPI-INF-3DHP

Dự án sử dụng dữ liệu liên quan đến MPI-INF-3DHP.

Trang dataset:

https://vcai.mpi-inf.mpg.de/3dhp-dataset/

Người sử dụng cần:

1. Truy cập trang chính thức của dataset.
2. Đăng ký hoặc chấp nhận các điều khoản sử dụng nếu được yêu cầu.
3. Tải dữ liệu cần thiết.
4. Chuẩn bị các kết quả ước lượng tư thế 3D từ WHAM.

### 5.4. WHAM Pose Estimation Outputs

Repository WHAM:

https://github.com/yohanshin/WHAM

WHAM được sử dụng để tạo ra các kết quả ước lượng tư thế 3D từ video đầu vào.

Các file PKL được tạo ra từ WHAM sẽ được sử dụng làm đầu vào cho pipeline của dự án này.

Ví dụ về cấu trúc dữ liệu:

```text
inputs/
└── datasets/
    └── mpi_inf_3dhpe/
        └── S8/
            └── Seq1/
                └── imageSequences/
                    └── wham_video_0.pkl
```

Sau khi chuẩn bị file PKL, có thể chạy:

```bash
python src/ray_casting/main.py
```

Hoặc chỉ định một file PKL khác bằng `--pkl`.

---

## 6. Quản lý dữ liệu và Git

Để giữ repository gọn nhẹ, các dữ liệu dung lượng lớn không được đưa vào Git.

Các loại tài nguyên cần quản lý bên ngoài repository bao gồm:

- Dataset và các file dữ liệu đầu vào.
- Các file PKL được xuất từ WHAM.
- Pretrained model weights.
- SMPL model files.
- Video và ảnh thực nghiệm.
- Các file kết quả trong quá trình chạy chương trình.

Các tài nguyên này được lưu trữ trên máy local trong các thư mục tương ứng và không được upload lên GitHub.

---

## 7. Tái lập thực nghiệm (Reproducibility)

Để đảm bảo khả năng tái lập kết quả nghiên cứu, cần lưu lại các thông tin sau:

- Dataset và sequence được sử dụng.
- Các thông số calibration của camera.
- Phương pháp monocular HPE và cấu hình tương ứng.
- Phiên bản pretrained weights.
- Các tham số của thuật toán ray-casting.
- Các tham số của quá trình quality gating và pose fusion.
- Các metric và cấu hình đánh giá.

Những thông tin này giúp các nhà nghiên cứu có thể kiểm tra, so sánh và tái lập kết quả thực nghiệm.

---

## 8. License và Citation

Các tài nguyên của bên thứ ba như SMPL, WHAM, MPI-INF-3DHP và pretrained weights có thể có các điều khoản cấp phép riêng.

Người sử dụng cần tuân thủ điều khoản của từng nguồn tài nguyên.

Thông tin về giấy phép sử dụng và trích dẫn công trình nghiên cứu của repository sẽ được cập nhật khi hoàn thiện.

---

**GitHub Repository:**

https://github.com/bkhnk48/ray_casting

**Research Area:**

- 3D Human Pose Estimation
- Multi-View Geometry
- Ray-Casting
- Occlusion Awareness
- Kinematic Quality Assessment
- Multi-View Pose Fusion