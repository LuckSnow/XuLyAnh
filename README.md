# Lumière Lab

Ứng dụng desktop Python cho đồ án Xử lý ảnh và Video. Mục **Ứng Dụng Duy Nhất**
hiện chứa **PhotoFix — Phục Hồi & Tô Màu**, một quy trình cải thiện ảnh cũ bằng
OpenCV và tô màu ảnh đen trắng bằng DDColor Tiny chạy local trên CPU. Giao diện
CustomTkinter hiện có được giữ nguyên.

## Phần thuật toán đã học

- **Biến đổi:** tuyến tính sáng/tương phản, ảnh xám, âm bản, logarit, gamma,
  kéo giãn mức xám.
- **Histogram:** cân bằng toàn cục và CLAHE.
- **Bộ lọc:** trung bình, Gaussian, trung vị, bilateral, làm sắc nét.
- **Hình thái:** co, giãn, mở, đóng.
- **Phát biên:** Roberts, Prewitt, Sobel, Canny.

Các nhóm này là nơi thử các phép xử lý riêng lẻ. Giao diện vẫn hỗ trợ mở/xuất
ảnh, xem trước, histogram, phóng to, hoàn tác và giữ `Space` để so sánh trước/sau.
Phần lý thuyết đối chiếu với các PDF trong `D:\Xu Li Anh Va Video\curriculum`
và mã ví dụ trong `D:\Xu Li Anh Va Video`.

## Ứng dụng duy nhất: PhotoFix

[single_application.py](single_application.py) đăng ký đúng một công cụ vào
giao diện; [photo_fix.py](photo_fix.py) chứa pipeline và mô hình AI. Chọn công
cụ trong mục **Ứng Dụng Duy Nhất**, rồi chọn chế độ:

- `0` — phục hồi ảnh cũ: vá đốm/xước sáng nhỏ bằng inpainting, giảm nhiễu,
  cải thiện ảnh tối/tương phản thấp khi cần và làm sắc vừa phải.
- `1` — chỉ tô màu bằng DDColor. Đây là yêu cầu tô màu tường minh, kể cả với
  ảnh đã có màu.
- `2` — phục hồi rồi tô màu nếu ảnh gốc là **ảnh xám hoặc ảnh sepia đơn sắc**.
  Ảnh màu thông thường được cải thiện nhưng không tự chuyển sang xám để tô lại.

Các thanh `0/1` cho phép bật/tắt vá đốm, sửa sáng, CLAHE, giảm nhiễu và khử
ám nâu của ảnh sepia. Ảnh `enhanced.png` là bản trung gian đen trắng sau khử
ám nâu; `final.png` là bản đã tô màu nếu chế độ có chạy DDColor.
Thanh làm sắc nét và độ đậm màu AI điều chỉnh cường độ. Chế độ phục hồi không
cần mô hình; hai chế độ tô màu cần `models/ddcolor_paper_tiny_w16.onnx` (~115 MB).
File này đã có trong dự án. Trên máy khác, app có thể tải khi dùng lần đầu,
hoặc tải trước bằng:

```powershell
python photo_fix.py --download-model
```

Mô hình chạy bằng `CPUExecutionProvider`, dùng lại session giữa các lượt xử lý
và không gửi ảnh của người dùng lên dịch vụ ngoài. Mỗi lượt áp dụng trong app
tạo ảnh `original`, `repaired`, `enhanced`, `colorized` (nếu có), `final` và
`report.json` trong `outputs/`. Bạn vẫn có thể dùng `Ctrl+S` để lưu ảnh cuối
theo đường dẫn mong muốn. Muốn xử lý từ dòng lệnh và có mask trắng đánh dấu
vùng hư hại:

```powershell
python photo_fix.py --input "anh_cu.jpg" --mode 2 --mask "mask_xuoc.png"
```

Vết rách lớn và phần ảnh mất hoàn toàn cần mask hoặc mô hình chuyên dụng;
inpainting không khôi phục chính xác nội dung đã mất. Màu do AI dự đoán, có thể
khác màu thật. Mô hình DDColor Tiny ONNX được lấy từ
[OpenPixels](https://huggingface.co/openpixels-app/browser-models) dựa trên
[DDColor](https://github.com/piddnad/DDColor); khi tải có kiểm tra SHA-256.
Giấy phép Apache-2.0 nằm tại `models/LICENSE-DDColor.txt`.

## Chạy ứng dụng

Nhấp đúp [run_app.bat](run_app.bat) trên Windows, hoặc chạy:

```powershell
cd D:\PrXLA
python -m pip install -r requirements.txt
python main.py
```

## Trình bày hiệu quả bộ lọc

Mở cùng một ảnh trong app trước mỗi phép lọc. Chỉnh tham số, giữ `Space` để
so sánh, rồi xuất kết quả. Có thể tạo bảng đối chiếu tự động bằng đúng hàm
lọc của app:

```powershell
python compare_filters.py --input "D:\Xu Li Anh Va Video\src\phunu.jpg" --output-dir filter_comparisons
```

- [Nhiễu Gaussian](filter_comparisons/so_sanh_nhieu_gaussian.png)
- [Nhiễu muối tiêu](filter_comparisons/so_sanh_nhieu_muoi_tieu.png)
- [Ảnh mờ và làm sắc nét](filter_comparisons/so_sanh_lam_net.png)

Các bảng dùng ảnh sạch làm tham chiếu, tạo hư hại giả lập với seed cố định và
đo PSNR. Ảnh thực không có bản sạch thì cần đánh giá thêm bằng mắt ở vùng phẳng,
đường biên và chi tiết nhỏ.
