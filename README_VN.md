# Động cơ Theo dõi Hai bàn tay & Thấu kính Filter Camera

[English](README.md) | [Tiếng Việt](README_VN.md)

Ứng dụng theo dõi hai bàn tay thời gian thực bằng OpenCV và MediaPipe Tasks Vision (`HandLandmarker`). Hỗ trợ kích hoạt ngón chụm-kéo riêng biệt, phong cách đường chỉ mảnh tối giản và thấu kính filter camera biến đổi hình ảnh bên trong khối.

---

## ✨ Tính năng chính

- **Theo dõi 2 bàn tay thời gian thực**: Nhận diện 21 mốc tọa độ 3D mỗi bàn tay.
- **Kích hoạt ngón chụm-kéo riêng biệt**: Chỉ nối những cặp ngón chụm chạm nhau ($<55\text{px}$) và kéo ra ($>85\text{px}$).
- **Đường chỉ mảnh tối giản**: Nét chỉ mảnh sắc nét, loại bỏ khung viền nhòe neon và vòng tròn ngón tay lớn.
- **5 Thấu kính Filter Camera**: Biến đổi video bên trong khối hình (`1`: X-Ray âm bản, `2`: Thermal bản đồ nhiệt, `3`: Pixelate 8-bit, `4`: Edge Sketch phác thảo, `5`: Cyberpunk Glitch).
- **Co giãn cửa sổ & Toàn màn hình**: Tùy chỉnh kích thước cửa sổ hoặc bấm `f` để xem Toàn màn hình không bị méo hình.

---

## ⚡ Hướng dẫn cài đặt & Chạy

1. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy ứng dụng**:
   ```bash
   python main.py
   ```

---

## 🎮 Phím tắt điều khiển

| Phím | Chức năng |
| :--- | :--- |
| **`f`** | Bật / Tắt Toàn màn hình (Fullscreen Mode) |
| **`s`** | Bật / Tắt Khung xương bàn tay MediaPipe |
| **`t`** | Chuyển đổi qua lại các hiệu ứng Filter |
| **`1` - `5`** | Chọn trực tiếp chế độ Filter số 1 đến 5 |
| **`d`** | Bật / Tắt bảng chữ thông tin Telemetry |
| **`q`** / **`ESC`** | Thoát ứng dụng |
