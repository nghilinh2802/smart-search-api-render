# Smart Search API (Render Version)

## ✅ Hướng dẫn triển khai lên Render (Python 3.10)

### Bước 1: Chuẩn bị Firebase
- Vào Firebase Console → Project Settings → Service Accounts
- Generate private key → Tải JSON → mở bằng Notepad
- Copy toàn bộ nội dung JSON

### Bước 2: Tạo GitHub repo
- Tạo repo mới
- Copy tất cả file từ thư mục này vào repo của bạn
- Thêm file `main.py` bạn đã có sẵn vào cùng repo
- Push lên GitHub

### Bước 3: Deploy lên Render
- Vào https://render.com → New Web Service
- Chọn repo bạn vừa tạo
- Chọn:
  - Runtime: Python 3.10
  - Build command: pip install -r requirements.txt
  - Start command: python main.py
- Vào tab Environment → Thêm:
  - `FIREBASE_CREDENTIALS` = dán nội dung JSON Firebase key

### Bước 4: Deploy và test API

```bash
curl https://your-app-name.onrender.com/search -X POST -H "Content-Type: application/json" -d '{"query": "cat food under 100000"}'
```
