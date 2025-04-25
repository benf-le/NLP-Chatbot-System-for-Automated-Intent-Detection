# Bắt đầu với image Python 3.9
FROM python:3.11-slim
LABEL authors="bang8"
# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép tất cả các file vào thư mục làm việc
COPY . /app/

# Cài đặt các dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Mở cổng 5000
EXPOSE 5000

# Lệnh để chạy ứng dụng khi container khởi động
CMD ["python", "chatbot_api.py"]
