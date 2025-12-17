# Bắt đầu với image Python 3.11
FROM python:3.11-slim
LABEL authors="bang8"

# Thiết lập thư mục làm việc
WORKDIR /app

# BẮT BUỘC: Ép buộc sử dụng Legacy Keras để tránh lỗi Keras 3
ENV TF_USE_LEGACY_KERAS=1
# Copy requirements.txt trước để tận dụng Docker layer caching
# Chỉ rebuild dependencies khi requirements.txt thay đổi
COPY requirements.txt /app/

# Cài đặt dependencies (không dùng cache mount)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Tải NLTK data trong build time (thay vì runtime)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('stopwords')"

# Copy toàn bộ code (sau khi đã cài dependencies)
COPY . /app/

# Mở cổng 5000
EXPOSE 5000

# Lệnh để chạy ứng dụng khi container khởi động
CMD ["python", "chatbot_api.py"]