# Bắt đầu với image Python 3.11
FROM python:3.11-slim
LABEL authors="bang8"

# Cài đặt system dependencies cho TensorFlow và các thư viện khác
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements.txt trước để tận dụng Docker layer caching
COPY requirements.txt /app/

# Cài đặt dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Cài đặt sentence-transformers nếu chưa có trong requirements.txt
RUN pip install --no-cache-dir sentence-transformers

# Tải NLTK data trong build time
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('stopwords')"

# Pre-download SentenceTransformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Pre-download GLiNER model
RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('gliner-community/gliner_large-v2.5')"

# Copy model files (tách riêng để cache tốt hơn)
COPY model/ /app/model/

# Copy toàn bộ code (sau khi đã cài dependencies và models)
COPY . /app/

# Mở cổng 5000
EXPOSE 5000

# Lệnh để chạy ứng dụng khi container khởi động
CMD ["python", "chatbot_api.py"]