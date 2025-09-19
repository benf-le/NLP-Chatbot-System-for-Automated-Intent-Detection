import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import random
import re
import nltk
import joblib
import threading
import time
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = FastAPI()

# Tải model và dữ liệu
try:
    # Tải mô hình và tokenizer
    # model = load_model('model/bilstm_model.h5')
    model = load_model('model/cnn_model.keras')

    tokenizer = joblib.load('model/tokenizer.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')

    # Tải dataset response
    df = pd.read_csv('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')

    # Thông số padding
    MAX_LEN = 100  # Thay bằng max_len đã dùng trong quá trình train

    # Thiết lập NLTK
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    logger.info("Model và dữ liệu đã được tải thành công")
except Exception as e:
    logger.error(f"Lỗi khi tải model: {str(e)}")
    raise

# Cấu hình Chatwoot - wweb Chatwoot
# CHATWOOT_BASE_URL = os.environ.get('CHATWOOT_BASE_URL', 'https://app.chatwoot.com')
# CHATWOOT_API_KEY = os.environ.get('CHATWOOT_API_KEY', '1MStthEsbxeHZjtBHm12gQN6')
# BOT_NAME = os.environ.get('BOT_NAME', 'Pet Shop Assistant')

# Cấu hình Chatwoot - web tự build
# CHATWOOT_BASE_URL = os.environ.get('CHATWOOT_BASE_URL', 'http://34.46.179.242:3000/')
# CHATWOOT_API_KEY = os.environ.get('CHATWOOT_API_KEY', 'U3uX6spJGaJe5g5CCBCATw6R')
# BOT_NAME = os.environ.get('BOT_NAME', 'Pet Shop Assistant')

# # Cấu hình Chatwoot - web verifySupp
CHATWOOT_BASE_URL = os.environ.get('CHATWOOT_BASE_URL',"http://34.55.138.114:3000/")
CHATWOOT_API_KEY = os.environ.get('CHATWOOT_BASE_URL',"jiViGpFB62PjfUNNbfPpcMwD")
BOT_NAME = os.environ.get('BOT_NAME', 'verifySupp Shop Assistant')
# Mẫu regex cho emoji
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "]+"
)


# Hàm xử lý văn bản
def remove_number(text):
    """Xóa số từ văn bản"""
    return re.sub(r'\d+', '', text)


def remove_punctuation(text):
    """Xóa dấu câu từ văn bản"""
    return re.sub(r'[^\w\s]', ' ', text)


def remove_whitespace(text):
    """Xóa khoảng trắng thừa"""
    return re.sub(r'\s+', ' ', text).strip()


def remove_similarletter(text):
    """Xóa các chữ cái tương tự lặp lại"""
    return re.sub(r'([a-z])\1{2,}', r'\1', text)


class MessageInput(BaseModel):
    message: str


class ChatwootService:
    def __init__(self):
        self.base_url = CHATWOOT_BASE_URL
        self.api_key = CHATWOOT_API_KEY
        self.headers = {
            "Content-Type": "application/json",
            "api_access_token": self.api_key
        }
        self.conversation_cache = {}  # Cache để lưu context của các cuộc hội thoại


        # --- Chuẩn bị mapping intent -> (instructions, responses) ---
        self.intent_to_qa = {}
        for q, intent, ans in zip(df['instruction'], df['intent'], df['response']):
            if intent not in self.intent_to_qa:
                self.intent_to_qa[intent] = {"instruction": [], "response": []}
            self.intent_to_qa[intent]["instruction"].append(q)
            self.intent_to_qa[intent]["response"].append(ans)

        # --- Encode toàn bộ câu hỏi trong mỗi intent ---
        self.vectorizers = {}
        self.tfidf_matrices = {}
        for intent, qa in self.intent_to_qa.items():
            vec = TfidfVectorizer()
            tfidf = vec.fit_transform(qa["instruction"])
            self.vectorizers[intent] = vec
            self.tfidf_matrices[intent] = tfidf

    def predict_response(self, user_input):
        """Hàm dự đoán phản hồi từ mô hình"""
        try:
            # Tiền xử lý văn bản đầu vào
            user_input = remove_number(user_input)
            user_input = remove_punctuation(user_input)
            user_input = remove_whitespace(user_input)
            user_input = remove_similarletter(user_input)
            user_input = re.sub(emoji_pattern, " ", user_input)

            tokens = nltk.word_tokenize(user_input.lower())
            filtered_tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
            cleaned_text = ' '.join(filtered_tokens)

            seq = tokenizer.texts_to_sequences([cleaned_text])
            padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

            pred = model.predict(padded_seq)
            intent = label_encoder.inverse_transform([np.argmax(pred)])[0]
            confidence = np.max(pred)

            responses = df[df['intent'] == intent]['response'].tolist()
            if responses:
                # ---- TF-IDF retrieval trong intent ----
                vec = self.vectorizers[intent].transform([cleaned_text])
                sims = cosine_similarity(vec, self.tfidf_matrices[intent])
                idx = sims.argmax()
                response = self.intent_to_qa[intent]["response"][idx]
            else:
                response = "I'm not sure how to respond to that."

            return {
                "intent": intent,
                "response": response,
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán phản hồi: {str(e)}")
            return {
                "intent": "error",
                "response": "Sorry, there was an error processing your request.",
                "confidence": 0.0
            }

    def handle_message(self, data):
        """Xử lý tin nhắn từ Chatwoot webhook"""
        try:
            # Kiểm tra loại sự kiện
            event_type = data.get('event')
            if event_type != 'message_created':
                return {"status": "ignored", "reason": "Not a message event"}

            # Kiểm tra loại tin nhắn
            message_type = data.get('message_type')
            sender_type = data.get('sender', {}).get('type')

            if message_type != 'incoming' or sender_type == 'bot':
                return {"status": "ignored", "reason": "Not an incoming user message"}

            # Lấy thông tin tin nhắn
            message_content = data.get('content', '')
            conversation_id = data.get('conversation', {}).get('id')
            account_id = data.get('account', {}).get('id')

            # Kiểm tra trạng thái cuộc hội thoại
            conversation_status = data.get('conversation', {}).get('status')

            # Nếu cuộc hội thoại đã được gán cho agent (không phải bot), không xử lý
            if conversation_status == 'open' and not self.is_assigned_to_bot(account_id, conversation_id):
                return {"status": "ignored", "reason": "Conversation assigned to human agent"}

            # Dự đoán câu trả lời
            prediction = self.predict_response(message_content)
            intent = prediction.get('intent')
            confidence = prediction.get('confidence')
            response = prediction.get('response')

            # Lưu context của cuộc hội thoại
            self.update_conversation_context(conversation_id, message_content, intent)

            # # Kiểm tra xem có cần chuyển giao cho agent không
            # if self.should_handover_to_agent(intent, confidence, conversation_id):
            #     # Gửi thông báo chuyển giao
            #     self.send_message(account_id, conversation_id,
            #                       "I'll connect you with a customer service representative right away.")
            #     # Chuyển cuộc hội thoại cho agent
            #     self.assign_conversation(account_id, conversation_id)
            #     return {"status": "handover", "reason": f"Low confidence ({confidence}) or handover intent"}

            # Gửi câu trả lời
            self.send_message(account_id, conversation_id, response)

            return {
                "status": "success",
                "intent": intent,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Lỗi khi xử lý tin nhắn: {str(e)}")
            return {"status": "error", "error": str(e)}

    def update_conversation_context(self, conversation_id, message, intent):
        """Cập nhật context của cuộc hội thoại"""
        if conversation_id not in self.conversation_cache:
            self.conversation_cache[conversation_id] = {
                "messages": [],
                "intents": [],
                "last_update": time.time()
            }

        # Thêm tin nhắn và intent vào context
        self.conversation_cache[conversation_id]["messages"].append(message)
        self.conversation_cache[conversation_id]["intents"].append(intent)
        self.conversation_cache[conversation_id]["last_update"] = time.time()

        # Giới hạn số lượng tin nhắn lưu trong context
        if len(self.conversation_cache[conversation_id]["messages"]) > 10:
            self.conversation_cache[conversation_id]["messages"].pop(0)
            self.conversation_cache[conversation_id]["intents"].pop(0)

    def should_handover_to_agent(self, intent, confidence, conversation_id):
        """Kiểm tra xem có cần chuyển giao cho agent không"""
        # Các intent cần chuyển giao
        handover_intents = ['human_agent', 'talk_to_human', 'live_agent', 'need_help', 'complaint']

        # Nếu intent thuộc danh sách cần chuyển giao
        if intent in handover_intents:
            return True

        # Nếu độ tin cậy quá thấp
        if confidence < 0.6:
            return True

        # Nếu đã có nhiều câu hỏi liên tiếp không hiểu hoặc lỗi
        if conversation_id in self.conversation_cache:
            recent_intents = self.conversation_cache[conversation_id]["intents"][-3:]
            # Nếu có 2/3 tin nhắn gần nhất là unknown hoặc error
            if len(recent_intents) >= 3 and recent_intents.count("unknown") + recent_intents.count("error") >= 2:
                return True

        return False

    def is_assigned_to_bot(self, account_id, conversation_id):
        """Kiểm tra xem cuộc hội thoại có được gán cho bot không"""
        url = f"{self.base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                assignee_type = data.get('meta', {}).get('assignee', {}).get('type')
                return assignee_type == 'bot' or assignee_type is None
            return True  # Mặc định xử lý nếu không kiểm tra được
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra assignee: {str(e)}")
            return True  # Mặc định xử lý

    def send_message(self, account_id, conversation_id, message):
        """Gửi tin nhắn tới Chatwoot"""
        url = f"{self.base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
        payload = {
            "content": message,
            "message_type": "outgoing",
            "private": False
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            if response.status_code not in [200, 201]:
                logger.error(f"Lỗi khi gửi tin nhắn: {response.text}")
            return response.json() if response.status_code in [200, 201] else None
        except Exception as e:
            logger.error(f"Lỗi khi gửi tin nhắn: {str(e)}")
            return None

    def assign_conversation(self, account_id, conversation_id, agent_id=None):
        """Chuyển cuộc hội thoại cho agent hoặc vào hàng đợi"""
        url = f"{self.base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}/assignments"
        payload = {}
        if agent_id:
            payload["assignee_id"] = agent_id

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            if response.status_code not in [200, 201]:
                logger.error(f"Lỗi khi chuyển cuộc hội thoại: {response.text}")
            return response.json() if response.status_code in [200, 201] else None
        except Exception as e:
            logger.error(f"Lỗi khi chuyển cuộc hội thoại: {str(e)}")
            return None

    def clean_old_conversations(self):
        """Dọn dẹp các cuộc hội thoại cũ trong cache"""
        current_time = time.time()
        to_remove = []

        for conv_id, data in self.conversation_cache.items():
            # Xóa các cuộc hội thoại không hoạt động quá 2 giờ
            if current_time - data["last_update"] > 7200:  # 2 giờ = 7200 giây
                to_remove.append(conv_id)

        for conv_id in to_remove:
            del self.conversation_cache[conv_id]
            logger.info(f"Đã xóa cuộc hội thoại {conv_id} khỏi cache do không hoạt động")


# Khởi tạo service
chatwoot_service = ChatwootService()


# Hàm dọn dẹp định kỳ
def cleanup_task():
    while True:
        chatwoot_service.clean_old_conversations()
        time.sleep(3600)  # Chạy mỗi giờ


# Khởi động thread dọn dẹp
cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
cleanup_thread.start()


@app.get("/health")
def health_check():
    """Endpoint kiểm tra trạng thái service"""
    return {"status": "healthy", "message": "Service is running"}


@app.post("/api/predict")
async def predict(message_input: MessageInput):
    """API dự đoán đơn giản cho mục đích kiểm tra"""
    if not message_input.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = chatwoot_service.predict_response(message_input.message)
    return result


@app.post("/webhook")
async def chatwoot_webhook(request: Request):
    """Webhook nhận tin nhắn từ Chatwoot"""
    try:
        data = await request.json()
        logger.info(f"Received webhook: {data.get('event')}")

        # Xử lý tin nhắn
        result = chatwoot_service.handle_message(data)
        return result
    except Exception as e:
        logger.error(f"Lỗi khi xử lý webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
