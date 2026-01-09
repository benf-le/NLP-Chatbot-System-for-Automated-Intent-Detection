import os
import json
import logging
import requests
import numpy as np
import re
import nltk
import joblib
import threading
import time
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import BackgroundTasks
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

# --- NEW: Import GLiNER ---
from gliner import GLiNER

# Tắt XLA để tránh compile chậm lần đầu
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Giảm log TensorFlow

# Hoặc tắt XLA hoàn toàn
os.environ['TF_DISABLE_XLA'] = '1'

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ==============================================================================
# 1. CẤU HÌNH MAPPING INTENT -> ENTITY (CHO GLINER)
# ==============================================================================
# ==============================================================================
# INTENT -> ENTITY SCHEMA (CANONICAL ENTITY TYPES + DESCRIPTIONS)
# ==============================================================================
ORDER_ID_REGEX = re.compile(r"(?<![A-Z0-9])[A-Z0-9]{12}(?![A-Z0-9])", re.IGNORECASE)

INTENT_TO_GLINER_LABELS = {

    # =========================
    # ORDER
    # =========================
    "track_order": {
        "order_id": "a unique identifier for a customer order, often alphanumeric or hex",
        "tracking_number": "a shipment tracking number provided by the carrier",
        "order_date": "the date when the order was placed"
    },

    "cancel_order": {
        "order_id": "a unique identifier for a customer order",
        "product_name": "name of the product in the order",
        "cancel_reason": "reason why the customer wants to cancel the order"
    },

    "change_order": {
        "order_id": "a unique identifier for a customer order",
        "product_name": "name of the product",
        "quantity": "number of items ordered",
        "new_product_name": "new product to replace the original one",
        "color": "color of the product",
        "size": "size of the product"
    },

    "place_order": {
        "product_name": "name of the product the customer wants to buy",
        "quantity": "number of items",
        "payment_method": "credit card, paypal, apple pay, bank transfer",
        "delivery_address": "street address for delivery",
        "delivery_date": "requested delivery date"
    },

    # =========================
    # SHIPPING & DELIVERY
    # =========================
    "delivery_options": {
        "delivery_method": "type of delivery such as standard or express",
        "shipping_cost": "cost of shipping",
        "delivery_time": "estimated delivery duration"
    },

    "delivery_period": {
        "order_id": "a unique identifier for a customer order",
        "delivery_date": "expected delivery date",
        "time_range": "delivery time window"
    },

    "change_shipping_address": {
        "order_id": "a unique identifier for a customer order",
        "street_address": "street name and house number",
        "city": "city name",
        "country": "country name",
        "postal_code": "postal or zip code"
    },

    "set_up_shipping_address": {
        "recipient_name": "full name of the recipient",
        "street_address": "street name and house number",
        "city": "city name",
        "country": "country name",
        "postal_code": "postal or zip code"
    },

    # =========================
    # INVOICE & PAYMENT
    # =========================
    "check_invoice": {
        "invoice_id": "a unique identifier for an invoice",
        "order_id": "a unique identifier for a customer order",
        "invoice_date": "invoice issue date",
        "amount": "total amount of money",
        "currency": "currency such as USD or EUR"
    },

    "get_invoice": {
        "invoice_id": "a unique identifier for an invoice",
        "order_id": "a unique identifier for a customer order",
        "email": "email address to receive the invoice"
    },

    "check_payment_methods": {
        "payment_method": "available payment method such as credit card or paypal",
        "card_type": "visa, mastercard, amex"
    },

    "payment_issue": {
        "payment_method": "credit card, paypal, apple pay",
        "transaction_id": "identifier for a payment transaction",
        "amount": "amount of money involved in the payment",
        "error_message": "payment error description",
        "card_last4": "last 4 digits of a credit card number"
    },

    # =========================
    # REFUND
    # =========================
    "check_refund_policy": {
        "product_name": "name of the product",
        "purchase_date": "date when the product was purchased",
        "product_condition": "condition of the product"
    },

    "get_refund": {
        "order_id": "a unique identifier for a customer order",
        "refund_amount": "amount of money to be refunded",
        "product_name": "name of the product",
        "refund_reason": "reason for requesting a refund"
    },

    "track_refund": {
        "refund_id": "a unique identifier for a refund",
        "order_id": "a unique identifier for a customer order",
        "refund_date": "date when the refund was processed"
    },

    # =========================
    # ACCOUNT
    # =========================
    "create_account": {
        "username": "user account name",
        "email": "email address",
        "date_of_birth": "date of birth",
        "full_name": "full legal name"
    },

    "delete_account": {
        "username": "user account name",
        "email": "email address",
        "phone_number": "phone number",
        "delete_reason": "reason for deleting the account"
    },

    "edit_account": {
        "username": "user account name",
        "email": "email address",
        "phone_number": "phone number",
        "address": "user address",
        "new_value": "new value to update"
    },

    "recover_password": {
        "email": "email address",
        "username": "user account name",
        "phone_number": "phone number"
    },

    "switch_account": {
        "username": "user account name",
        "email": "email address",
        "account_type": "type of account"
    },

    "registration_problems": {
        "error_message": "description of the error",
        "email": "email address",
        "username": "user account name"
    },

    # =========================
    # CONTACT & SUPPORT
    # =========================
    "contact_customer_service": {
        "phone_number": "phone number",
        "email": "email address",
        "preferred_contact_method": "preferred contact method",
        "contact_time": "preferred contact time"
    },

    "contact_human_agent": {
        "phone_number": "phone number",
        "issue_summary": "short summary of the issue",
        "waiting_time": "waiting time mentioned by user"
    },

    # =========================
    # PRODUCT / HEALTH (GIỮ LOGIC CŨ, NHƯNG CHUẨN HÓA)
    # =========================
    "general": {
        "product_name": "name of the product",
        "product_category": "category of the product",
        "brand": "brand name"
    },

    "mushrooms": {
        "mushroom_type": "type of mushroom",
        "benefit": "health benefit",
        "form": "form such as powder or capsule"
    },

    "protein": {
        "protein_type": "type of protein such as whey or vegan",
        "flavor": "product flavor",
        "dietary_preference": "dietary preference such as vegan or keto"
    },

    "vitamins": {
        "vitamin_name": "name of the vitamin",
        "dosage": "recommended dosage",
        "deficiency_symptom": "symptom of deficiency"
    },

    "minerals": {
        "mineral_name": "name of the mineral",
        "dosage": "recommended dosage",
        "health_benefit": "health benefit"
    },

    "amino_acid": {
        "amino_acid_name": "name of the amino acid",
        "function": "function in the body",
        "serving_size": "recommended serving size"
    },

    "amino_acid_dosage": {
        "product_name": "name of the product",
        "daily_dosage": "daily dosage",
        "frequency": "how often to take"
    },

    "amino_acid_child": {
        "product_name": "name of the product",
        "child_age": "age of the child",
        "child_weight": "weight of the child",
        "safety_concern": "safety concern"
    },

    "gut_health": {
        "probiotic_strain": "strain of probiotic",
        "symptom": "digestive symptom",
        "product_name": "name of the product"
    },

    "sleep": {
        "supplement_name": "name of sleep supplement",
        "sleep_issue": "sleep problem such as insomnia",
        "side_effect": "side effect mentioned"
    },

    # =========================
    # DEFAULT (FALLBACK)
    # =========================
    "default": {
        "order_id": "a unique identifier for a customer order",
        "product_name": "name of the product",
        "date": "a calendar date",
        "person": "name of a person",
        "location": "place or location"
    }
}

# ==============================================================================
# 2. LOAD MODEL VÀ DỮ LIỆU
# ==============================================================================
try:
    # Model chính (CNN dự đoán Intent từ Vector 384d)
    cnn_model = load_model('model/cnn_bilstm_model.keras')

    # Model nhúng (Biến text thành Vector 384d)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Kho tri thức đã lưu (embeddings, responses, intents)
    vectors_db = np.load('model/knowledge_embeddings.npy')
    knowledge_data = joblib.load('model/knowledge_data.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')

    # GliNER (Trích xuất thực thể)
    gliner_model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5")

    logger.info("Hệ thống SBERT + CNN + GliNER đã sẵn sàng!")
except Exception as e:
    logger.error(f"Lỗi khởi tạo hệ thống: {str(e)}")
    raise

# Cấu hình Chatwoot
CHATWOOT_BASE_URL = os.environ.get('CHATWOOT_BASE_URL', "https://chatwoot.lecambang.id.vn")
CHATWOOT_API_KEY = os.environ.get('CHATWOOT_API_KEY', "pUcbGjbnAXr3K6kWwwwgk268")
BOT_NAME = os.environ.get('BOT_NAME', 'verifySupp Shop Assistant')

# Mẫu regex cho emoji
emoji_pattern = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]+")  # (Rút gọn cho ngắn code)


class MessageInput(BaseModel):
    message: str


# ==============================================================================
# 3. SERVICE CLASS CHÍNH (ĐÃ CẬP NHẬT)
# ==============================================================================
class ChatwootService:
    def __init__(self):
        self.base_url = CHATWOOT_BASE_URL
        self.api_key = CHATWOOT_API_KEY
        self.headers = {
            "Content-Type": "application/json",
            "api_access_token": self.api_key
        }
        self.conversation_cache = {}


    # --- NEW: HÀM TRÍCH XUẤT ENTITY THÔNG MINH ---
    def extract_entities_optimized(self, user_text, predicted_intent):
        """
        Trích xuất entity theo intent + regex guard + GLiNER with descriptions
        """
        results = []

        # ==============================
        # 1. REGEX GUARD (ORDER ID)
        # ==============================
        for match in ORDER_ID_REGEX.finditer(user_text):
            results.append({
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
                "label": "order_id",
                "score": 1.0,
                "source": "regex"
            })

        # Nếu đã detect order_id bằng regex → bỏ qua GLiNER cho ID
        cleaned_text = user_text
        for r in results:
            cleaned_text = cleaned_text.replace(r["text"], "")

        # ==============================
        # 2. GLiNER ENTITY EXTRACTION
        # ==============================
        if not gliner_model:
            return results

        label_schema = INTENT_TO_GLINER_LABELS.get(
            predicted_intent,
            INTENT_TO_GLINER_LABELS["default"]
        )

        try:
            gliner_entities = gliner_model.predict_entities(
                cleaned_text,
                labels=label_schema,  # <-- LABEL + DESCRIPTION
                threshold=0.3,
                max_length=100
            )

            # Filter: không cho GLiNER override regex order_id
            for e in gliner_entities:
                if e["label"] == "order_id":
                    continue
                e["source"] = "gliner"
                results.append(e)

        except Exception as e:
            logger.error(f"Lỗi GLiNER: {e}")

        return results

    def predict_response(self, user_input):
        """Hàm dự đoán phản hồi sử dụng CNN + SBERT Search"""
        try:
            # 1. Tiền xử lý tối giản (Giữ nguyên cấu trúc câu cho SBERT)
            raw_text = user_input.strip()
            cleaned_text = raw_text.lower()

            # 2. Vector hóa câu hỏi (SBERT)
            user_vec = sbert_model.encode([cleaned_text], show_progress_bar=False)

            # 3. Dự đoán Intent (CNN)
            pred = cnn_model.predict(user_vec, verbose=0)
            intent_idx = np.argmax(pred)
            intent = label_encoder.inverse_transform([intent_idx])[0]
            confidence = float(np.max(pred))

            # 4. Search Vector (Chỉ search trong Intent đã đoán)
            indices_in_intent = np.where(knowledge_data['intents'] == intent)[0]
            sub_vectors = vectors_db[indices_in_intent]

            # Tính tương đồng Cosine
            sims = cosine_similarity(user_vec, sub_vectors)[0]
            best_sub_idx = np.argmax(sims)
            best_global_idx = indices_in_intent[best_sub_idx]
            similarity_score = float(sims[best_sub_idx])

            if similarity_score >= 0.5:
                # Trường hợp 1: Rất giống câu mẫu -> Trả về câu trả lời mẫu chính xác
                best_global_idx = indices_in_intent[best_sub_idx]
                response = knowledge_data['responses'][best_global_idx]
            elif 0.4 <= similarity_score < 0.5:
                # Trường hợp 2: Có vẻ giống nhưng không chắc chắn
                # Thay vì lấy bừa, ta đưa ra câu trả lời mang tính định hướng
                response = f"It seems you are asking about '{intent.replace('_', ' ')}'. Could you please be more specific so I can assist you better?"
            else:
                # Trường hợp 3: Quá khác biệt (Out of Distribution)
                response = "I recognized your intent as related to our services, but I couldn't find a specific answer. Let me connect you with a human agent."
            # 5. Trích xuất thực thể (GliNER)
            entities = self.extract_entities_optimized(raw_text, intent)

            return {
                "intent": intent,
                "response": response,
                "confidence": confidence,
                "similarity_score": similarity_score,
                "entities": entities
            }

        except Exception as e:
            logger.error(f"Lỗi Inference: {str(e)}")
            return {"intent": "error", "response": "System error.", "confidence": 0, "entities": []}

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

            if conversation_status == 'open' and not self.is_assigned_to_bot(account_id, conversation_id):
                return {"status": "ignored", "reason": "Conversation assigned to human agent"}

            # Dự đoán câu trả lời
            prediction = self.predict_response(message_content)
            intent = prediction.get('intent')
            confidence = prediction.get('confidence')
            response = prediction.get('response')
            similarity_score = prediction.get('similarity_score')
            entities = prediction.get('entities')  # Lấy entity

            # Lưu context
            self.update_conversation_context(conversation_id, message_content, intent)

            # Tạo JSON response với predict + entities
            json_response = {
                "intent": intent,
                "response": response,
                "confidence": float(confidence),
                "entities": entities,
                "similarity_score": similarity_score,
                "original_message": message_content
            }

            print(json_response)

            # Gửi JSON về Chatwoot (dạng string JSON)
            json_message = json.dumps(json_response, ensure_ascii=False, indent=2)
            self.send_message(account_id, conversation_id, json_message)

            return {
                "status": "success",
                "intent": intent,
                "confidence": confidence,
                "entities": entities
            }

            # self.send_message(account_id, conversation_id, response)

            # return {
            #     "status": "success",
            #     "intent": intent,
            #     "confidence": confidence,
            #     "entities": entities
            # }

        except Exception as e:
            logger.error(f"Lỗi khi xử lý tin nhắn: {str(e)}")
            return {"status": "error", "error": str(e)}

    # ... (Các hàm support cũ giữ nguyên: update_conversation_context, should_handover, is_assigned, send_message...)
    def update_conversation_context(self, conversation_id, message, intent):
        if conversation_id not in self.conversation_cache:
            self.conversation_cache[conversation_id] = {
                "messages": [], "intents": [], "last_update": time.time()
            }
        self.conversation_cache[conversation_id]["messages"].append(message)
        self.conversation_cache[conversation_id]["intents"].append(intent)
        self.conversation_cache[conversation_id]["last_update"] = time.time()
        if len(self.conversation_cache[conversation_id]["messages"]) > 10:
            self.conversation_cache[conversation_id]["messages"].pop(0)
            self.conversation_cache[conversation_id]["intents"].pop(0)

    def is_assigned_to_bot(self, account_id, conversation_id):
        url = f"{self.base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                assignee_type = data.get('meta', {}).get('assignee', {}).get('type')
                return assignee_type == 'bot' or assignee_type is None
            return True
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra assignee: {str(e)}")
            return True

    def send_message(self, account_id, conversation_id, message):
        url = f"{self.base_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
        payload = {"content": message, "message_type": "outgoing", "private": False}
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            return response.json() if response.status_code in [200, 201] else None
        except Exception as e:
            logger.error(f"Lỗi khi gửi tin nhắn: {str(e)}")
            return None

    def clean_old_conversations(self):
        current_time = time.time()
        to_remove = []
        for conv_id, data in self.conversation_cache.items():
            if current_time - data["last_update"] > 7200:
                to_remove.append(conv_id)
        for conv_id in to_remove:
            del self.conversation_cache[conv_id]
            logger.info(f"Đã xóa cuộc hội thoại {conv_id} khỏi cache")


# Khởi tạo service
chatwoot_service = ChatwootService()


# Hàm dọn dẹp định kỳ
def cleanup_task():
    while True:
        chatwoot_service.clean_old_conversations()
        time.sleep(3600)


cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
cleanup_thread.start()


@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Service is running"}


@app.post("/api/predict")
async def predict(message_input: MessageInput):
    """API dự đoán có trả về Entity"""
    if not message_input.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    result = chatwoot_service.predict_response(message_input.message)
    return result


# Thêm executor cho CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=2)


@app.post("/webhook")
async def chatwoot_webhook(request: Request, background_tasks: BackgroundTasks):
    """Webhook nhận tin nhắn từ Chatwoot"""
    try:
        data = await request.json()
        logger.info(f"Received webhook: {data.get('event')}")

        # Trả response ngay, xử lý ở background
        background_tasks.add_task(process_webhook_async, data)

        return {"status": "accepted", "message": "Processing in background"}
    except Exception as e:
        logger.error(f"Lỗi khi xử lý webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def process_webhook_async(data):
    """Xử lý webhook trong background"""
    try:
        result = chatwoot_service.handle_message(data)
        logger.info(f"Webhook processed: {result.get('status')}")
    except Exception as e:
        logger.error(f"Lỗi xử lý webhook background: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)