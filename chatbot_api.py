from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import random
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import joblib
import pandas as pd
# Stopwords và Lemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from cleandata import *
# Cấu hình FastAPI
app = FastAPI()

# Load các thành phần đã lưu từ mô hình đã huấn luyện
model = load_model('model/bilstm_model.h5')  # mô hình CNN

tokenizer = joblib.load('model/tokenizer.pkl')

label_encoder = joblib.load('model/label_encoder.pkl')

df = pd.read_csv('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')

# Thông số padding
MAX_LEN = 100  # Bạn cần thay bằng max_len đã dùng trong quá trình train


lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Dữ liệu đầu vào từ Webhook
class MessageInput(BaseModel):
    message: str

# Hàm trả lời
def chatbot_response(user_input):
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

    responses = df[df['intent'] == intent]['response'].tolist()
    if responses:
        return random.choice(responses)
    else:
        return "I'm not sure how to respond to that."

@app.get("/")
async def root():
    return {"message": "Chatbot API is running!"}

# Endpoint chính
@app.post("/chat")
async def chat(input: MessageInput):
    reply = chatbot_response(input.message)
    return {"response": reply}
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 5000))  # Render cấp PORT, local thì dùng 5000
    uvicorn.run("chatbot_api:app", host="0.0.0.0", port=port)

