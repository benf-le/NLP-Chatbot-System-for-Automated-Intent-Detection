run in local: uvicorn chatbot_api:app --host 0.0.0.0 --port 5000 --reload

Build Docker image: docker build -t chatbot-fastapi .


Run Docker containerP: docker run -p 5000:5000 chatbot-fastapi

