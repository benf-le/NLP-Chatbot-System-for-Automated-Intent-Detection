# ⚡ api_chatbot

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/benf-le/api_chatbot?style=for-the-badge)](https://github.com/benf-le/api_chatbot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/benf-le/api_chatbot?style=for-the-badge)](https://github.com/benf-le/api_chatbot/network)
[![GitHub issues](https://img.shields.io/github/issues/benf-le/api_chatbot?style=for-the-badge)](https://github.com/benf-le/api_chatbot/issues)
[![GitHub license](https://img.shields.io/github/license/benf-le/api_chatbot?style=for-the-badge)](LICENSE)

**A Python-based chatbot API built using a pre-trained model and a cleaned customer support dataset.**

</div>

## 📖 Overview

This project implements a RESTful API for a chatbot powered by a pre-trained language model.  It utilizes a cleaned dataset of customer support interactions (`Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`) to improve the chatbot's responses.  The API is containerized using Docker for easy deployment. The project uses Jupyter Notebooks for model training and experimentation.

## ✨ Features

- **RESTful API:**  Provides a simple interface for interacting with the chatbot.
- **Pre-trained Model:** Leverages an existing language model (specific model needs to be identified from `chatbot-ver-5.ipynb`).
- **Cleaned Dataset:** Employs a processed dataset for improved chatbot performance.
- **Docker Containerization:** Enables easy deployment and portability.

## 🛠️ Tech Stack

**Backend:**
- [Python](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/) (inferred from common Python API structures, needs verification)

**Data Processing:**
- [Pandas](https://pandas.pydata.org/) (inferred from common data manipulation in Jupyter notebooks)

**Model Training (Notebooks):**
- [Jupyter Notebook](https://jupyter.org/)
- [Specific ML Libraries](TODO: Identify from `chatbot-ver-5.ipynb`)


**Containerization:**
- [Docker](https://www.docker.com/)

## 🚀 Quick Start

### Prerequisites

- Python 3.x (version specified in `requirements.txt`)
- Docker

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/benf-le/api_chatbot.git
   cd api_chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the Docker image:**
   ```bash
   docker build -t chatbot-fastapi .
   ```

4. **Run the Docker container:**
   ```bash
   docker run -p 5000:5000 chatbot-fastapi
   ```
4. **Run in Local:**
   ```bash
   uvicorn chatbot_api:app --host 0.0.0.0 --port 5000 --reload
   ```
## 📁 Project Structure

```
api_chatbot/
├── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
├── Dockerfile
├── Readme.md
├── __pycache__/
│   └── ...
├── chatbot-ver-5.ipynb
├── chatbot_api.py
├── cleandata.py
└── model/
    └── ...

```

## ⚙️ Configuration

No explicit configuration files are detected, but environment variables might be used within the `chatbot_api.py` (needs investigation).


## 🧪 Testing

No dedicated test suite is found.  Testing should be added for production readiness.  (TODO: Add testing)


## 🚀 Deployment

The Dockerfile provides a simple deployment mechanism.  Deploy to any Docker-compatible environment.


## 📚 API Reference (Partial)

The API endpoints need further investigation within `chatbot_api.py`. (TODO:  Detailed API documentation from code analysis)


## 🤝 Contributing

Contributions are welcome!  Please open an issue or submit a pull request. (TODO:  Add detailed contribution guidelines)


## 📄 License

This project is under the (TODO: Specify License).


---

<div align="center">

**⭐ Star this repo if you find it helpful!**

</div>



