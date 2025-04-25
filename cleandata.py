import re
import pandas as pd

# from pyvi import ViUtils

# Hàm tiền xử lý
def text_lowercase(text):
    return text.lower()

def remove_number(text):
    result = re.sub(r'\d+', '', text)
    return result



def remove_punctuation(text):
    text = text.replace(",", " ").replace(".", " ") \
               .replace(";", " ").replace("“", " ") \
               .replace(":", " ").replace("”", " ") \
               .replace('"', " ").replace("'", " ") \
               .replace("!", " ").replace("?", " ") \
               .replace("-", " ").replace("=", " ") \
               .replace(")", " ").replace("(", " ") \
               .replace("~", " ").replace("!", " ") \
               .replace("@", " ").replace("#", " ") \
               .replace("$", " ").replace("%", " ") \
               .replace("^", " ").replace("&", " ") \
               .replace("*", " ").replace("_", " ") \
               .replace("+", " ").replace("=", " ") \
               .replace("{", " ").replace("}", " ") \
               .replace("[", " ").replace("]", " ") \
               .replace(":", " ").replace(";", " ") \
               .replace('"', " ").replace("'", " ") \
               .replace("<", " ").replace(">", " ") \
               .replace(",", " ").replace(".", " ") \
               .replace("/", " ").replace("?", " ")
                 # Có thể thêm dấu cách ở cuối nếu muốn
    return text

def remove_whitespace(text):
    return " ".join(text.split())

def remove_similarletter(text):
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
    return text





emoji_pattern = re.compile("[" 
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u'\U00010000-\U0010ffff'
    u"\u200d"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\u3030"
    u"\ufe0f"
"]+", flags=re.UNICODE)

