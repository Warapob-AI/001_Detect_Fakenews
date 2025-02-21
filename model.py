import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

# โหลดข้อมูล
def load_data(filename): 
    with open(filename, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

# ตัดคำ และลบคำที่ไม่จำเป็น
def preprocess_text(data_list): 
    final = "".join(data for data in data_list)
    final = word_tokenize(final, keep_whitespace=False)
    final = " ".join(word for word in final if word not in ('"\'', '"‘', '\'"', '[', ']', '!"', '!!', "'",'\u200b', '|', '+', '#', '‘', '’', '...', '..', "ๆ", "“", "”", "?", ".", ";", ":", "!", '"', "(", ")", ",", "«", "»", "—", "-", "–", "…", "№", " ", "\n", "\t", "\r", "\ufeff", ">", "<", 'ฯ','.', '/'))
    final = " ".join(word for word in final.split() if word not in thai_stopwords())
    return final

# โหลดข้อความจริงและปลอม
true_news = load_data('true_news.txt')
fake_news = load_data('fake_news.txt')

list_true_news = [preprocess_text(data) for data in true_news]
list_fake_news = [preprocess_text(data) for data in fake_news]

# เตรียมข้อมูล
sentences = list_true_news + list_fake_news
labels = [0] * len(list_true_news) + [1] * len(list_fake_news)  # 0 = ข่าวจริง, 1 = ข่าวปลอม

# แบ่งข้อมูล train-test
x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=42)

# โหลด BERT Tokenizer และ BERT Model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

# ฟังก์ชันแปลงข้อความเป็นเวกเตอร์ด้วย BERT
def get_bert_embedding(texts, tokenizer, model, max_length=128):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # ใช้เวกเตอร์จาก [CLS] token

# แปลงข้อความเป็นเวกเตอร์
x_train_vectors = get_bert_embedding(x_train, tokenizer, bert_model)
x_test_vectors = get_bert_embedding(x_test, tokenizer, bert_model)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train_vectors, y_train)

# ทำนายผล
y_pred = rf_model.predict(x_test_vectors)

# ประเมินผล
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
