from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import joblib

# โหลดเวกเตอร์ที่แปลงแล้ว
x_train_vectorizer = joblib.load('x_train_vectorizer.pkl')
x_test_vectorizer = joblib.load('x_test_vectorizer.pkl')

# โหลดข้อมูลที่จัดการแล้ว
from NLP import NeuroLinguisticProgramming
NLP = NeuroLinguisticProgramming()

list_fake_news = NLP.Load_data('fake_news.txt')
list_true_news = NLP.Load_data('true_news.txt')

list_fake_news = NLP.Lemmatization(list_fake_news)
list_fake_news = NLP.Remove_Special_character(list_fake_news)
list_fake_news = NLP.Tokenization(list_fake_news)
list_fake_news = NLP.Stopwords(list_fake_news)
list_fake_news = NLP.Lowercase(list_fake_news)

list_true_news = NLP.Lemmatization(list_true_news)
list_true_news = NLP.Remove_Special_character(list_true_news)
list_true_news = NLP.Tokenization(list_true_news)
list_true_news = NLP.Stopwords(list_true_news)
list_true_news = NLP.Lowercase(list_true_news)

# รวมข้อมูล
sentences = list_fake_news + list_true_news
labels = [0] * len(list_fake_news) + [1] * len(list_true_news)

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.1)

# ใช้เวกเตอร์ที่โหลดมาจากไฟล์
X_train_vect = x_train_vectorizer
X_test_vect = x_test_vectorizer


# สร้าง TPOTClassifier และฝึกโมเดล
tpot = TPOTClassifier(generations=5, population_size=20, random_state=42, verbosity=2)
tpot.fit(X_train_vect, y_train)

# ทดสอบโมเดล
accuracy = tpot.score(X_test_vect, y_test)
print(f"Test accuracy: {accuracy}")

# บันทึก pipeline ที่ดีที่สุด
tpot.export('best_model_pipeline.py')
