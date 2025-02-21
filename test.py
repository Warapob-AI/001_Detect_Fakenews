from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

def load_data(filename): 
    import codecs

    with codecs.open(filename, 'r', 'utf-8') as file: 
        return [line.strip() for line in file.readlines()]

def cut_word(data_list): 
    final = "".join(data for data in data_list)
    final = word_tokenize(final, keep_whitespace=False)
    final = " ".join(word for word in final if word not in ('"\'', '"‘', '\'"', '[', ']', '!"', '!!', "'",'\u200b', '|', '+', '#', '‘', '’', '...', '..', "ๆ", "“", "”", "?", ".", ";", ":", "!", '"', "(", ")", ",", "«", "»", "—", "-", "–", "…", "№", " ", "\n", "\t", "\r", "\ufeff", ">", "<", 'ฯ','.', '/'))

    return final

true_news = load_data('true_news.txt')
fake_news = load_data('fake_news.txt')

list_true_news = [cut_word(data) for data in true_news]
list_fake_news = [cut_word(data) for data in fake_news]

sentences = list_true_news + list_fake_news
labels = ['ข่าวจริง'] * len(list_true_news) + ['ข่าวปลอม'] * len(list_fake_news)

x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size = 0.25, random_state = 42)
    
tf = TfidfVectorizer(ngram_range = (1, 2))

x_train_tf = tf.fit_transform(x_train)
x_test_tf = tf.transform(x_test)

model = LogisticRegression(random_state=42)
model.fit(x_train_tf, y_train)

y_pred = model.predict(x_test_tf)

accuracy = accuracy_score(y_test, y_pred)
print(f"ความแม่นยำ : {accuracy * 100:.2f}")

classification = classification_report(y_test, y_pred)
print(classification)

while True: 
    # list_fake_test = load_data(r"fake_test.txt")
    # result = []
    # for data in list_fake_test: 
    #     text_cut_word = cut_word(data)
    #     sentence_tfidf = tf.transform([text_cut_word])
    #     sentence_prediction = model.predict(sentence_tfidf)
    #     sentence_accuracy = max(model.predict_proba(sentence_tfidf)[0])
    
    #     if (sentence_prediction[0] == 'ข่าวจริง'):
    #         result.append(f"ข้อความ : {data}, ผลการพยากรณ์ : {'ข่าวจริง'}")
    #         print(f"ข้อความ : {data}, ผลการพยากรณ์ : {'ข่าวจริง'}")
    # print(f"จากข่าวปลอม {len(list_fake_test)} ข่าว -- โมเดล logistic regression : ทายผิดว่าเป็นข่าวจริง : {len(result)}")

    sentence = str(input('\nEnter Sentence : '))

    sentence_cut_word = cut_word(sentence)
    print(sentence_cut_word.split())

    sentence_tfidf = tf.transform([sentence_cut_word])
    print(sentence_tfidf)

    sentence_prediction = model.predict(sentence_tfidf)

    sentence_accuracy = max(model.predict_proba(sentence_tfidf)[0])
    

    print(f"ผลลัพธ์ของการทำนาย : {sentence_prediction[0]} | ความมั่นใจ : {sentence_accuracy * 100:.2f} \n")



    