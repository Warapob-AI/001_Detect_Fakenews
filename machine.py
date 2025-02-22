
from sklearn.metrics import accuracy_score, classification_report
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import GridSearchCV
from NLP import NeuroLinguisticProgramming
from lightgbm import LGBMClassifier
from scipy.stats import loguniform
from xgboost import XGBClassifier
from scipy.stats import randint
from sklearn.neural_network import MLPClassifier
import numpy as np
import joblib
import torch

class Machine:
    def __init__(self, list_model_0, list_model_1): 
        self.list_model_0 = list_model_0
        self.list_model_1 = list_model_1
        
        self.sentences = self.list_model_0 + self.list_model_1
        self.labels = [0] * len(self.list_model_0) + [1] * len(self.list_model_1) 

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.sentences, self.labels, test_size = 0.1, random_state = 42) 

    def encode_sentences(self, sentences, batch_size=8):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.check = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.check.to(self.device)

        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
            
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.check(**inputs)
            
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def Text_To_BERT(self): 
        self.x_train_vectorizer = self.encode_sentences(self.x_train)
        self.x_test_vectorizer = self.encode_sentences(self.x_test)

    def Save_Vectorizer(self):
        joblib.dump(self.x_train_vectorizer, 'x_train_vectorizer.pkl')
        joblib.dump(self.x_test_vectorizer, 'x_test_vectorizer.pkl')

    def Load_Vectorizer(self):
        self.x_train_vectorizer = joblib.load('x_train_vectorizer.pkl')
        self.x_test_vectorizer = joblib.load('x_test_vectorizer.pkl')

    def Logistic_Regresstion(self):
        self.model = LogisticRegression(random_state = 42)
        self.model.fit(self.x_train_vectorizer, self.y_train)

        self.y_pred = self.model.predict(self.x_test_vectorizer)
        
    def Random_Forest(self):
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.x_train_vectorizer, self.y_train)

        self.y_pred = self.model.predict(self.x_test_vectorizer)

    def LightGBM_BERT(self):
        self.model = LGBMClassifier(
            n_estimators=2000,
            max_depth=-1,
            learning_rate=0.01,
            num_leaves=512,
            colsample_bytree=0.9,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            min_data_in_leaf=10, 
            scale_pos_weight=6703 / 4205,
            boosting_type='gbdt',
            max_bin=255,
            force_col_wise=True,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(self.x_train_vectorizer, self.y_train) 
        self.y_pred = self.model.predict(self.x_test_vectorizer)

    def MLP(self):
        self.model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),  # จำนวน neurons ในแต่ละ layer
        activation='tanh',                    # activation function ที่ลองใช้ 'tanh'
        solver='adam',                        # ใช้ Adam optimizer
        max_iter=1000,                        # จำนวน iterations เพิ่ม
        learning_rate_init=0.001,             # กำหนด learning rate
        batch_size=64,                        # ขนาด batch สำหรับ Adam
        early_stopping=True,                  # ใช้ Early Stopping
        n_iter_no_change=10                   # หยุดหากผลไม่ดีขึ้น 10 รอบ
        )
        
        self.model.fit(self.x_train_vectorizer, self.y_train)
        self.y_pred = self.model.predict(self.x_test_vectorizer)

    def Save_Model(self): 
        joblib.dump(self.model, 'model_training.pkl')

    def XGBoost_BERT(self):
        self.model = XGBClassifier(
            learning_rate=0.01,
            max_depth=10,
            min_child_weight=17,
            n_estimators=100,
            n_jobs=-1,
            subsample=1.0,
            verbosity=0,
            random_state=42
        )

        self.model.fit(self.x_train_vectorizer, self.y_train)
        self.y_pred = self.model.predict(self.x_test_vectorizer)

    def XGBoost_BERT_Best_Params(self):
        param_dist = {
            'n_estimators': randint(100, 2000),
            'max_depth': randint(3, 50),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['sqrt', 'log2']
        }

        random_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=50,  # ลอง 50 ครั้ง
            scoring='accuracy',
            cv=5,
            verbose=2,
            n_jobs=-1
        )

        random_search.fit(self.x_train_vectorizer, self.y_train)
        print("Best params:", random_search.best_params_)
    
    def CatBoostClassifier(self):
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier(random_state=42)
        self.model.fit(self.x_train_vectorizer, self.y_train)
        self.y_pred = self.model.predict(self.x_test_vectorizer)

    def Classification_Report(self): 
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f'ความแม่นยำ : {self.accuracy * 100:.2f}')

        self.classification = classification_report(self.y_test, self.y_pred)
        print(self.classification)

    def Sentence(self, data_list):
        data_list = NLP.Lemmatization(data_list)
        data_list = NLP.Remove_Special_character(data_list)
        data_list = NLP.Tokenization(data_list)
        data_list = NLP.Stopwords(data_list)
        data_list = NLP.Lowercase(data_list)

        sentence_vectorizer = self.encode_sentences(data_list)
        sentence_pred = self.model.predict(sentence_vectorizer)
        sentence_accuracy = max(self.model.predict_proba(sentence_vectorizer)[0])

        if (sentence_pred[0] == 1): 
            print(f'ผลลัพธ์การทำนาย : ข่าวจริง | ความมั่นใจ : {sentence_accuracy * 100:.2f}\n')
        else: 
            print(f'ผลลัพธ์การทำนาย : ข่าวปลอม | ความมั่นใจ : {sentence_accuracy * 100:.2f}\n')

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


Model = Machine(list_fake_news, list_true_news)
# Model.Text_To_BERT()
# Model.Save_Vectorizer()
Model.Load_Vectorizer()
# Model.XGBoost_BERT()
Model.MLP()
Model.Classification_Report()

while True: 
    sentence = str(input('Enter Sentence : '))

    Model.Sentence([sentence])
