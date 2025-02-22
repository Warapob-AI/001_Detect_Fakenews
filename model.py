from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from NLP import NeuroLinguisticProgramming
from Vectorizer import Vectorizer
class Model: 
    def __init__(self, x_train_pkl, x_test_pkl, y_train_pkl, y_test_pkl):
        self.x_train = x_train_pkl
        self.x_test = x_test_pkl
        self.y_train = y_train_pkl
        self.y_test = y_test_pkl

    def LightGBM_BERT(self):
        self.model = LGBMClassifier(
                n_estimators=2500,
                learning_rate=0.008,
                num_leaves=700,
                colsample_bytree=0.8,
                subsample=0.7,
                reg_alpha=0.2,
                reg_lambda=0.6,
                min_data_in_leaf=15,
                scale_pos_weight=1.6,  # ลดลง
                class_weight='balanced',  # ให้โมเดลบาลานซ์เอง
                boosting_type='gbdt',
                max_bin=255,
                force_col_wise=True,
                random_state=42,
                n_jobs=-1
            )

        self.model.fit(self.x_train, self.y_train) 
        self.y_pred = self.model.predict(self.x_test)

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

        sentence_vectorizer = vectorizer.encode_sentences(data_list)
        sentence_pred = self.model.predict(sentence_vectorizer)
        sentence_accuracy = max(self.model.predict_proba(sentence_vectorizer)[0])

        if (sentence_pred[0] == 1): 
            print(f'ผลลัพธ์การทำนาย : ข่าวจริง | ความมั่นใจ : {sentence_accuracy * 100:.2f}\n')
        else: 
            print(f'ผลลัพธ์การทำนาย : ข่าวปลอม | ความมั่นใจ : {sentence_accuracy * 100:.2f}\n')

NLP = NeuroLinguisticProgramming()
vectorizer = Vectorizer()
x_train_pkl, x_test_pkl, y_train_pkl, y_test_pkl = vectorizer.Load_Vectorizer()

model = Model(x_train_pkl, x_test_pkl, y_train_pkl, y_test_pkl)
model.LightGBM_BERT()
model.Classification_Report()

while True: 
    sentence = str(input('Enter Sentence : '))

    model.Sentence([sentence])
