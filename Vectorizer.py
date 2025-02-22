from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from NLP import NeuroLinguisticProgramming

import numpy as np
import joblib
import torch 

class Vectorizer: 
    def Import_Data(self, list_fake_news, list_true_news):
        self.list_fake_news = list_fake_news
        self.list_true_news = list_true_news

        self.sentences = list_fake_news + list_true_news
        self.labels = [0] * len(list_fake_news) + [1] * len(list_true_news)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.sentences, self.labels, test_size = 0.2, random_state = 42)

        print('Test 1 Success ✅')

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
        print('Test 2 Success ✅')
        self.x_test_vectorizer = self.encode_sentences(self.x_test)
        print('Test 3 Success ✅')


    def Save_Vectorizer(self):
        joblib.dump(self.x_train_vectorizer, 'x_train_vectorizer.pkl')
        joblib.dump(self.x_test_vectorizer, 'x_test_vectorizer.pkl')
        joblib.dump(self.y_train, 'y_train.pkl')
        joblib.dump(self.y_test, 'y_test.pkl')

        print('Test 4 Success ✅')

    def Load_Vectorizer(self): 
        self.x_train_vectorizer = joblib.load('x_train_vectorizer.pkl')
        self.x_test_vectorizer = joblib.load('x_test_vectorizer.pkl')
        self.y_train = joblib.load('y_train.pkl')
        self.y_test = joblib.load('y_test.pkl')

        return self.x_train_vectorizer, self.x_test_vectorizer, self.y_train, self.y_test


# NLP = NeuroLinguisticProgramming()

# list_fake_news = NLP.Load_data('fake_news.txt')
# list_true_news = NLP.Load_data('true_news.txt')

# list_fake_news = NLP.Lemmatization(list_fake_news)
# list_fake_news = NLP.Remove_Special_character(list_fake_news)
# list_fake_news = NLP.Tokenization(list_fake_news)
# list_fake_news = NLP.Stopwords(list_fake_news)
# list_fake_news = NLP.Lowercase(list_fake_news)

# list_true_news = NLP.Lemmatization(list_true_news)
# list_true_news = NLP.Remove_Special_character(list_true_news)
# list_true_news = NLP.Tokenization(list_true_news) 
# list_true_news = NLP.Stopwords(list_true_news)
# list_true_news = NLP.Lowercase(list_true_news) 

# vectorizer = Vectorizer()
# vectorizer.Import_Data(list_fake_news, list_true_news)
# vectorizer.Text_To_BERT()
# vectorizer.Save_Vectorizer()
