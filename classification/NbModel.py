#import Library
import re
import pandas as pd
import time
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
nltk.download('stopwords')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class NbModel():
    #constructor
    def __init__(self, datasetPath):
        self.classifier = None
        self.tf_vect = None

        self.dataset = pd.read_csv(datasetPath)

        # self.label = dataset.label      #set kolom label pd dataset ke dalam variabel
        
        # self.dataset = dataset.drop("label", axis = 1)      #drop kolom label pada dataset

    #fungsi untuk train_test_split pada scikit-learn
    def variabel(self):
        #split dataset, 6/10 menjadi data latih, 4/10 menjadi data uji
        perkara_train, perkara_test, pasal_train, pasal_test = train_test_split(self.dataset['kronologi'], self.dataset['pasal'], test_size=0.2, random_state=32)
        return perkara_train, perkara_test, pasal_train, pasal_test

    #fungsi untuk vectorizing split_data with TFIDF
    def split_data(self):
        self.tf_vect = TfidfVectorizer()
        self.text_tf = self.tf_vect.fit_transform(self.dataset['kronologi'].astype('U'))
    #         tf_vect.fit(perkara['kronologi'])
    #     perkara_train, perkara_test, pasal_train, pasal_test = variabel()
        perkara_train_tf, perkara_test_tf, pasal_train, pasal_test = train_test_split(self.text_tf, self.dataset['pasal'], test_size=0.2, random_state=32)

        #ubah dataset menjadi vector)
        return perkara_train_tf, perkara_test_tf, pasal_train, pasal_test
        # ----------------------------


        # self.tf_vect = TfidfVectorizer()
        # self.tf_vect.fit(self.dataset['kronologi'])
        # perkara_train, perkara_test, pasal_train, pasal_test = self.variabel()

        # #ubah dataset menjadi vector
        # perkara_train_tf = self.tf_vect.fit_transform(perkara_train)
        # perkara_test_tf = self.tf_vect.transform(perkara_test)
        # return perkara_train_tf, perkara_test_tf, pasal_train, pasal_test

        

    #fungsi untuk vectorizing data uji
    def vectorizer_data_test(self, datatestPath):
        return self.tf_vect.transform(datatestPath)
    
    #fungsi untuk klasifikasi svm>ganti jadi nb
    def nb(self, perkara_train_tf, pasal_train):
        self.classifier = MultinomialNB()
        self.classifier.fit(perkara_train_tf, pasal_train)
    
    #fungsi untuk prediksi data uji
    def predict(self, nb_test):
        return self.classifier.predict(nb_test)

    #fungsi untuk skor akurasi
    def akurasi(self, y_asli, y_output):
        return metrics.accuracy_score(y_asli, y_output)

    #fungsi untuk nampilin klasifikasi report (precission, recall, f1-score, support)
    def classi_report(self, y_test, y_output):
        return classification_report(y_test, y_output)

    def conf_matrix(self, y_test, y_output):
        return confusion_matrix(y_test, y_output)

    #fungsi untuk preprocessing data testing
    def read_split(self, datatestPath):
        with open(datatestPath, "r",encoding="utf8") as file:
            test_file = file.read()
            file.close()

        #casefolding menjadi huruf kecil
        text_low = test_file.lower().strip()



        #cleaning text
        re_clean = re.sub('(@[A-Za-z0-9]+)|(#{A-Za-20-9]+)|(\w+\/\/\S+)|(http\S+)','',text_low)
        re_clean = re.sub(r'_', '', re_clean) # menghapus _
        re_clean = re.sub(r'/', '  ', re_clean) #menggantikan / menjadi spasi
        re_clean = re.sub(r'\d+', '', re_clean) #menghapus angka
        re_clean = re.sub(r'\n', ' ', re_clean) # untk mengubah enter menjadi spasi
        re_clean = re.sub(r'[^A-Za-z\s\/]' ,' ', re_clean) #menghapus karakter yang bukan huruf

        stop_words=set(stopwords.words('indonesian'))
        stop_words.update(['terdakwa','twitter','fb','facebook','status','mengupload','memposting','wa','bbm','saksi','blackberry','messenger','januari','februari','maret','april','mei','juni','juli','agustus','september','oktober','november','desember','media','sosial','medsos','tanggal'])
        # stop_words.update(['januari','februari','maret','april','mei','juni','juli','agustus','september','oktober','november','desember','media','sosial','medsos','tanggal'])
        no_stopwords = [word for word in re_clean.split() if word not in stop_words]
        no_step = ' '.join(no_stopwords)
        # return no_step

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return [stemmer.stem(no_step)]

        sentence = re_clean.split(" ")
        # print(re_clean)
        
        new_text = ""
        for i in sentence:
            new_text += i+" "
        a = new_text
        b = a.split(".")
        del b[-1]
        return b

    def casefolding(perkara):
        perkara = perkara.lower().strip()
        return perkara

    def clean(perkara):
        perkara = ''.join(re.sub("(@[A-Za-z0-9]+)|(#{A-Za-20-9]+)|(\w+\/\/\S+)|(http\S+)", "", perkara)) #hapus

        perkara = re.sub(r'_', '', perkara) # menghapus _
        perkara = re.sub(r'/', '  ', perkara) #menggantikan / menjadi spasi
        perkara = re.sub(r'\d+', '', perkara) #menghapus angka
        perkara = re.sub(r'\n', ' ', perkara) # untk mengubah enter menjadi spasi
        perkara = re.sub(r'[^A-Za-z\s\/]' ,' ', perkara) #menghapus karakter yang bukan huruf
        return perkara

    def remove_stopwords(perkara):
        stop_words=set(stopwords.words('indonesian'))
        stop_words.update(['terdakwa','twitter','fb','facebook','status','mengupload','memposting','wa','bbm','saksi','blackberry','messenger','januari','februari','maret','april','mei','juni','juli','agustus','september','oktober','november','desember','media','sosial','medsos','tanggal'])
    # stop_words.update(['januari','februari','maret','april','mei','juni','juli','agustus','september','oktober','november','desember','media','sosial','medsos','tanggal'])
        no_stopwords = [word for word in perkara.split() if word not in stop_words]
        no_step = ' '.join(no_stopwords)
        return no_step

    def stemmer(perkara):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return stemmer.stem(perkara)


    #fungsi untuk proses prediksi data uji
    def ProcessingText(self,datatestPath):
        sentence_vect = self.vectorizer_data_test(datatestPath)
        prediksi_test = self.predict(sentence_vect)
        dataHasil = {}
        # result = pd.DataFrame({'Kalimat' : datatestPath, 'Label' : prediksi_test})
        for i in range(len(prediksi_test)):
            dataHasil = {
                'kalimat':datatestPath,
                'label':prediksi_test
            }
        # print(dataHasil["label"])
        return [dataHasil]

#Main
if __name__=='__main__':
    print("berhasil")