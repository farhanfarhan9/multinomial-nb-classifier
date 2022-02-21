from flask import Flask, render_template, jsonify, request, Response, send_file
from werkzeug.utils import secure_filename
from classification.NbModel import NbModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import json, os, csv
import pandas as pd
import datetime

UPLOAD_FOLDER = 'data/'
ALLOWED_EXTENSIONS = set(['json'])
ALLOWED_EXTENSIONS_TXT = set(['txt'])
ALLOWED_EXTENSIONS_SVM = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file_txt(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_TXT

def allowed_file_svm(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_SVM

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/training_result', methods=['POST', 'GET'])
def training_result():
    error = ""
    if 'file_dataset' in request.files:
        filetext = request.files["file_dataset"]
        print(filetext)
        filetext1 = request.files["file_dataset"]
        if filetext and allowed_file_svm(filetext.filename):
            filename = secure_filename(filetext.filename)
            filetext.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            dataset = pd.read_csv('data/' + filetext.filename)
        else:
            error = "Format file salah !"

    nbModelPath = 'data/' + filetext.filename

    obj = NbModel(nbModelPath)
    # split_p = obj.split_data()
    perkara_train_tf, perkara_test_tf, pasal_train, pasal_test = obj.split_data()
    obj.nb(perkara_train_tf, pasal_train)

    prediksi_dataset = obj.predict(perkara_test_tf)
    akurasi_dataset = obj.akurasi(pasal_test, prediksi_dataset) * 100

    table = classification_report(pasal_test, prediksi_dataset)
    split_tab = table.split()

    data_classi_report = [akurasi_dataset, split_tab]

    return render_template ('datatable_classification_result.html', data = data_classi_report)

@app.route('/uji_dokumen')
def test_classification_result():
    return render_template('/testing_dokumen.html')


@app.route('/testing_document_result', methods=['POST', 'GET'])
def hasil_uji_dokumen():
    error = ""
    if 'text_textbox' in request.form:
        filetextnya = request.form['text_textbox']
        # print(filetext)
        filenamenya = "uploaded"+datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+".txt"
        try:
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filenamenya), 'w', encoding='utf-8') as f:
                f.write(filetextnya)
        except FileNotFoundError:
            print("The 'data' directory does not exist")


        datatestPath = 'data/' + filenamenya
        nbModelPath = 'data/data_pasal_bersih.csv'
        objTest = NbModel(nbModelPath)
        cerita_train_tf, cerita_test_tf, label_train, label_test = objTest.split_data()
        objTest.nb(cerita_train_tf, label_train)
        read_file = objTest.read_split(datatestPath)
    
    else:
        if 'file_testing' in request.files:
            filetext = request.files["file_testing"]
            # print(filetext)
            # filetext = request.files["file_testing"]
            if filetext and allowed_file_txt(filetext.filename):
                filename = secure_filename(filetext.filename)
                filetext.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                with open(os.path.join(app.config['UPLOAD_FOLDER'], filetext.filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    f.close()
            else:
                error = "Format file salah !" 

            datatestPath = 'data/' + filetext.filename
            nbModelPath = 'data/data_pasal_bersih.csv'


            objTest = NbModel(nbModelPath)
            # split_p = objTest.split_data()
            cerita_train_tf, cerita_test_tf, label_train, label_test = objTest.split_data()
            objTest.nb(cerita_train_tf, label_train)
            read_file = objTest.read_split(datatestPath)
    

    hasilnya = objTest.ProcessingText(read_file)

    penampung = []

    # return "berhasil"
    return render_template('datatable_test_result.html', data=hasilnya)

if __name__=="__main__":
    app.run(debug = True)

    app.run()