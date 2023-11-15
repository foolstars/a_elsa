import os
import csv
import subprocess
from flask import Flask, render_template, request, send_file
from network_01 import creat_network

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def root():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    dataFile = request.files['dataFile']
    if dataFile.filename == '':
        return 'No selected file', 400
    else:
        if dataFile and allowed_file(dataFile.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataFile.filename)
            dataFile.save(file_path)
        first_file = dataFile

    if 'extraFile' in request.files:
        extraFile = request.files['extraFile']
        if extraFile.filename == '':
            return 'No extraFile', 400
        else:
            
            if extraFile and allowed_file(extraFile.filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], extraFile.filename)
                extraFile.save(file_path)
            second_file = extraFile
    else:
        second_file = dataFile

    delayLimit=0
    repNum=1
    spotNum=4
    minoccur=50
    precision=1000
    approxVar=1.0
    progressive=1

    if (request.form['delayLimit']):
        delayLimit = int(request.form['delayLimit']) 
        # print(delayLimit,"hello")
        # print("hello")

    if (request.form['minoccur']):
        minoccur = int(request.form['minoccur'])
        # print(minoccur)

    if (request.form['precision']):
        precision = int(request.form['precision'])
        # print(precision)

    if (request.form['repNum']):
        repNum = int(request.form['repNum'])
        # print(repNum)

    if (request.form['spotNum']):
        spotNum = int(request.form['spotNum'])
        # print(spotNum)

    if (request.form['trendThresh']):
        trendThresh = float(request.form['trendThresh'])
    if (request.form['approxVar']):
        approxVar = float(request.form['approxVar'])
        # print(approxVar)

    if (request.form['progressive']):
        progressive = int(request.form['progressive'])
        # print(progressive)

    qvalueMethod = request.form['qvalueMethod']
    # print(qvalueMethod)

    pvalueMethod = request.form['pvalueMethod']
    # print(pvalueMethod)
    
    bootNum = request.form['bootNum']
    # print(bootNum)

    transFunc = request.form['transFunc']   
    # print(transFunc)

    normMethod = request.form['normMethod']
    # print(normMethod)

    fillMethod = request.form['fillMethod']
    # print(fillMethod)

    if (request.form['resultFile']):
        resultFile = (request.form['resultFile'])+'.csv'
    else:
        resultFile = 'resultFile.csv'


    # 构建命令
    script_name = "../lsa_gpu_compute.py"
    command = f"python {script_name} {os.path.join(app.config['UPLOAD_FOLDER'], first_file.filename)} {resultFile} \
                        -e {os.path.join(app.config['UPLOAD_FOLDER'], second_file.filename)} \
                        -d {delayLimit} -r {repNum} -s {spotNum} -m {minoccur} -p {pvalueMethod} \
                        -x {precision} -b {bootNum} -t {transFunc} -f {fillMethod} -n {normMethod} \
                        -q {qvalueMethod} -a {approxVar} -v {progressive}"
    
    subprocess.run(command, shell=True)
    
    return 'computation completed'

@app.route('/network', methods=['GET', 'POST'])
def network():
    creat_network()
    return "Network visualization is being created..."

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5002)
