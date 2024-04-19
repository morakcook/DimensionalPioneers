from flask import Flask, request, jsonify, render_template, send_file
import Run
import pandas as pd
import os
from werkzeug.utils import secure_filename
import shutil
import glob

def find_file_by_name(folder_path, file_name):
    for file in os.listdir(folder_path):
        base, extension = os.path.splitext(file)
        if base == file_name:
            return os.path.join(folder_path, file)
    return None

def get_web_service_app(run):

    app = Flask(__name__)
    app.config['INPUT_FOLDER'] = f'{os.getcwd()}/static/Data/Input'
    app.config['INPUT_LOG_FOLDER'] = f'{os.getcwd()}/static/Data/Input_log'
    app.config['OUTPUT_LOG_FOLDER'] = f'{os.getcwd()}/static/Data/Output_log'
    weights_path = f"{os.getcwd()}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    input_path = f"{os.getcwd()}/static/Data/Input/Input.png"
    Interim_deliverables = f"{os.getcwd()}/static/Data/Interim_deliverables"
    output_file_path = f"{os.getcwd()}/static/Data/Output/output_model.glb"
    load_Interim_deliverables = "../static/Data/Interim_deliverables"
    load_output_file_path = "../static/Data/Output/output_model.glb"

    @app.route('/')
    def index():
        return render_template('index_main.html')  

    @app.route('/start', methods=['POST'])
    def start():

        # 입력한 이미지를 input디렉토리, input_log디렉토리에 저장
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            inputname = 'Input.png'
            file_path = os.path.join(app.config['INPUT_FOLDER'], inputname)
            log_path = os.path.join(app.config['INPUT_LOG_FOLDER'], filename)
            file.save(file_path)
            file.seek(0)
            file.save(log_path)
            
        # 이미지 생성 및 3d파일 생성
        # run(input_path, Interim_deliverables, output_file_path, weights_path)

        # 3d파일 log디렉토리에 복제
        base = os.path.splitext(filename)[0]  # 확장자를 제외한 파일 이름
        new_filename = base + ".glb"  # 새 확장자 추가
        output_log_path = os.path.join(app.config['OUTPUT_LOG_FOLDER'], new_filename)
        shutil.copy(output_file_path, output_log_path)
        
        # 준비된 파일들을 읽기위한 경로 준비
        response_data = {"Interim_deliverables": load_Interim_deliverables, "output_file_path": load_output_file_path}  # 입력 텍스트와 생성된 텍스트들을 JSON 데이터로 만듦
        response = jsonify(response_data)  # JSON 응답 생성
        return response  # JSON 응답 반환

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

    @app.route('/download', methods=['GET'])
    def download_file():
        # 실제 파일 경로 설정
        file_path = '/home/DimensionalPioneers/static/Data/Output_log/img006.glb'
        # 클라이언트에게 파일 제공
        return send_file(file_path, as_attachment=True)
    
    @app.route('/<path:page>')
    def dynamic_page(page):
        if page == "webpage_history":
            path = '/home/DimensionalPioneers/static/Data/Output_log/*'
            log = [os.path.basename(file) for file in glob.glob(path)]
            return render_template(f'{page}.html', log=log)
        return render_template(f'{page}.html')

    @app.route('/read_row', methods=['POST'])
    def read_row():
        if 'file' not in request.form:
            return jsonify({'error': 'No file part'}), 400

        file = request.form['file']
        print(file)
        if file == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file)
        base, extension = os.path.splitext(filename)

        file_path = find_file_by_name("/home/DimensionalPioneers/static/Data/Input_log", base)
        file_path = file_path.split('/')[-1]
        origin_img_path = f"../static/Data/Input_log/{file_path}"
        glb_path = f"../static/Data/Output_log/{filename}"
        print(glb_path)
        response_data = {
            "origin_img_path": origin_img_path,
            "glb_path": glb_path
        }
        response = jsonify(response_data)
        return response

    return app

if __name__ == "__main__":
    app = get_web_service_app(Run.run)
    app.run(debug=True)
