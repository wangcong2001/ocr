from flask import Blueprint, render_template, request, jsonify, make_response, Response, redirect, url_for, session, \
    send_from_directory
from App.exts import db
from App.models.models import User, Image, ImageData
from datetime import datetime
from werkzeug.utils import secure_filename
from App.utils import rec, contains_chinese, check
import wordninja
from autocorrect import Speller
from pycorrector import Corrector, EnSpellCorrector, MacBertCorrector

import os

img_page = Blueprint('image', __name__)

UPLOAD_FOLDER = 'App\\static\\uploads'
UPLOAD_FOLDER_DB = '/uploads'

# 创建拼写校正器
spell = Speller()
zh_corrector = Corrector()
en_corrector = EnSpellCorrector()
deep_corrector = MacBertCorrector()


@img_page.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)


@img_page.route('/upload/', methods=['POST'])
def upload_file():
    current_time = datetime.now()
    current_time_str = current_time.strftime('%Y_%m_%d_%H_%M_%S')
    current_time_str1 = current_time.strftime('%Y-%m-%d %H:%M:%S')
    if 'file' not in request.files:
        return jsonify({"msg": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"msg": "No selected file"})
    if file:
        user_id = request.cookies.get('user_id')
        if user_id is None:
            user_id = 0
        user_folder = os.path.join(UPLOAD_FOLDER, f'user_{user_id}')
        user_folder_db = os.path.join(UPLOAD_FOLDER_DB, f'user_{user_id}')
        os.makedirs(user_folder, exist_ok=True)
        filename = secure_filename(file.filename)
        filename = f"{filename.split('.')[0]}_{current_time_str}.{filename.split('.')[1]}"
        file_path = os.path.join(user_folder, filename)
        file_path_db = os.path.join(user_folder_db, filename)
        file.save(file_path)
        new_img = Image(
            image_path=file_path_db,
            user_id=user_id,
            image_collection_id=0,
            is_deleted=False,
            created_at=current_time_str1
        )
        try:
            db.session.add(new_img)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
            return jsonify({"msg": "上传失败"})
        # 将文件路径添加到数据库
        return jsonify({"msg": "success",
                        "data":
                            {'user_id': user_id,
                             'id': new_img.id,
                             'image_path': new_img.image_path,
                             'img_text': []
                             }
                        })


@img_page.route('/uploadmul/', methods=['POST'])
def upload_files():
    current_time = datetime.now()
    current_time_str = current_time.strftime('%Y_%m_%d_%H_%M_%S')
    current_time_str1 = current_time.strftime('%Y-%m-%d %H:%M:%S')

    user_id = request.cookies.get('user_id')
    if user_id is None:
        user_id = 0
    # print(user_id)
    user_folder = os.path.join(UPLOAD_FOLDER, f'user_{user_id}')
    user_folder_db = os.path.join(UPLOAD_FOLDER_DB, f'user_{user_id}')
    os.makedirs(user_folder, exist_ok=True)

    results = []
    print(request.files.getlist('files[]'))
    for file in request.files.getlist('files[]'):
        if file.filename == '':
            results.append({"msg": "No selected file"})
            continue

        if file:
            filename = secure_filename(file.filename)
            print(filename)
            filename = f"{filename.split('.')[0]}_{current_time_str}.{filename.split('.')[1]}"
            file_path = os.path.join(user_folder, filename)
            file_path_db = os.path.join(user_folder_db, filename)
            file_path_db = file_path_db.replace('\\', '/')
            file.save(file_path)
            new_img = Image(
                image_path=file_path_db,
                user_id=user_id,
                image_collection_id=0,
                is_deleted=False,
                created_at=current_time_str1
            )

            try:
                db.session.add(new_img)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                db.session.flush()
                print(e)
                results.append({"msg": "上传失败", "file_name": file.filename})
                continue

            results.append({"msg": "success",
                            "data":
                                {'user_id': user_id,
                                 'id': new_img.id,
                                 'image_path': new_img.image_path,
                                 'img_text': []
                                 }
                            })
    # print(results)
    return jsonify(results)


@img_page.route('/getimgdata/')
def getimgdata():
    user_id = int(session.get('user_session'))
    ret = []
    img_all = Image.query.filter_by(user_id=user_id, is_deleted=False).all()
    imgdata = {}
    for img in img_all:
        item = {'img_id': str(img.id), 'img_path': str(img.image_path)}
        data = ImageData.query.filter_by(img_id=img.id, is_deleted=False).order_by(ImageData.id).first()
        if data is None:
            item['img_text'] = "无"
        else:
            item['img_text'] = data.data_content
        ret.append(item)
    return jsonify({'msg': 'success', 'data': ret})


@img_page.route('/deleteimages/', methods=['DELETE'])
def deleteimages():
    if request.method == 'DELETE':
        user_id = int(session.get('user_session'))
        # print(request.json['selectedImages'])
        img_list = request.json['selectedImages']
        for img_id in img_list:
            # print(img_id)
            img_id = int(img_id)
            img = Image.query.filter_by(id=img_id, user_id=user_id, is_deleted=False).first()
            if img:
                data_id = img.data_id
                data = ImageData.query.filter_by(id=data_id, user_id=user_id, is_deleted=False).first()
                if data:
                    data.is_deleted = True
                img.is_deleted = True
                try:
                    db.session.commit()
                except Exception as e:
                    print(e)
                    db.session.rollback()
                    db.session.flush()
        return jsonify({'msg': 'success'})


@img_page.route('/recognize/')
def recognize():
    if request.method == 'GET':
        img_id = request.args.get('id')
        user_id = int(session.get('user_session'))
        # print(img_id)
        img = Image.query.filter_by(id=img_id, user_id=user_id, is_deleted=False).first()
        if img is None:
            return jsonify({'msg': '图片不存在'})
        img_path = img.image_path
        img_path = './App/static' + img_path
        # print(img_path)
        # print(os.path.exists(img_path))
        # result, elapse = engine(img_path)
        result = rec(img_path)
        text = ''
        # 带矫正
        for item in result:
            box = item['box']
            sentence = item['text']
            score = item['score']
            if contains_chinese(sentence):
                mid_sentence = deep_corrector.correct(sentence)
                flag = check(mid_sentence['errors'])
                if score < 0.85 or flag:
                    print("矫正前: ", sentence)
                    print("矫正后: ", mid_sentence['target'])
                    new_sentence = mid_sentence['target']
                else:
                    new_sentence = sentence
            else:
                tokens = wordninja.split(sentence)
                mid_sentence = ' '.join(tokens)
                print("矫正前: ", sentence)
                print("矫正后: ", mid_sentence)
                new_sentence = mid_sentence
            text += new_sentence
            text += '\n'
        # 无矫正
        # for item in result:
        #     sentence = item['text']
        #     text += sentence
        #     text += '\n'

        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        img_data = ImageData(
            user_id=user_id,
            img_id=img_id,
            data_content=text,
            is_deleted=False,
            created_at=current_time_str
        )
        try:
            # 将新的反馈记录对象添加到数据库中
            db.session.add(img_data)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
            return jsonify({'msg': '系统繁忙请稍后再试'})
        return jsonify({'msg': 'success', 'data': text})
