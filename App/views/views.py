from flask import Blueprint, render_template, request, jsonify, make_response, Response, redirect, url_for, session
from App.models.models import *
from datetime import datetime
from App.utils import send_email, generate_verification_code

page = Blueprint('user', __name__)


@page.route('/index/')
@page.route('/')
def index():
    userid = request.cookies.get('user_id')
    try:
        user_session = session.get('user_session')
    except Exception as e:
        print(e)
        return render_template('home.html')
    if str(user_session) == str(userid):
        return render_template('home.html', userid=userid)
    else:
        return render_template('home.html')


@page.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password, is_deleted=False).first()
        if user:
            try:
                current_time = datetime.now()
                current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
                new_history_record = History(
                    user_id=user.id,
                    action='用户登录',  # 记录操作内容为用户注册
                    action_time=current_time_str  # 使用当前时间作为操作时间
                )
                # 将新的历史记录对象添加到数据库中
                db.session.add(new_history_record)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                db.session.flush()
                print(e)
                return jsonify({'msg': '系统繁忙请稍后再试'})
            session['user_session'] = user.id
            return jsonify({'msg': 'success', 'id': user.id})
        else:
            return jsonify({'msg': '用户名或密码错误'})


@page.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    elif request.method == 'POST':
        # 获取当前时间并格式化为字符串
        username = request.form['username']
        password = request.form['password']
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        query_user = User.query.filter_by(username=username).first()
        if query_user:
            return jsonify({'msg': '该用户名已注册'})
        # 创建新用户对象
        user = User(
            username=username,
            nickname=username,
            password=password,
            is_deleted=False,
            created_at=current_time_str  # 使用格式化后的时间字符串
        )

        try:
            db.session.add(user)
            db.session.commit()
            # 创建新用户权限对象，权限级别设置为 1
            new_user_permission = UserPermission(
                user_id=user.id,
                permission_level=1,  # 设置权限级别为 1
                is_deleted=False,
                created_at=current_time_str  # 使用当前时间作为创建时间
            )
            # 将新用户权限对象添加到会话中
            db.session.add(new_user_permission)
            db.session.commit()  # 提交会话以保存用户权限到数据库中
            # 创建一个新的历史记录对象
            new_history_record = History(
                user_id=user.id,
                action='用户注册',  # 记录操作内容为用户注册
                action_time=current_time_str  # 使用当前时间作为操作时间
            )
            # 将新的历史记录对象添加到数据库中
            db.session.add(new_history_record)
            db.session.commit()
            session['user_session'] = user.id
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
            return jsonify({'msg': '系统繁忙请稍后再试'})
        return jsonify({'msg': 'success', 'id': user.id})


@page.route('/logout/')
def logout():
    response = redirect('/')
    userid = request.cookies.get('user_id')
    response.delete_cookie('user_id')
    session.clear()
    try:
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        new_history_record = History(
            user_id=userid,
            action='用户退出',  # 记录操作内容为用户注册
            action_time=current_time_str  # 使用当前时间作为操作时间
        )
        # 将新的历史记录对象添加到数据库中
        db.session.add(new_history_record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        db.session.flush()
        print(e)
    return response


@page.route('/feedback/', methods=['GET', 'POST'])
def feedback():
    if request.method == 'GET':
        return render_template('feedback.html')
    elif request.method == 'POST':
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        userid = request.cookies.get('user_id')
        if userid is None:
            userid = 0
        user = User.query.filter_by(id=userid, is_deleted=False).first()
        if user is None:
            return jsonify({'msg': '系统繁忙请稍后再试'})
        if user.email is None:
            return jsonify({'msg': '请先完善邮箱信息'})
        feedbackstr = request.form['feedback']
        # 创建一个新的反馈记录对象
        new_feedback = Feedback(
            user_id=userid,
            feedback_content=feedbackstr,
            feedback_time=current_time_str
        )
        new_history_record = History(
            user_id=userid,
            action='反馈',  # 记录操作内容为用户注册
            action_time=current_time_str  # 使用当前时间作为操作时间
        )
        try:
            # 将新的反馈记录对象添加到数据库中
            db.session.add(new_feedback)
            db.session.add(new_history_record)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
            return jsonify({'msg': '系统繁忙请稍后再试'})
        return jsonify({'msg': 'success'})


@page.route('/forgotPasswd/', methods=['GET', 'POST'])
def forgotPasswd():
    if request.method == 'GET':
        return render_template('forgotPasswd.html')
    elif request.method == 'POST':
        # print(request.form)
        username = request.form['username']
        email = request.form['email']
        code = request.form['verificationCode']
        newpassword = request.form['newPassword']
        try:
            verification_code = session['verification_code']
        except Exception as e:
            print(e)
            return jsonify({'msg': '请重新获取验证码'})
        # print(session.keys())
        if code != verification_code:
            return jsonify({'msg': '验证码错误'})
        query_user = User.query.filter_by(username=username, email=email, is_deleted=False).first()
        if query_user is None:
            return jsonify({'msg': '用户名或邮箱错误'})
        query_user.password = newpassword
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
        session.pop('verification_code')
        # print(session.keys())
        return jsonify({'msg': 'success'})


@page.route('/sendEmail/', methods=['GET', 'POST'])
def sendEmail():
    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        data = request.json
        email = data['email']
        username = data['username']
        # print(data)
        # print(email, username)

        if email is None or username is None:
            return jsonify({'msg': '网络拥挤请重试'})
        query_user = User.query.filter_by(username=username, email=email, is_deleted=False).first()
        # print(query_user)
        if query_user is None:
            return jsonify({'msg': '用户邮箱错误'})
        code = generate_verification_code()
        session['verification_code'] = code
        session.permanent = True
        try:
            # print(email, code)
            send_email(email, code)
        except Exception as e:
            print(e)
            return jsonify({'msg': '系统繁忙请稍后再试'})
        return jsonify({'msg': 'success'})
