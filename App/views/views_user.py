from flask import Blueprint, render_template, request, jsonify, session, redirect

from App.exts import db
from App.models.models import User, UserPermission
from App.utils import generate_verification_code, send_email

user_center = Blueprint('center', __name__)


@user_center.route('/center/')
def center_page():
    response = redirect('/login/')
    permission = 1
    try:
        user_id = str(request.cookies.get('user_id'))
        user_session = str(session.get('user_session'))
        print(user_id, user_session)
        if user_id:
            role = UserPermission.query.filter_by(user_id=user_id).first()
            permission = role.permission_level
            print(role.permission_level)
    except Exception as e:
        print(e)
        return response
    if user_session == user_id:
        user = User.query.filter_by(id=int(user_id), is_deleted=False).first()
        if permission == 1:
            return render_template("userCenter.html", user=user)
        else:
            return render_template("admin.html", user=user)
        # return render_template("userCenter.html")
    return response


@user_center.route('/getuser/')
def get_user():
    try:
        user_id = str(request.cookies.get('user_id'))
        user_session = str(session.get('user_session'))
    except Exception as e:
        print(e)
        return jsonify({'msg': '用户错误'})
    if user_session != user_id:
        return jsonify({'msg': '用户错误'})
    user = User.query.filter_by(id=user_id, is_deleted=False).first()
    if user is None:
        return jsonify({'msg': '用户错误'})
    if not user.is_deleted:
        status = '活跃'
    else:
        status = '禁用'
    ret = jsonify({'msg': 'success', 'data': {
        'userid': user.id,
        'username': user.username,
        'nickname': user.nickname,
        'email': user.email,
        'status': status
    }})
    return ret


@user_center.route('/deluser/', methods=['DELETE'])
def del_user():
    if request.method == 'DELETE':
        try:
            user_id = str(request.cookies.get('user_id'))
            user_session = str(session.get('user_session'))
        except Exception as e:
            print(e)
            return jsonify({'msg': '用户错误'})
        if user_session != user_id:
            return jsonify({'msg': '用户错误'})
        # print(user_id)
        user = User.query.filter_by(id=user_id, is_deleted=False).first()
        if user is None:
            return jsonify({'msg': '用户错误'})
        user.is_deleted = True
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
        response = redirect('/')
        response.delete_cookie('user_id')
        session.clear()
        return jsonify({'msg': 'success'})


@user_center.route('/changeemail/', methods=['GET', 'POST'])
def change_email():
    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        data = request.json
        email = data['email']
        nickname = data['username']
        code = generate_verification_code()
        session['change_code'] = code
        session.permanent = True
        # print(email, code)
        try:
            print(email, code)
            send_email(email, code)
        except Exception as e:
            print(e)
            return jsonify({'msg': '系统繁忙请稍后再试'})
        return jsonify({'msg': 'success'})


@user_center.route('/changeuser/', methods=['GET', 'POST'])
def change_user():
    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        user_session = str(session.get('user_session'))
        nickname = request.form['username']
        email = request.form['email']
        code = request.form['verificationCode']
        try:
            change_code = session['change_code']
        except Exception as e:
            print(e)
            return jsonify({'msg': '请重新获取验证码'})
        if code != change_code:
            return jsonify({'msg': '验证码错误'})
        user = User.query.filter_by(id=user_session).first()
        user.nickname = nickname
        user.email = email
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
        session.pop('change_code')
        return jsonify({'msg': 'success'})


@user_center.route('/resetpasswd/', methods=['GET', 'POST'])
def reset_passwd():
    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        # print(request.form)
        user_session = str(session.get('user_session'))
        old_password = request.form['oldPassword']
        new_password = request.form['newPassword']
        # print(old_password)
        # print(new_password)
        user = User.query.filter_by(id=user_session, password=old_password, is_deleted=False).first()
        if user is None:
            return jsonify({'msg': '用户密码不匹配'})
        user.password = new_password
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
        return jsonify({'msg': 'success'})
