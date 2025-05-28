from flask import Blueprint, render_template, request, jsonify, session, redirect

from App.exts import db
from App.models.models import User, UserPermission, Feedback
from App.utils import generate_verification_code, send_email
from App.utils import reply_feedback, generate_verification_code, hash_password_with_sha256
admin_page = Blueprint('admin_page', __name__)


@admin_page.route('/getalluser/')
def getAllUser():
    try:
        user_id = str(request.cookies.get('user_id'))
        user_session = str(session.get('user_session'))
    except Exception as e:
        print(e)
        return jsonify({'msg': '用户错误'})
    role = UserPermission.query.filter_by(user_id=user_id).first()
    if role is None or role.permission_level != 0:
        return jsonify({'msg': '用户权限错误'})
    users = User.query.filter_by(is_deleted=False).all()
    if users is None:
        return jsonify({'msg': '无用户'})
    user_list = []
    for user in users:
        user_data = {
            'userid': user.id,
            'username': user.username,
            'nickname': user.nickname,
            'email': user.email,
        }
        user_list.append(user_data)
    return jsonify({'msg': 'success', 'data': user_list})


@admin_page.route('/deleteuser/<delete_id>', methods=['DELETE'])
def deleteUser(delete_id):
    if request.method == 'DELETE':
        try:
            user_id = str(request.cookies.get('user_id'))
            user_session = str(session.get('user_session'))
        except Exception as e:
            print(e)
            return jsonify({'msg': '用户错误'})
        role = UserPermission.query.filter_by(user_id=user_id).first()
        if role is None or role.permission_level != 0:
            return jsonify({'msg': '用户权限错误'})
        user = User.query.filter_by(id=delete_id, is_deleted=False).first()
        if user is None:
            return jsonify({'msg': '用户不存在'})
        else:
            print("shanchu ", user)
            user.is_deleted = True
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                db.session.flush()
                print(e)
            print("删除成功")
            return jsonify({'msg': 'success'})


@admin_page.route('/admingetuser/<userid>', methods=['GET'])
def getUser(userid):
    if request.method == 'GET':
        try:
            user_id = str(request.cookies.get('user_id'))
            user_session = str(session.get('user_session'))
        except Exception as e:
            print(e)
            return jsonify({'msg': '用户错误'})
        role = UserPermission.query.filter_by(user_id=user_id).first()
        if role is None or role.permission_level != 0:
            return jsonify({'msg': '用户权限错误'})
        user = User.query.filter_by(id=userid, is_deleted=False).first()
        if user is None:
            return jsonify({'msg': '用户错误'})
        return jsonify({'msg': 'success', 'data': {
            'userid': user.id,
            'username': user.username,
            'nickname': user.nickname,
            'email': user.email,
        }})


@admin_page.route('/adminupdateuser/<userid>', methods=['PUT'])
def updateUser(userid):
    if request.method == 'PUT':
        data = request.json
        try:
            user_id = str(request.cookies.get('user_id'))
            user_session = str(session.get('user_session'))
        except Exception as e:
            print(e)
            return jsonify({'msg': '用户错误'})
        role = UserPermission.query.filter_by(user_id=user_id).first()
        if role is None or role.permission_level != 0:
            return jsonify({'msg': '用户权限错误'})
        user = User.query.filter_by(id=userid, is_deleted=False).first()
        if user is None:
            return jsonify({'msg': '用户错误'})
        user.nickname = data['nickname']
        user.email = data['email']
        user.username = data['username']
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
            return jsonify({'msg': '系统繁忙请稍后再试'})
        return jsonify({'msg': 'success'})


@admin_page.route('/adminresetuser/<userid>', methods=['POST'])
def resetUser(userid):
    if request.method == 'POST':
        try:
            user_id = str(request.cookies.get('user_id'))
            user_session = str(session.get('user_session'))
        except Exception as e:
            print(e)
            return jsonify({'msg': '用户错误'})
        role = UserPermission.query.filter_by(user_id=user_id).first()
        if role is None or role.permission_level != 0:
            return jsonify({'msg': '用户权限错误'})
        user = User.query.filter_by(id=userid, is_deleted=False).first()
        passwd = generate_verification_code()
        hash_passwd = hash_password_with_sha256(passwd)
        print(passwd)
        print(hash_passwd)
        user.password = hash_passwd

        reply_feedback(user.email, "您的密码已重置为:" + passwd)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
            return jsonify({'msg': '系统繁忙请稍后再试'})
        return jsonify({'msg': 'success'})


@admin_page.route('/getallfeedback/', methods=['GET'])
def getAllFeedback():
    if request.method == 'GET':
        try:
            user_id = str(request.cookies.get('user_id'))
            user_session = str(session.get('user_session'))
        except Exception as e:
            print(e)
            return jsonify({'msg': '用户错误'})
        role = UserPermission.query.filter_by(user_id=user_id).first()
        if role is None or role.permission_level != 0:
            return jsonify({'msg': '用户权限错误'})
        feedbacks = Feedback.query.filter_by().all()
        if feedbacks is None:
            return jsonify({'msg': '无反馈'})
        feed_list = []
        for item in feedbacks:
            user = User.query.filter_by(id=item.user_id, is_deleted=False).first()
            if user is None:
                continue
            item_data = {
                'userid': item.user_id,
                'feedid': item.id,
                'username': user.username,
                'nickname': user.nickname,
                'email': user.email,
            }
            feed_list.append(item_data)
        if len(feed_list) == 0:
            return jsonify({'msg': '无反馈'})
        return jsonify({'msg': 'success', 'data': feed_list})


@admin_page.route('/deletefeedback/<deleteid>', methods=['DELETE'])
def deleteFeedback(deleteid):
    if request.method == 'DELETE':
        try:
            user_id = str(request.cookies.get('user_id'))
            user_session = str(session.get('user_session'))
        except Exception as e:
            print(e)
            return jsonify({'msg': '用户错误'})
        role = UserPermission.query.filter_by(user_id=user_id).first()
        if role is None or role.permission_level != 0:
            return jsonify({'msg': '用户权限错误'})
        feed = Feedback.query.filter_by(id=deleteid).first()
        if feed is None:
            return jsonify({'msg': '用户不存在'})
        else:
            db.session.delete(feed)
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                db.session.flush()
                print(e)
                return jsonify({'msg': '请稍后再试'})

            return jsonify({'msg': 'success'})


@admin_page.route('/admingetfeedback/<feedbackid>', methods=['GET'])
def getFeedback(feedbackid):
    if request.method == 'GET':
        try:
            user_id = str(request.cookies.get('user_id'))
            user_session = str(session.get('user_session'))
        except Exception as e:
            print(e)
            return jsonify({'msg': '用户错误'})
        role = UserPermission.query.filter_by(user_id=user_id).first()
        if role is None or role.permission_level != 0:
            return jsonify({'msg': '用户权限错误'})
        feed = Feedback.query.filter_by(id=feedbackid).first()
        if feed is None:
            return jsonify({'msg': '反馈错误'})
        return jsonify({'msg': 'success', 'data': {
            'userid': feed.user_id,
            'content': feed.feedback_content,
        }})


@admin_page.route('/submitfeed/', methods=['POST'])
def submitFeedback():
    if request.method == 'POST':
        data = request.json
        try:
            user_id = str(request.cookies.get('user_id'))
            user_session = str(session.get('user_session'))
        except Exception as e:
            print(e)
            return jsonify({'msg': '用户错误'})
        role = UserPermission.query.filter_by(user_id=user_id).first()
        if role is None or role.permission_level != 0:
            return jsonify({'msg': '用户权限错误'})
        feedid = data['feedbackId']
        feed = Feedback.query.filter_by(id=feedid).first()
        userid = feed.user_id
        user = User.query.filter_by(id=userid, is_deleted=False).first()
        reply_content = data['replyContent']
        try:
            reply_feedback(user.email, reply_content)
            db.session.delete(feed)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            db.session.flush()
            print(e)
            return jsonify({'msg': '请稍后再试'})
        return jsonify({'msg': 'success'})
