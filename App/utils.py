import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
import random
import re
import hashlib

from OCR.myocr import PytorchOcr

pytorchocr = PytorchOcr()

# 设置SMTP服务器地址和端口
smtp_server = 'smtp.163.com'
smtp_port = 465  # 对于163邮箱，通常使用SSL加密的465端口

# 发件人信息
sender_email = 'shiguang12424@163.com'
sender_password = 'BBCUOXFWRIJESYVL'  # 这里填写SMTP授权码
sender_name = '图片文字识别系统官方'
word_list = ['此', '在', '樟', '闲']


def generate_verification_code(length=6):
    return ''.join(str(random.randint(0, 9)) for _ in range(length))


def send_email(receiver, code):
    # 收件人信息
    receiver_email = receiver

    # 创建邮件主体
    message = MIMEMultipart()
    message['From'] = formataddr((Header(sender_name, 'utf-8').encode(), sender_email))
    message['To'] = receiver_email
    message['Subject'] = Header('图片文字识别验证码', 'utf-8')

    # 邮件正文
    content = '你的图片文字识别系统账号验证码：' + code
    print(content)
    message.attach(MIMEText(content, 'plain', 'utf-8'))
    # 创建SMTP连接并登录
    smtp_obj = smtplib.SMTP_SSL(smtp_server, smtp_port)
    smtp_obj.login(sender_email, sender_password)
    # 发送邮件
    smtp_obj.sendmail(sender_email, [receiver_email], message.as_string())
    smtp_obj.quit()


def reply_feedback(receiver, content):
    # 收件人信息

    receiver_email = receiver
    # 创建邮件主体
    message = MIMEMultipart()
    message['From'] = formataddr((Header(sender_name, 'utf-8').encode(), sender_email))
    message['To'] = receiver_email
    message['Subject'] = Header('图片文字识别系统 反馈回复', 'utf-8')

    # 邮件正文
    content = '尊敬的用户： 你好\n' + content
    print(content)
    message.attach(MIMEText(content, 'plain', 'utf-8'))
    # 创建SMTP连接并登录
    smtp_obj = smtplib.SMTP_SSL(smtp_server, smtp_port)
    smtp_obj.login(sender_email, sender_password)
    # 发送邮件
    smtp_obj.sendmail(sender_email, [receiver_email], message.as_string())
    smtp_obj.quit()


def rec(img_path):
    img_path = img_path
    ret = pytorchocr.predict(img_path)
    return ret


def check(errors):
    for error in errors:
        if error[0] in word_list:
            return True
    return False


def contains_chinese(text):
    """判断文本是否包含汉字"""
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(pattern.search(text))


def hash_password_with_sha256(password):
    # 使用hashlib库的sha256进行哈希
    hash_object = hashlib.sha256()
    hash_object.update(password.encode('utf-8'))  # 确保密码被编码为字节串
    # 转换为十六进制字符串
    hash_hex = hash_object.hexdigest()
    return hash_hex
