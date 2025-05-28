from App.exts import db


# 定义用户模型类
class User(db.Model):
    __tablename__ = 'users'
    # 用户ID (UserID) [主键]: 整数型 (INT)，自增长，唯一标识用户
    # 用户名 (Username): 字符串型 (VARCHAR)，存储用户的用户名，长度限制可根据需求设定
    # 昵称 (Nickname): 字符串型 (VARCHAR)，存储用户的昵称，长度限制可根据需求设定
    # 电子邮箱 (Email): 字符串型 (VARCHAR)，存储用户的电子邮箱地址
    # 密码 (Password): 字符串型 (VARCHAR)，存储用户的密码，通常应加密存储
    # 是否删除 (Is_deleted): 字符串型 (BOOLEAN)，存储用户的状态
    # 创建时间 (CreatedAt): 时间戳 (TIMESTAMP)，记录用户账号的创建时间
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    nickname = db.Column(db.String(80), unique=False, nullable=False)
    email = db.Column(db.String(120), unique=False, nullable=True)
    password = db.Column(db.String(128), nullable=False)
    is_deleted = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return self.username


# 定义图片集模型类
class ImageCollection(db.Model):
    __tablename__ = 'image_collections'

    # 图片集ID(CollectionID)[主键]: 整数型(INT)，自增长，唯一标识图片集
    # 用户ID(UserID): 整数型(INT)，外键关联用户表的用户ID，指示图片集所属用户
    # 图片数量(ImageCount): 整数型(INT)，记录图片集中包含的图片数量
    # 是否删除 (Is_deleted): 字符串型 (BOOLEAN)，存储的状态
    # 创建时间(CreatedAt): 时间戳(TIMESTAMP)，记录图片集的创建时间
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    image_count = db.Column(db.Integer, nullable=False)
    is_deleted = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.String(80), nullable=False)


# 定义图片表模型类
class Image(db.Model):
    __tablename__ = 'images'

    # 图片ID(ImageID)[主键]: 整数型(INT)，自增长，唯一标识图片
    # 图片路径(ImagePath): 字符串型(VARCHAR)，存储图片的路径或文件名
    # 图片集ID(CollectionID): 整数型(INT)，外键关联图片集表的图片集ID，指示图片所属图片集
    # 识别数据ID(DataID): 整数型(INT)，外键关联图片数据表的数据ID，指示图片对应的识别数据
    # 是否删除 (Is_deleted): 字符串型 (BOOLEAN)，存储的状态
    # 创建时间(CreatedAt): 时间戳(TIMESTAMP)，记录图片的创建时间
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    image_collection_id = db.Column(db.Integer, nullable=True)
    data_id = db.Column(db.Integer, nullable=True)
    is_deleted = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'<Image {self.id}>'


# 定义图片数据表模型类
class ImageData(db.Model):
    __tablename__ = 'image_data'
    # 数据ID(DataID)[主键]: 整数型(INT)，自增长，唯一标识识别数据
    # 数据内容(DataContent): 文本型(TEXT)，存储图片识别后的数据，采用JSON格式
    # 创建时间(CreatedAt): 时间戳(TIMESTAMP)，记录数据的创建时间
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    img_id = db.Column(db.Integer, nullable=False)
    data_content = db.Column(db.Text, nullable=False)
    is_deleted = db.Column(db.Boolean, default=False, nullable=False)  # 添加 is_deleted 字段
    created_at = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'<ImageData {self.id}>'


# 定义用户权限表模型类
class UserPermission(db.Model):
    __tablename__ = 'user_permissions'

    # 权限ID(PermissionID)[主键]: 整数型(INT)，自增长，唯一标识权限
    # 用户ID(UserID): 整数型(INT)，外键关联用户表的用户ID，指示权限所属用户
    # 权限级别(PermissionLevel): 整数型(INT)，存储用户的权限级别，例如管理员权限、普通用户权限等
    # 创建时间(CreatedAt): 时间戳(TIMESTAMP)，记录权限的创建时间
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    permission_level = db.Column(db.Integer, nullable=False)
    is_deleted = db.Column(db.Boolean, default=False, nullable=False)  # 添加 is_deleted 字段
    created_at = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'<UserPermission {self.id}>'


# 定义历史记录表模型类
class History(db.Model):
    __tablename__ = 'history'
    # 记录ID(RecordID)[主键]: 整数型(INT)，自增长，唯一标识历史记录
    # 用户ID(UserID): 整数型(INT)，外键关联用户表的用户ID，记录操作的用户
    # 操作内容(Action): 字符串型(VARCHAR)，记录操作的内容或描述
    # 操作时间(ActionTime): 时间戳(TIMESTAMP)，记录操作的时间
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    action = db.Column(db.String(255), nullable=False)
    action_time = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'<History {self.id}>'


# 定义错误日志表模型类
class ErrorLog(db.Model):
    __tablename__ = 'error_logs'
    # 日志ID(LogID)[主键]: 整数型(INT)，自增长，唯一标识错误日志
    # 错误内容(ErrorMessage): 文本型(TEXT)，记录错误的详细信息
    # 发生时间(ErrorTime): 时间戳(TIMESTAMP)，记录错误发生的时间

    id = db.Column(db.Integer, primary_key=True)
    error_message = db.Column(db.Text, nullable=False)
    error_time = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'<ErrorLog {self.id}>'


# 定义反馈表模型类
class Feedback(db.Model):
    __tablename__ = 'feedback'
    # 反馈ID(FeedbackID)[主键]: 整数型(INT)，自增长，唯一标识反馈
    # 用户ID(UserID): 整数型(INT)，外键关联用户表的用户ID，记录反馈的用户
    # 反馈内容(FeedbackContent): 文本型(TEXT)，记录用户的反馈内容
    # 反馈时间(FeedbackTime): 时间戳(TIMESTAMP)，记录反馈的时间

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    feedback_content = db.Column(db.Text, nullable=False)
    feedback_time = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f'<Feedback {self.id}>'
