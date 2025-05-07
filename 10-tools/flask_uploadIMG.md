- [目录结构](#目录结构)
- [app.py 代码解析](#apppy-代码解析)
  - [基础模块导入](#基础模块导入)
  - [应用配置](#应用配置)
  - [文件类型验证](#文件类型验证)
  - [文件名生成](#文件名生成)
  - [核心路由逻辑](#核心路由逻辑)
  - [运行配置](#运行配置)
- [emplates/upload.html](#emplatesuploadhtml)
- [运行](#运行)



# 目录结构

```
/project
  ├── app.py
  ├── uploads/          # 图片存储目录（自动创建）
  └── templates/
      └── upload.html
```


# app.py 代码解析

## 基础模块导入

```
from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
```

- ​Flask：核心框架模块
- ​render_template：用于渲染HTML模板（对应templates/upload.html）
- ​request：处理HTTP请求对象
- ​redirect/url_for：实现页面跳转和路由反向解析
- ​flash：消息提示功能（需配合secret_key使用）
- ​secure_filename：安全文件名处理，防止路径遍历攻击

## 应用配置

```
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于消息提示
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制16MB文件大小
```

- ​secret_key：启用flash消息和session功能必须的密钥
- ​UPLOAD_FOLDER：定义上传文件存储路径
- ​MAX_CONTENT_LENGTH：安全限制文件体积（超过会触发413错误）


## 文件类型验证

```
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

- 安全策略：双重验证文件扩展名（需注意需配合MIME类型验证更安全）
- ​filename.rsplit：从右分割扩展名，确保处理多后缀文件名
- ​lower()：统一转为小写避免大小写绕过验证

## 文件名生成

```
import uuid

def random_filename(filename):
    ext = os.path.splitext(filename)[1]
    return f"{uuid.uuid4().hex}{ext}"
```

- uuid4：生成唯一文件名防止重复（比时间戳更可靠）
- ​os.path.splitext：分离原文件扩展名
- ​优势：解决中文文件名问题 + 防止文件名冲突


## 核心路由逻辑

```
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # POST请求处理
    if request.method == 'POST':
        # 验证请求中是否存在文件
        if 'file' not in request.files:
            flash('未选择文件')
            return redirect(request.url)
        
        file = request.files['file']
        # 验证文件名非空
        if file.filename == '':
            flash('未选择文件')
            return redirect(request.url)
        
        # 文件类型验证
        if file and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # 如果路径不存在则创建对应路径
            # 安全处理文件名
            filename = random_filename(secure_filename(file.filename))
            # 保存文件
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash(f'文件已上传！文件名：{filename}')
            return redirect(url_for('upload_file'))
        else:
            flash('仅支持PNG/JPG/JPEG/GIF格式')
    # GET请求返回上传页面
    return render_template('upload.html')
```

1. 流程控制：完整处理文件上传全流程（验证->处理->反馈）
2. ​安全措施：三级验证（文件存在性/文件名非空/文件类型）
3. ​文件保存：组合使用安全文件名和UUID命名策略
4. ​用户反馈：通过flash消息实现交互提示


## 运行配置

```
if __name__ == '__main__':
    app.run(debug=True)
```

- 调试模式：开发阶段开启debug自动重载（生产环境需关闭）
- ​服务启动：默认启动在127.0.0.1:5000

# emplates/upload.html

```
<!-- templates/upload.html -->
<!DOCTYPE html>
<html>
<head>
    <title>图片上传</title>
    <style>
        .container { max-width: 600px; margin: 20px auto; }
        .preview { max-width: 200px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>上传图片</h2>
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                <div class="alert">
                    {% for message in messages %}
                        {{ message }}
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}

        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" id="file" required>
            <input type="submit" value="上传">
        </form>

        <!-- 实时预览 -->
        <img id="preview" class="preview" src="#" alt="图片预览">
    </div>

    <script>
        // 实时预览功能
        document.getElementById('file').onchange = function(evt) {
            const [file] = evt.target.files
            if (file) {
                document.getElementById('preview').src = URL.createObjectURL(file)
            }
        }
    </script>
</body>
</html>
```

# 运行

**推荐使用虚拟环境隔离项目依赖**

```
python -m venv venv
# macOS/Linux
source venv/bin/activate  
# Windows
venv\Scripts\activate
pip install flask
```
