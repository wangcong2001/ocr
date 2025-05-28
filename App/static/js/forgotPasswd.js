document.addEventListener('DOMContentLoaded', function () {
        const resetPasswordForm = document.getElementById('forgotPasswordForm');
        const errorMsg = document.getElementById('errorMsg');

        resetPasswordForm.addEventListener('submit', function (event) {
            event.preventDefault(); // 阻止表单默认提交行为
            function hashPasswordWithCryptoJS(password) {
                // 使用CryptoJS的SHA256进行哈希
                const hash = CryptoJS.SHA256(password);
                // 转换为十六进制字符串
                const hashHex = hash.toString(CryptoJS.enc.Hex);
                return hashHex;
            }
            // 获取表单数据
            const formData = new FormData(resetPasswordForm);

            // 确认新密码是否匹配
            const newPassword = formData.get('newPassword');
            const confirmNewPassword = formData.get('confirmNewPassword');
            if (newPassword !== confirmNewPassword) {
                errorMsg.textContent = '新密码与确认新密码不匹配';
                return;
            }
            const hashedPassword = hashPasswordWithCryptoJS(newPassword);
            formData.set('newPassword', hashedPassword);
            console.log(newPassword)

            // 向后端发送 POST 请求
            fetch('/forgotPasswd/', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('网络链接失败');
                }
                return response.json(); // 将响应解析为 JSON 对象
            })
            .then(data => {
                console.log(data)
                // 根据后端返回的 JSON 数据执行相应操作
                if (data.msg === 'success') {
                    // 重置密码成功
                    alert('密码重置成功！');
                    window.location.href = '/login/'
                } else {
                    // 重置密码失败，显示错误消息
                    errorMsg.textContent = data.msg;
                }
            })
            .catch(error => {
                console.error('错误:', error);
            });
        });
    });

    document.addEventListener('DOMContentLoaded', function () {
        const sendCodeBtn = document.getElementById('sendCodeBtn');
        const emailInput = document.getElementById('email');
        const usernameInput = document.getElementById('username');
        let countdown = 60; // 倒计时时长，单位为秒

        sendCodeBtn.addEventListener('click', function () {
            // 验证邮箱是否为空
            if (!emailInput.value.trim()) {
                alert('请输入邮箱');
                return;
            }else if (!validateEmail(emailInput.value.trim())) {
                alert('请输入有效的邮箱地址');
                return;
            }
            if (!usernameInput.value.trim()) {
                alert('请输入用户名');
                return;
            }

            // 禁用按钮防止重复点击
            sendCodeBtn.disabled = true;

            // 发送请求到后端
            fetch('/sendEmail/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email: emailInput.value, username:usernameInput.value })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('网络链接失败');
                }
                return response.json(); // 将响应解析为 JSON 对象


            })
            .then(data =>{
                console.log(data)
                if(data.msg === 'success'){
                    startCountdown();
                    countdown = 60
                }
                else {
                    alert(data.msg);
                    window.location.reload()
                }

            })
            .catch(error => {
                console.error('发送验证码失败:', error);
            });
        });

        // 验证邮箱格式
        function validateEmail(email) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return emailRegex.test(email);
        }

        // 倒计时函数
        function startCountdown() {
            sendCodeBtn.classList.add('btnDisabled');
            sendCodeBtn.disabled = true; // 可以在这里明确地再次禁用按钮，双重保险
            const timer = setInterval(function () {
                if (countdown <= 1) {
                    clearInterval(timer);
                    sendCodeBtn.innerText = '发送验证码';
                    sendCodeBtn.classList.remove('btnDisabled');
                    sendCodeBtn.disabled = false; // 恢复按钮可点击状态

                    // 强制触发按钮点击事件
                    // {#sendCodeBtn.onclick();#}

                    // 或者尝试手动触发点击事件
                    // sendCodeBtn.dispatchEvent(new Event('click'));
                } else {
                    sendCodeBtn.innerText = `${countdown - 1}秒`;
                    countdown--;
                }
            }, 1000);
        }
    });