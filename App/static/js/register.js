
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('registerForm');
    const errorMsg = document.getElementById('errorMsg');

    function hashPasswordWithCryptoJS(password) {
        // 使用CryptoJS的SHA256进行哈希
        const hash = CryptoJS.SHA256(password);
        // 转换为十六进制字符串
        const hashHex = hash.toString(CryptoJS.enc.Hex);
        return hashHex;
    }

    form.addEventListener('submit',  function (event) {
        event.preventDefault(); // 阻止表单默认提交行为

        const formData = new FormData(form);
        const password = formData.get('password');
        const confirmPassword = formData.get('confirmPassword');
        if (password !== confirmPassword) {
            errorMsg.textContent = '确认密码与密码不一致';
            return;
        }
        const hashedPassword = hashPasswordWithCryptoJS(password);
        formData.set('password', hashedPassword);
        console.log(hashedPassword)
        fetch('/register/', {
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
                if (data.msg === 'success') {
                    // 如果注册成功，执行相应操作
                    document.cookie = `user_id=${data.id}; max-age=${7 * 24 * 3600}; path=/`;
                    window.location.href = '/'; // 页面跳转到首页

                } else {
                    // 如果注册失败，显示错误消息
                    errorMsg.textContent = data.msg;
                }
            })
            .catch(error => {
                console.error('错误:', error);
            });
    });
});