document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('loginForm');
    const errorMsg = document.getElementById('errorMsg');

    form.addEventListener('submit', function (event) {
        event.preventDefault(); // 阻止表单默认提交行为
        function hashPasswordWithCryptoJS(password) {
            // 使用CryptoJS的SHA256进行哈希
            const hash = CryptoJS.SHA256(password);
            // 转换为十六进制字符串
            const hashHex = hash.toString(CryptoJS.enc.Hex);
            return hashHex;
        }
        const formData = new FormData(form);
        const password = formData.get('password');
        const hashedPassword = hashPasswordWithCryptoJS(password); // 假设hashPassword是异步的
        console.log(password)
        console.log(hashedPassword)
        formData.set('password', hashedPassword); // 将原始密码替换为哈希值

        fetch('/login/', {
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
                // 如果登录成功，设置 Cookie，并执行页面跳转
                document.cookie = `user_id=${data.id}; max-age=${7*24*3600}; path=/`;
                window.location.href = '/'; // 页面跳转到首页
            } else {
                // 如果登录失败，显示错误消息
                errorMsg.textContent = data.msg;
            }
        })
        .catch(error => {
            console.error('错误:', error);
            // 可以在这里处理其他错误情况
        });
    });
});