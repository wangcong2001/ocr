document.addEventListener('DOMContentLoaded', function () {
    const feedbackForm = document.getElementById('feedbackForm');
    const errorMsg = document.getElementById('errorMsg');

    feedbackForm.addEventListener('submit', function (event) {
        event.preventDefault(); // 阻止表单默认提交行为

        const formData = new FormData(feedbackForm);
        const feedbackText = formData.get('feedback');

        fetch('/feedback/', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('网络链接错误');
            }
            return response.json(); // 将响应解析为 JSON 对象
        })
        .then(data => {
            if (data.msg === 'success') {
                // 如果反馈成功，执行相应操作（例如显示成功消息）
                // {#console.log('提交成功');#}
                window.location.href='/'
            } else {
                // 如果反馈失败，显示错误消息
                errorMsg.textContent = data.msg;
            }
        })
        .catch(error => {
            console.error('错误:', error);
        });
    });
});