let img_data_list=[]
// {#批量识别#}
document.addEventListener('DOMContentLoaded', function () {
const recognizeAllBtn = document.getElementById('recognizeAllBtn');

recognizeAllBtn.addEventListener('click', function () {
    // 显示所有 ImgItem 上的加载动画
    const imgItems = document.querySelectorAll('.ImgItem');
    imgItems.forEach(function (item) {
        // 在每个ImgItem内部创建loadingSpinner
        const loadingSpinner = document.createElement('div');
        const spinner = document.createElement('div');
        loadingSpinner.classList.add('loadingSpinner');
        spinner.classList.add('spinner');
        item.appendChild(loadingSpinner);
        loadingSpinner.appendChild(spinner)

    });
    imgItems.forEach(imgItem => {
        const loadingSpinner = imgItem.querySelector('.loadingSpinner');
        // console.log(loadingSpinner)
        loadingSpinner.style.display = 'block';
    });
    recognizeImagesOneByOne(imgItems, 0);
});

function recognizeImagesOneByOne(imgItems, index) {
    if (index < imgItems.length) {
        const imgItem = imgItems[index];
        const loadingSpinner = imgItem.querySelector('.loadingSpinner');
        loadingSpinner.style.display = 'block';

        const imageData = img_data_list[index];
        recognizeImage(imageData)
            .then(() => {
                // 隐藏加载动画
                loadingSpinner.style.display = 'none';
                // 处理下一张图片
                recognizeImagesOneByOne(imgItems, index + 1);
            })
            .catch(error => {
                console.error(error);
                // 处理请求失败的情况
                loadingSpinner.style.display = 'none';
                // 处理下一张图片
                recognizeImagesOneByOne(imgItems, index + 1);
            });
    }
}

function recognizeImage(imageData) {
    return new Promise((resolve, reject) => {
        fetch(`/recognize?id=${imageData.id}`, {
            method: 'GET'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('网络链接失败');
            }
            return response.json();
        })
        .then(data => {
            if (data.msg === 'success') {
                // 更新对应ID的img_text
                imageData.img_text = data.data;
                console.log(imageData)
                // 更新ImgTxt显示
                const imgTxtDiv = document.querySelector('.ImgTxt');
                imgTxtDiv.textContent = imageData.img_text;
                resolve();
            } else {
                reject(new Error(`识别图片 ${imageData.id} 失败: ${data.error}`));
            }
        })
        .catch(error => {
            reject(new Error(`识别图片 ${imageData.id} 失败: ${error.message}`));
        });
    });
}
});
// {#单个识别#}
document.addEventListener('DOMContentLoaded', function () {
    const recognizeBtn = document.getElementById('recognizeBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');

    recognizeBtn.addEventListener('click', function () {
        // 显示加载动画
        loadingSpinner.style.display = 'block';

        // 获取当前展示图片的ID
        const imgBox = document.getElementById('selectedImg');
        const imgSrc = imgBox.style.backgroundImage.replace('url("', '').replace('")', '');
        if (imgSrc === '')
        {
            alert('请选图片')
            loadingSpinner.style.display = 'none';
            return ;
        }
        const currentImgData = img_data_list.find(item => item.image_path === imgSrc);
        const currentImgId = currentImgData.id;


        // 向后端发送识别请求并更新img_text
        fetch(`/recognize?id=${currentImgId}`, {
            method: 'GET'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('网络链接失败');
            }
            return response.json();
        })
        .then(data => {
            if (data.msg === 'success') {
                // 更新对应ID的img_text
                currentImgData.img_text = data.data;
                // 更新ImgTxt显示
                const imgTxtDiv = document.querySelector('.ImgTxt');
                imgTxtDiv.textContent = currentImgData.img_text;
            }
        })
        .catch(error => {
            console.error('识别请求失败:', error);
            // 处理请求失败的情况
        })
        .finally(() => {
            // 隐藏加载动画
            loadingSpinner.style.display = 'none';
        });
    });
});
// {#批量上传#}
document.addEventListener('DOMContentLoaded', function () {
    const uploadBtn = document.getElementById('uploadBtnmul');
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.multiple = true; // 允许选择多个文件
    fileInput.style.display = 'none';

    uploadBtn.addEventListener('click', function () {
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        const files = fileInput.files;

        if (files.length === 0) {
            alert('请选择要上传的图片！');
            return;
        }
        // console.log(files)
        const formData = new FormData();

        for (let i = 0; i < files.length; i++) {
            const encodedFileName = encodeURIComponent(files[i].name);
            formData.append('files[]', files[i], encodedFileName);
            // console.log(files[i].name, encodedFileName)
        }

        // 发送文件到后端
        fetch('/uploadmul/', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('网络链接失败');
            }
            return response.json();
        })
        .then(data => {
            // console.log(data)
            if (Array.isArray(data)) { // 假设后端返回的是一个包含每个图片上传结果的数据数组
                data.forEach(item => {
                    // console.log(item)
                    if (item.msg === 'success') {
                        // 处理每个上传成功的图片数据
                        let imageData = {
                            msg: item.msg,
                            id: item.data.id,
                            image_path: item.data.image_path.replace(/\\/g, '/'),
                            img_text: item.data.img_text
                        };
                        img_data_list.push(imageData)
                        // console.log(1,imageData)
                        // 创建 ImgItem 和 img 元素并添加到页面中
                        const ImgItem = document.createElement('div');
                        ImgItem.classList.add('ImgItem');
                        const img = document.createElement('img');
                        img.classList.add('imgsize');
                        img.src = imageData.image_path;
                        ImgItem.appendChild(img);
                        const ImgList = document.querySelector('.ImgList');
                        ImgList.appendChild(ImgItem);
                        // 添加点击事件监听器
                        ImgItem.addEventListener('click', function () {
                            let imgSrc = img.getAttribute('src');
                            imgSrc = imgSrc.replace(/\\/g, '/');
                            const imgBox = document.getElementById('selectedImg');
                            imgBox.style.backgroundImage = `url(${imgSrc})`;
                            const imageData = img_data_list.find(data => data.image_path === imgSrc);
                            // console.log(imageData)
                            const imgTxtDiv = document.querySelector('.ImgTxt');
                            imgTxtDiv.textContent = imageData.img_text;
                            // console.log(imgTxtDiv.textContent)


                        });
                        ImgItem.click()
                    } else {
                        console.error(`图片 ${item.file_name} 上传失败: ${item.error}`);
                    }
                });
            } else {
                console.error('无法处理服务器返回的数据格式');
            }
        })
        .catch(error => {
            console.error('文件上传失败:', error);
        });
    });
});
