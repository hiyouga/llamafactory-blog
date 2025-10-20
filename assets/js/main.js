// 全局 JavaScript 代码
// 这个文件会在每个页面加载时执行

console.log('LlamaFactory Blog 已加载');

const footer = document.getElementsByClassName('footer')[0];
footer.getElementsByTagName('span').forEach(span => {
   if(span.innerText.includes('Hugo')) {
    span.style.display = 'none';
   }

});

