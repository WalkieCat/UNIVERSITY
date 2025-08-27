// Script js code
var body = document.querySelector('Body');
body.style.backgroundColor = 'lightgreen';

var head = document.querySelector('head');
var viewport = '<meta name="viewport" content="width=device-width, initial-scale=1.0">';
head.innerHTML += viewport; 

var cssLink = '<link rel="stylesheet" type="text/css" href="./css/flex_style.css" />';
head.innerHTML += cssLink;

var article = document.createElement('article');
article.className = 'flex-container';
body.appendChild(article);

var divHTML1 =
    '<div class="item">' +
        ' <figure class="relative-container">' +
            ' <img class="same-size" src="./image/chatgpt.jpg" alt="chat gpt">' +
            ' <figcaption class="inner-text">Chat GPT</figcaption>' +
        ' </figure>' +
    '</div>';
article.innerHTML += divHTML1;

var divHTML2 =
    '<div class="item">' +
        ' <figure class="relative-container">' +
            ' <img class="same-size" src="./image/gemini.jpg" alt="gemini">' +
            ' <figcaption class="inner-text">Gemini</figcaption>' +
        ' </figure>' +
    '</div>';
article.innerHTML += divHTML2;

var divHTML3 = '<div class="item"></div>';
article.innerHTML += divHTML3;

var divHTML4 = '<div class="item"></div>';
article.innerHTML += divHTML4;

var divHTML5 = '<div class="item">Responsive design with Flexbox</div>';
article.innerHTML += divHTML5;