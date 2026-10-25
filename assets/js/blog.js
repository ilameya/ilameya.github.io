// Small enhancements for blog posts: copy buttons on code blocks and "Try it" callouts.
document.querySelectorAll('.content div.highlight').forEach(function (block) {
  var btn = document.createElement('button');
  btn.className = 'copy-btn';
  btn.type = 'button';
  btn.textContent = 'Copy';
  btn.addEventListener('click', function () {
    var code = block.querySelector('pre').innerText;
    navigator.clipboard.writeText(code).then(function () {
      btn.textContent = 'Copied';
      setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
    });
  });
  block.appendChild(btn);
});

document.querySelectorAll('.content blockquote').forEach(function (q) {
  var first = q.querySelector('strong');
  if (first && /^(Try it|Exercise)/.test(first.textContent)) q.classList.add('try');
});

// Click any figure to open it full size (useful for wide diagrams on phones).
document.querySelectorAll('.content img').forEach(function (img) {
  img.addEventListener('click', function () { window.open(img.src, '_blank', 'noopener'); });
});
