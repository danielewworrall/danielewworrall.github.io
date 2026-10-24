/* ============================================================
   script.js — Daniel Worrall Academic Site
   Theme toggle, active nav link, code highlighting
   ============================================================ */

// ── Theme Toggle ─────────────────────────────────────────────
// The page follows the system theme until the toggle stores a choice.
// Each page's <head> applies a stored choice before first paint.
(function initTheme() {
  const toggle = document.getElementById('theme-toggle');
  if (!toggle) return;
  const root = document.documentElement;
  const dark = () => root.getAttribute('data-theme') === 'dark' ||
    (!root.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);

  function updateLabel() {
    toggle.textContent = dark() ? 'Light' : 'Dark';
    toggle.setAttribute('aria-label', dark() ? 'Switch to light mode' : 'Switch to dark mode');
  }
  updateLabel();
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateLabel);

  toggle.addEventListener('click', () => {
    const next = dark() ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) { }
    updateLabel();
  });
})();


// ── Active Nav Link ──────────────────────────────────────────
(function setActiveLink() {
  const path = window.location.pathname;
  const links = document.querySelectorAll('.nav__links a');
  links.forEach((link) => {
    const href = link.getAttribute('href');
    if (path === href || (href !== '/' && path.startsWith(href))) {
      link.classList.add('active');
    }
  });
})();

// ── Syntax Highlighting (Prism.js) ───────────────────────────
(function loadPrism() {
  if (!document.querySelector('code[class*="language-"]')) return;

  // Token colours live in style.css so they follow the light and dark themes.
  // Load JS + Python language
  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js';
  script.onload = function () {
    const py = document.createElement('script');
    py.src = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js';
    py.onload = function () { Prism.highlightAll(); };
    document.head.appendChild(py);
  };
  document.head.appendChild(script);
})();

// ── Reading Time and Views (blog posts) ──────────────────────
// Reading time from the post's word count, at about 230 words a minute.
(function addReadingTime() {
  const content = document.querySelector('.blog-post__content');
  const meta = document.querySelector('.blog-post__meta');
  if (!content || !meta) return;
  const copy = content.cloneNode(true);
  copy.querySelectorAll('pre, script, style').forEach(el => el.remove());
  const words = copy.textContent.split(/\s+/).filter(Boolean).length;
  const span = document.createElement('span');
  span.className = 'blog-post__reading-time';
  span.textContent = `${Math.max(1, Math.round(words / 230))} min read`;
  const time = meta.querySelector('time');
  time ? time.after(span) : meta.prepend(span);
})();

// ── Page Views (GoatCounter) ─────────────────────────────────
// Cookie-free counts at https://danielewworrall.goatcounter.com. Only the
// live site counts, never the password-protected drafts. Posts show their
// own total once "allow adding visitor counts" is on in GoatCounter.
(function goatCounter() {
  const GC = 'https://danielewworrall.goatcounter.com';
  const path = window.location.pathname;
  if (window.location.hostname !== 'danielewworrall.github.io' || path.startsWith('/drafts/')) return;
  const s = document.createElement('script');
  s.async = true;
  s.src = 'https://gc.zgo.at/count.js';
  s.dataset.goatcounter = `${GC}/count`;
  document.body.appendChild(s);

  const meta = document.querySelector('.blog-post__meta');
  if (!meta) return;
  fetch(`${GC}/counter/${encodeURIComponent(path)}.json`)
    .then(r => (r.ok ? r.json() : null))
    .then(data => {
      if (!data || !data.count) return;
      const span = document.createElement('span');
      span.className = 'blog-post__views';
      span.textContent = `${data.count} ${data.count === '1' ? 'view' : 'views'}`;
      const after = meta.querySelector('.blog-post__reading-time') || meta.querySelector('time');
      after ? after.after(span) : meta.append(span);
    })
    .catch(() => { }); // fail silently
})();
