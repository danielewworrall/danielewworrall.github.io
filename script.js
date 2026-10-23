/* ============================================================
   script.js — Daniel Worrall Academic Site
   Theme toggle, active nav link, code highlighting, citations
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

  // Load CSS
  const theme = document.createElement('link');
  theme.rel = 'stylesheet';
  theme.href = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.min.css';
  document.head.appendChild(theme);

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

// ── Citation Counts (Semantic Scholar) ───────────────────────
(function loadCitations() {
  const cards = document.querySelectorAll('.pub-card');
  if (!cards.length) return;

  const AUTHOR_ID = '3471551'; // Daniel E. Worrall
  const API = `https://api.semanticscholar.org/graph/v1/author/${AUTHOR_ID}/papers?fields=title,citationCount,url&limit=100`;

  function normalise(s) {
    return s.toLowerCase().replace(/[^a-z0-9]/g, '');
  }

  fetch(API)
    .then(r => r.json())
    .then(data => {
      const papers = data.data || [];
      cards.forEach(card => {
        const titleEl = card.querySelector('.pub-card__title');
        if (!titleEl) return;
        const cardTitle = normalise(titleEl.textContent);

        // Find best match
        let best = null, bestScore = 0;
        papers.forEach(p => {
          const pTitle = normalise(p.title);
          // Check if one contains the other (handles slight differences)
          const shorter = cardTitle.length < pTitle.length ? cardTitle : pTitle;
          const longer = cardTitle.length < pTitle.length ? pTitle : cardTitle;
          if (longer.includes(shorter) && shorter.length > 10) {
            const score = shorter.length / longer.length;
            if (score > bestScore) { bestScore = score; best = p; }
          }
        });

        if (best && best.citationCount > 0) {
          let linksRow = card.querySelector('.pub-card__links');
          if (!linksRow) {
            linksRow = document.createElement('div');
            linksRow.className = 'pub-card__links';
            card.appendChild(linksRow);
          }
          const badge = document.createElement('a');
          badge.className = 'pub-card__cite-badge';
          badge.href = best.url || '#';
          badge.target = '_blank';
          badge.textContent = `Cited by ${best.citationCount}`;
          linksRow.appendChild(badge);
        }
      });
    })
    .catch(() => { }); // fail silently
})();
