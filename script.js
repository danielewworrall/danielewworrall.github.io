/* ============================================================
   script.js — Daniel Worrall Academic Site
   Neural-net canvas, theme toggle, scroll reveals, mobile menu
   ============================================================ */

// ── Neural Network Canvas Background ─────────────────────────
(function initCanvas() {
  const canvas = document.getElementById('neural-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let width, height;
  const nodes = [];
  const NODE_COUNT = 60;
  const CONNECT_DIST = 150;
  let mouse = { x: -1000, y: -1000 };
  let animId;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }

  function initNodes() {
    nodes.length = 0;
    for (let i = 0; i < NODE_COUNT; i++) {
      nodes.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        r: Math.random() * 2 + 1,
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const nodeColor = isLight ? 'rgba(79, 70, 229,' : 'rgba(99, 102, 241,';
    const lineColor = isLight ? 'rgba(79, 70, 229,' : 'rgba(99, 102, 241,';

    // Update positions
    for (const node of nodes) {
      node.x += node.vx;
      node.y += node.vy;

      if (node.x < 0 || node.x > width) node.vx *= -1;
      if (node.y < 0 || node.y > height) node.vy *= -1;

      // Mouse repulsion
      const dx = node.x - mouse.x;
      const dy = node.y - mouse.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 120) {
        node.vx += dx / dist * 0.15;
        node.vy += dy / dist * 0.15;
      }

      // Dampen velocity
      node.vx *= 0.999;
      node.vy *= 0.999;
    }

    // Draw connections
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONNECT_DIST) {
          const alpha = (1 - dist / CONNECT_DIST) * 0.35;
          ctx.beginPath();
          ctx.strokeStyle = lineColor + alpha + ')';
          ctx.lineWidth = 0.5;
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (const node of nodes) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
      ctx.fillStyle = nodeColor + '0.6)';
      ctx.fill();
    }

    animId = requestAnimationFrame(draw);
  }

  window.addEventListener('resize', () => {
    resize();
  });

  document.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });

  resize();
  initNodes();
  draw();
})();


// ── Theme Toggle ─────────────────────────────────────────────
(function initTheme() {
  const toggle = document.getElementById('theme-toggle');
  if (!toggle) return;

  const stored = localStorage.getItem('theme');
  if (stored) {
    document.documentElement.setAttribute('data-theme', stored);
  }
  updateIcon();

  toggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateIcon();
  });

  function updateIcon() {
    const theme = document.documentElement.getAttribute('data-theme');
    toggle.textContent = theme === 'light' ? '🌙' : '☀️';
  }
})();


// ── Scroll Reveal ────────────────────────────────────────────
(function initReveal() {
  const reveals = document.querySelectorAll('.reveal');
  if (!reveals.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -40px 0px',
  });

  reveals.forEach((el) => observer.observe(el));
})();


// ── Navbar Scroll Effect ─────────────────────────────────────
(function initNavScroll() {
  const nav = document.querySelector('.nav');
  if (!nav) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 20) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }
  }, { passive: true });
})();


// ── Mobile Hamburger Menu ────────────────────────────────────
(function initMobileMenu() {
  const hamburger = document.getElementById('hamburger');
  const navLinks = document.getElementById('nav-links');
  if (!hamburger || !navLinks) return;

  hamburger.addEventListener('click', () => {
    navLinks.classList.toggle('open');
    hamburger.classList.toggle('active');
  });

  // Close menu when clicking a link
  navLinks.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('open');
      hamburger.classList.remove('active');
    });
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
