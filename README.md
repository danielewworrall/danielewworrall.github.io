# danielewworrall.github.io

Personal academic website for **Daniel Worrall** — scientific machine learning, Digital Twins Lead at [CuspAI](https://www.cusp.ai/).

🔗 **[danielewworrall.github.io](https://danielewworrall.github.io)**

## Stack

- Pure HTML, CSS, and vanilla JavaScript (no frameworks)
- Main pages set like a NeurIPS paper in self-hosted [TeX Gyre](https://www.gust.org.pl/projects/e-foundry/tex-gyre) Termes and Heros (`fonts/`, GUST Font License)
- Blog posts in a margin-notes layout set in [EB Garamond](https://fonts.google.com/specimen/EB+Garamond)
- Light and dark themes: follows the system, with a toggle saved in `localStorage`
- [MathJax 3](https://www.mathjax.org/) for LaTeX rendering in blog posts
- Responsive design

## Local development

```bash
python3 -m http.server 8765
# Open http://localhost:8765
```

## Structure

```
index.html          # Homepage
publications.html   # Publications
talks.html          # Talks
teaching.html       # Teaching
404.html            # Not-found page
style.css           # Design system (CSS custom properties)
script.js           # Theme toggle, active nav link, code highlighting, citation counts
fonts/              # TeX Gyre Termes and Heros (woff2 subsets) and their licence
images/             # Headshot and selected-paper figures
blog/               # Blog posts (static HTML)
media/              # Blog post images (SVGs, etc.)
docs/               # Lecture and talk slides (PDF)
publications/, talks/, teaching/  # Redirects from old URLs
```
