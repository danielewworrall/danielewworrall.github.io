# danielewworrall.github.io

Personal academic website for **Daniel Worrall** — Research Scientist at Google DeepMind.

🔗 **[danielewworrall.github.io](https://danielewworrall.github.io)**

## Stack

- Pure HTML, CSS, and vanilla JavaScript (no frameworks)
- [Crimson Pro](https://fonts.google.com/specimen/Crimson+Pro) serif + [Inter](https://fonts.google.com/specimen/Inter) sans-serif typography
- Dark/light theme toggle with `localStorage` persistence
- [MathJax 3](https://www.mathjax.org/) for LaTeX rendering in blog posts
- Animated canvas background
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
style.css           # Design system (CSS custom properties)
script.js           # Animations, theme toggle, nav
blog/               # Blog posts (static HTML)
media/              # Blog post images (SVGs, etc.)
```
