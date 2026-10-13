<!-- self-describing: v1 -->

# danielewworrall.github.io

Daniel Worrall's personal academic website, live at
https://danielewworrall.github.io. Hand-written static HTML, served by
GitHub Pages from `main`.

## Purpose

A current, accurate home page for Daniel's research: role and bio, news,
publications and patents, talks, teaching, students and a small technical
blog. Success is that the live site says what is true now and nothing
unpublished or private is served.

## Setup

None. No dependencies or build step to run locally. Python 3 is enough to
preview.

## Run and test

```sh
python3 -m http.server 8765    # then open http://localhost:8765
```

There are no automated tests. Check a change by opening the edited pages at
desktop and phone widths (headless Chromium works) in light and dark mode,
and checking internal links resolve.

Deploy is merging to `main`: GitHub Pages runs its Jekyll build (the
"pages build and deployment" workflow, about 30 s) and publishes. A failed
build leaves the previous version live. To preview the real build locally:
`gem install github-pages`, then `jekyll build` and inspect `_site/` (do not
commit `_site/`).

## Entry points

| Path | What it is |
|---|---|
| `index.html` | Home page: hero, blog cards, news, bio timeline, students. |
| `publications.html`, `talks.html`, `teaching.html` | The other top-level pages. |
| `blog/index.html`, `blog/YYYY/MM/<slug>/index.html` | Blog index and posts (MathJax 3.2.2 for maths). |
| `style.css`, `script.js` | Shared design system (CSS custom properties) and theme toggle, nav, background canvas. |
| `404.html`; `publications/`, `talks/`, `teaching/` | Not-found page; redirects from old academicpages URLs. |
| `images/`, `media/`, `docs/*.pdf`, `favicon.svg` | Photos, blog figures, lecture and talk slides. |
| `_config.yml` | Only lists files Jekyll must not publish. |

## Conventions

- **Nothing private on the site.** Every file in the repo is published
  unless `_config.yml` excludes it. When you add a file for maintainers or
  agents (not visitors), add it to `exclude` in the same change. The repo is
  public, so excluded files are still readable on GitHub: write nothing
  there that should not be public.
- **Drafts are not committed to `main`.** Old drafts are in git history
  (see `docs/decisions.md`).
- Pages are plain HTML with no front matter; do not add front matter or
  Liquid (`{{ }}`, `{% %}`) unless you mean Jekyll to process the file.
- The nav, header and footer are repeated in each page; change them in every
  page.
- `docs/` mixes public slide PDFs (linked from the site, keep their URLs)
  with the excluded maintainer docs.
- Claude may merge its own PRs here after checking the pages render.

## Current status

As of: 2026-10-13

- Working: Feb 2026 design with Oct 2026 content refresh and polish (PRs
  #1 to #3): Cusp announcement, linked publications, 404 and redirects,
  drafts and template leftovers removed.
- In progress: nothing.
- Next: when Daniel starts at Cusp (1 Dec 2026), update the hero, which says
  "from 1 Dec 2026", and link Cusp once a URL is confirmed.
- Waiting on: a Cusp URL; optionally a higher-resolution original of the
  homepage photo (the current one is 256×456).

## Where the rest lives

- [docs/decisions.md](docs/decisions.md): dated decisions and why they were made.
- [docs/context.md](docs/context.md): people, goals, deadlines, background.
- [.claude/skills/](.claude/skills/): repeatable agent workflows for this repo.
- [README.md](README.md): public-facing summary of the stack and layout.
