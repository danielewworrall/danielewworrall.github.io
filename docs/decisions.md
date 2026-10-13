# Decisions

<!--
One entry per decision, newest first. A decision is a choice between real
alternatives that a later session might otherwise reopen or undo.

- Date it (YYYY-MM-DD), the day it was made, not the day it was written down.
- Say what was decided, why, and what was rejected.
- Never edit an old entry to change its meaning. When a decision is reversed,
  add a new entry and mark the old one "Superseded by YYYY-MM-DD <title>".
- Status is one of: decided, tentative (leaning, not signed off), superseded.
- If the source was a conversation rather than the repo, say so in one word
  (e.g. "Source: chat"), so readers know it is not visible elsewhere.

Copy this block for each new decision:

## YYYY-MM-DD: <short title>

Status: decided
Decision: <what was chosen, in one or two sentences>
Why: <the reason, including the evidence if any>
Rejected: <the alternatives and why not>
Open: <only if part of the question was never answered; say which part>
-->

## 2026-10-13: Keep maintainer docs off the live site with Jekyll `exclude`

Status: decided
Decision: Remove `.nojekyll` and add `_config.yml` whose `exclude` list keeps
`CLAUDE.md`, `README.md`, `docs/decisions.md` and `docs/context.md` off the
published site. `.claude/` is skipped by Jekyll as a dot-folder.
Why: Daniel asked that the self-describing files be readable by agents with
the repo but not served to people reading the website. With `.nojekyll` every
file in the repo was published. The pages have no front matter, so Jekyll
copies them byte for byte.
Rejected: a GitHub Actions deploy that uploads only the site files (needs a
Pages settings change and a workflow to maintain); moving the docs to another
repo (breaks the self-describing standard, which expects them here).
Open: the repo itself is public, so these files are still readable on
GitHub. Making it private would need a paid plan for Pages; not decided.
Source: chat

## 2026-10-12: Light polish, not a redesign

Status: decided
Decision: Keep the Feb 2026 design. Polish only: plain hero links without
emoji, three topic pills, compact publications list with four selected
papers, no scroll fade-in. The front page is allowed to be long.
Why: the design was judged still modern and consistent at desktop and phone
widths; Daniel agreed, except that the home page should keep all its sections.
Rejected: a facelift; trimming News to the latest few items.
Source: chat (shipped in PR #3)

## 2026-10-12: Announce the Cusp role

Status: decided
Decision: Hero, bio and news say Daniel joins Cusp as Lead of the Digital
Twins team on 1 December 2026. Cusp is not linked because no URL was
confirmed.
Why: Daniel said on 2026-10-12 this can now be public. On 2026-10-10 he had
asked not to say where he was going yet.
Source: chat (PR #3)

## 2026-10-10: Claude merges its own PRs on this repo

Status: decided
Decision: Claude may merge its own PRs here without waiting for review, after
checking the pages render.
Why: Daniel is the only author of the site and said review only slows it down.
Source: chat

## 2026-10-10: Drafts stay off the public site

Status: decided
Decision: Unpublished blog posts are not kept on `main`. The three Feb 2026
drafts (algebra-of-differentiation, b-series, rk-order-conditions) and the
old `_drafts/` were removed in PR #1. They remain in git history, e.g.
`git show 33b9de2:blog/2026/02/b-series/index.html`.
Why: they were live and indexable under `/blog/` and `/draft/`. Daniel: drafts
should not be public.
Rejected: keeping them under `/draft/` with `noindex` (still public).
Open: where drafts should live while being written was not decided. The repo
is public, so a branch or an excluded folder would still be readable on GitHub.
Source: chat

## 2026-10-10: Delete the academicpages leftovers

Status: decided
Decision: Delete the unused academicpages/Jekyll template files (`_config.yml`
with academicpages settings, `_posts/`, `_pages/`, sample PDFs in `files/`,
and so on) and add a 404 page plus redirects from old `/publications/`,
`/talks/` and `/teaching/` URLs.
Why: `.nojekyll` caused them to be served raw; nothing used them. Recoverable
from history.
Source: chat (PR #1); audit at the project's
`audit/website-audit-2026-10-10.md`

## 2026-02-21: Plain static HTML instead of academicpages

Status: decided
Decision: Rebuild the site as hand-written HTML, CSS and vanilla JS with no
framework or build step, and skip Jekyll with `.nojekyll`.
Why: inferred from commits `1f26b4c` and `0a2f6e5`; no reason recorded.
The `.nojekyll` part is superseded by 2026-10-13 above.
