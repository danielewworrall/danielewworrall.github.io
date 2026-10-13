---
name: keep-it-current
description: Run at the end of significant work (a milestone, an experiment, a design choice, a long session) to write new decisions, results and status back into the repo so they do not stay trapped in chat or memory.
---

# keep-it-current

The repo is the record. Anything a future session would need from this
session's conversation goes into the repo now, in the files below. Keep the
diff small and factual.

## 1. Work out what changed

- Find the last update: the `As of:` date under Current status in
  `CLAUDE.md`, and `git log --since=<that date> --oneline`.
- Go through this session's conversation and work since then for:
  decisions made or reversed, results measured, things tried that failed,
  new commands or setup steps, new people or deadlines, prompt changes,
  questions now waiting on the owner.
- If the repo has no `<!-- self-describing: v1 -->` marker in `CLAUDE.md`,
  stop and suggest running `make-self-describing` first.

## 2. Write it down

| What | Where |
|---|---|
| Decision (a choice between alternatives someone might reopen) | New entry at the top of `docs/decisions.md`: date made, Status, Decision, Why, Rejected. A reversal is a new entry; mark the old one `Superseded by YYYY-MM-DD <title>`. |
| Result (a measurement, a benchmark, an experiment outcome, a negative result) | Where `CLAUDE.md` Conventions says results go. If it says nothing, add to `docs/results.md` and add that convention. Include the date, the command or script that produced it, and the commit. |
| Status | Rewrite `## Current status` in `CLAUDE.md`: As of today, Working, In progress, Next, Waiting on. Replace, do not append; history lives in git and decisions. |
| New command, dependency or setup step | `## Setup` or `## Run and test` in `CLAUDE.md`. |
| New entry point, module or script | `## Entry points` in `CLAUDE.md`. |
| People, goals, deadlines | `docs/context.md`. |
| Prompt changed | New file `prompts/<name>.v<N+1>.md` with `changed:` filled in; update the table in `prompts/README.md`. |
| A procedure done twice | A new skill under `.claude/skills/`, listed in its README. |
| Gotcha (slow test, broken tool, misleading output) | `## Conventions` or `## Run and test` in `CLAUDE.md`, one line. |
| A question asked and left unanswered, then overtaken | Decision entry with an `Open:` line, or `Waiting on` in Current status if it is still live. |

Rules:

- Write only what happened. Mark inferences as such.
- Dates are absolute (YYYY-MM-DD), never "yesterday" or "last week".
- If a doc now says something false, fix it in the same change.
- Never write secrets.
- Keep CLAUDE.md short: move anything long into `docs/` and link it.

## 3. Tidy memory

If this session has project memory, look for facts there that belong in the
repo (decisions, status, how to run things). Write them into the repo as
above, then trim the memory entry to a pointer to the repo file. Memory keeps
only what is about the person or the project setup, not the work itself.

## 4. Commit

Commit on the current branch with the message
`docs: keep-it-current YYYY-MM-DD`. If the current branch is the default
branch and the project works through PRs, open a small PR instead. Tell the
owner in one line what was recorded and where.
