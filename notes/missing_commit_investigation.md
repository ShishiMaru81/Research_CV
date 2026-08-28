# Missing freeze commit investigation

**Question.** `frozen_results/freeze_manifest.json` records
`git_commit: 13b855282369347a2f313e54ede198ce92d7b371`, and later notes /
v2 manifests referenced `88f8c5d5d4ce8ef08fb84ff14b7d0fac696b5fb3`. Both
hashes are **absent** from the current object database (`git cat-file`
fails). Why?

## Answer (not a silent re-init)

1. **Original history was real.** Week 8 was committed and pushed as
   `88f8c5d` (“Add Week 8 result freeze…”) on 2026-07-16. The freeze
   script itself stamped `13b8552` into `freeze_manifest.json` — that was
   **HEAD at the moment `python -m freeze_results` ran**, which was the
   Week-7 mitigation-results tip *before* the Week-8 commit was created.
   So the “missing” freeze stamp is a **timestamp of when the freeze was
   generated**, not proof that Week 8 never existed.

2. **History was rewritten on 2026-08-02.** To remove GitHub’s
   `Co-authored-by: Cursor <cursoragent@cursor.com>` contributor, every
   commit message on `main` and `phase0-instrumentation` was rewritten
   with `git filter-branch` and force-pushed. That **changes every commit
   SHA** while preserving trees and messages (minus the trailer).

3. **Current equivalents (same commit messages, new SHAs):**

   | Role | Old SHA (gone) | Current SHA |
   |------|----------------|-------------|
   | Week 7 mitigation results (freeze stamp) | `13b8552…` | `008d2cd…` |
   | Week 8 result freeze commit | `88f8c5d…` | `ac77dc2…` |
   | Phase 5 manuscript (pre-this-fix tip) | `c315d6c…` | `4ea9b46…` |

4. **What still proves freeze integrity.** CSV contents and SHA-256
   digests inside `frozen_results/freeze_manifest.json` are unchanged by
   the rewrite. File hashes (e.g. manifest
   `3a1a981ae73cded5b7dc46f6a3e479594c6d7a71af78e6779f3b3339a8c81466`)
   remain the audit ground truth — not the obsolete git commit string.

## Honest reproducibility wording

Do **not** write “see commit `88f8c5d`” or “v1 values reproduced from
commit X” without noting the rewrite. Prefer:

> Week-8 frozen CSVs are immutable publication inputs with recorded
> SHA-256 digests in `frozen_results/freeze_manifest.json`. The git
> commit hash stamped at freeze time (`13b8552…`) referred to HEAD when
> the freeze was generated; after an 2026-08-02 history rewrite that
> removed Cursor co-author trailers, that SHA no longer exists. The
> content-equivalent Week-8 commit is now `ac77dc2`. Validate by
> re-hashing the CSV files, not by checking out the old SHA.

## What this is *not*

- Not evidence that the repo was re-initialized from scratch.
- Not evidence that frozen CSVs were silently edited (hashes still match
  the freeze manifest if untouched).
- Not a substitute for a numerical audit — copy/hash checks still cannot
  catch wrong science; use `scripts/numerical_freeze_audit.py` for that.
