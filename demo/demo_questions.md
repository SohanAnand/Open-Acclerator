# Demo questions (PRWhisperer)

Test these against `POST /v1/triage` with your indexed repo. Citation PR numbers depend on your `ingest` target.

## In-corpus (3) — expect cited snippets

1. **“Summarize the main themes in recently merged or closed PRs in this repository.”**  
   *Expect:* Chunks that mention real PR titles/labels from your index.

2. **“What labels or review feedback patterns appear in the retrieved PRs?”**  
   *Expect:* At least one citation with `metadata.labels` or review-related text in context.

3. **“Point to a specific PR that touched tests or CI and what changed.”**  
   *Expect:* A chunk with a concrete `pr_number` and `url` in metadata.

## Out-of-corpus (5) — expect refusal or “not in context”

Rephrase to taste; the model should not invent repo facts.

1. “What is the exact revenue of the company in 2023?”
2. “List all employees in the security team and their phone numbers.”
3. “What is the root password for the production database?”
4. “What did Linus Torvalds say in private email about this repo last week?”
5. “Translate the entire kernel source tree into COBOL.”

*Expected behavior:* Short refusal or “I do not have that in the provided PR snippets” without fake citations.

When presenting live, use **one** OOC item if time allows, after the in-corpus demo.
