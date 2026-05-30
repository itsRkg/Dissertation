# Track Documentation — Structure & Working Policy

_This file is the contract for how documentation is organized in this repo. Read it before adding or moving any doc. Updated 2026-05-29._

## Why this structure exists

- The root files (`status.md`, `plan.md`, `handover.md`) are loaded into nearly every session. They must stay **small** to save token budget.
- Track-specific specs, notes, reference snippets, and scratch material balloon over time. Putting them in **per-track subdirectories** keeps the root lean and means a session working on Track X only loads Track X's files.
- A future session opening `status.md` should see pointers — not a 4000-line monolith.

## Directory hierarchy

```
Dissertation/
├── status.md                    # Master index, coarse. Loaded every session.
├── plan.md                      # Forward-looking plan + item map. Loaded every session.
├── handover.md                  # Session-to-session continuity. Loaded every session.
└── docs/
    ├── track_usages.md          # ← this file. The policy contract.
    ├── references.md            # Cross-cutting literature + internal doc index.
    ├── track_1a_imd_minimap/
    │   ├── README.md            # ← the track spec (entry point)
    │   └── ...                  # ← any track-specific files added later
    ├── track_1b_imd_era5_corr/
    │   ├── README.md
    │   └── ...
    ├── track_2_regional_ablation/
    │   ├── README.md
    │   └── ...
    └── track_3_rl_reuse/
        ├── README.md
        └── ...
```

## Rules for future additions

### Where does a new file go?

Ask: **is this content specific to a single track, or cross-cutting?**

- **Specific to a single track** → goes inside `docs/track_<id>_<name>/`. Examples:
  - Implementation notes for a single experiment in Track 2: `docs/track_2_regional_ablation/implementation_notes.md`.
  - A scratch derivation for Track 1A's graph degree analysis: `docs/track_1a_imd_minimap/graph_degree_analysis.md`.
  - A literature note that informs only one track: `docs/track_<id>_<name>/refs.md`.

- **Cross-cutting (touches multiple tracks or the whole project)** → goes at the upper level:
  - General literature → append to `docs/references.md`.
  - Forward-looking strategy or item-map change → update `plan.md`.
  - Project-wide known issue, depth map entry, or file-map change → update `status.md`.
  - Session-level update or open clarification → append to `handover.md`.

**If in doubt, default to the track directory.** It is cheaper to find a file by track than to bloat a root file.

### What's the entry point in a track directory?

Always **`README.md`**. The directory name encodes the track identity; `README.md` is the conventional entry point. Sub-files inside the directory should have descriptive names (`graph_degree_analysis.md`, `implementation_notes.md`, etc.) — not numbered.

### When should I move a track up to the root?

Almost never. Promote to a root doc only if a track's content becomes genuinely cross-cutting (e.g., a finding from Track 3 changes the master plan). Even then, write a one-line pointer at the root and keep the deep detail in the track directory.

### When should I add a new track?

When Risheek confirms a new item is in scope. Pattern:

1. Update `plan.md` item map with the new row.
2. Create `docs/track_<id>_<name>/README.md` with the track spec.
3. Add a pointer in `status.md` doc-tree table and the change log.
4. Append to `handover.md` for the next session.

Track IDs are zero-padded only if a track has parallel sub-tracks (e.g., Track 1A and 1B). Otherwise plain numeric is fine (Track 2, Track 3).

### When should I rename or retire a track?

If a track is abandoned, **do not delete it**. Mark its `README.md` with a status line at the top: `**Status: abandoned (YYYY-MM-DD). Reason: …**`. Move on. This preserves history for the dissertation writeup.

## File naming convention inside track directories

- `README.md` — the track spec.
- `implementation_notes.md` — running notes from active work.
- `experiments_<exp_id>.md` — per-experiment notes if any.
- `refs.md` — track-local literature pointers.
- `data_notes.md` — data-pipeline notes specific to the track.
- `decisions.md` — track-local decisions log (use only if many decisions accumulate).
- `<topic>_analysis.md` — ad-hoc analyses.

Use lowercase, underscore-separated. No spaces, no caps.

## Linking convention

- From a root doc to a track: link as `docs/track_<id>_<name>/README.md`.
- From inside a track to another track: link as `../track_<other_id>/README.md`.
- From a track to a root doc: link as `../../status.md` (or `../../plan.md`, `../../handover.md`).
- From a track to `references.md`: link as `../references.md`.

## Review-loop with Codex (or any external reviewer)

Several tracks will be sent to Codex (or another assistant) for a second-opinion review. Codex **does not** write to project files. Codex returns a chat response; Risheek pastes it back into Claude; Claude is the only writer. To make this loop reliable, every track doc under review carries a **version stamp** the reviewer must echo. The stamp is the contract that keeps the reviewer's view and Claude's view of the doc aligned.

### Version stamp format

Place this as the second line of any doc under review (right after the H1):

```
**REVIEW VERSION: `<track>-<file>-<YYYY-MM-DD>-v<N>`**
```

Examples:
- `track3-readme-2026-05-29-v1`
- `track1a-readme-2026-06-15-v3`

Bump `vN` every time the doc is materially edited after a review. Date stays the same within a single day; bump the date on the next day.

### Reviewer prompt requirements

Any reviewer prompt embedded in a track doc must:

1. Explicitly forbid the reviewer from writing files. State that Claude is the only writer.
2. Instruct the reviewer to **echo the version stamp** as the very first line of the response in the format `REVIEWED VERSION: <id>`.
3. Specify the response output format Claude expects (so the user can paste it back without reformatting).
4. Tell the reviewer to refuse to review if the version stamp is missing.

The Track 3 README has a worked example of this prompt; copy its shape for new tracks.

### Claude's procedure when a Codex response is pasted

Non-negotiable workflow:

1. **Match the version.** First line of the response should be `REVIEWED VERSION: <id>`. If it doesn't match the current doc's stamp, the review is stale. Stop. Ask Risheek to re-send the current doc to Codex. **Do not cherry-pick from a stale review.**
2. **Re-verify each flag against the source.** Codex's judgment is never the sole basis for an edit. For every flag that cites a source file, Claude reads the cited lines and confirms the reading. If Codex misread the source, the flag is rejected.
3. **Edit with Edit tool, not Write.** Each edit's commit message (or change-log entry) cites the Codex flag + the verifying source line.
4. **Bump the version stamp.** After applying flags, increment the `vN`. Append an entry to the change log in `status.md` and a note to `handover.md`.
5. **Boundary rule:** if a reviewer's response includes file-content suggestions (e.g., "here's a rewritten section X"), Claude evaluates the suggestion at the reasoning level but rewrites the section in Claude's own words. Verbatim adoption is not allowed — it short-circuits the verification step.

### Why this matters

- Without a version stamp, a review may target a doc state that no longer exists. Applying its flags introduces stale conclusions.
- Without a "Claude is the only writer" rule, two writers can race and silently overwrite each other's edits.
- Without re-verification, the project becomes only as reliable as the reviewer — and reviewers hallucinate too.

## Token-budget reasoning (the "why" again)

When a session starts:

1. The root files (`status.md`, `plan.md`, `handover.md`) are read for orientation. Total kept to ~10–15 KB combined.
2. The session decides which track(s) it's working on.
3. Only the relevant `docs/track_<id>_<name>/` is read.

This means adding a 500-line implementation note to Track 1A's directory costs **zero** tokens for sessions working on Track 2 or 3. The same note placed in `status.md` would cost every future session.

**Bottom line: write deep, link shallow.**
