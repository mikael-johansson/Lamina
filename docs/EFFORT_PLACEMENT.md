# Reasoning effort placement and "No response from provider": findings (2026-11-05/06)

**Conclusions**

- **Cause of the stops:** `"effort_position": "end"` caused most of Hermes's "No response from provider" errors. These are replies that end inside their thinking. It was rolled back to `"start"` on 2026-11-05, and since then 0 of 420 Hermes requests have stopped this way.
- **Effort sentence:** the `xhigh` sentence makes the model think longer wherever it goes, but gave no measurable gain in correct answers. A system turn right before the reply causes the stops. Appending the sentence to the last message is safe from stops, but it doesn't help either.
- **Recommendation:** keep `start` and leave `xhigh` where it is.

Data and scripts: `bench/results/2026-11-06-effort-placement/`. Related issues: stack/strata#11, stack/hermes-agent#2 and stack/hermes-agent#19.

## 1. Background

- **2026-11-02:** `effort_position: "end"` was enabled in `configs/fork-iq3_s-abliterated.json` (13:41 UTC) and `configs/fork-iq3_s-split-prod.json` (13:55 UTC) (strata#11). Backups: `*.bak-preeffortpos`.
    - **What it does:** `Service.render_prompt` (`serve/server.py:4768`) renders every prompt start as the default (`xhigh`, sentence at the top of the system block), and puts a `low`/`medium` effort in a short system turn right before the reply. The engine checkpoints in front of that turn (`--tail-role-token 8678`), so switching effort doesn't re-read the conversation.
    - **What it doesn't do:** `xhigh` and `high` stay at the top (`serve/server.py:4775`). Their prompts are the same under `start` and `end`.
- **Motivation:** hermes-agent#2's measurement on vLLM / W4A16 (40 Hermes prompts, 18–40k tokens), comparing `xhigh` thinking length against `medium`: sentence at the top 1.17×, appended to the last message 1.77×, own system turn before the reply 2.03×. That measured `xhigh` at the end, which `end` never implemented.
- **`medium` under `start`:** it renders no effort sentence at all.
- **Synthetic `</think>`:** the `</think>` in the request logs is written by `serve/request_log.py:135` when a reasoning reply finishes. It doesn't show that the model closed its thinking.

## 2. Production stop rates

These cover Hermes agent requests, with each reply classed as thinking only: no answer and no tool call.

| Period | `medium` requests stopping in thinking |
|---|---:|
| 27 Oct to 2 Nov, `start` | 4 / 7,919 (0.05%) |
| 2–5 Nov, `end` | 18 / 1,256 (1.43%): regular 7/589, abliterated 11/667 |
| After the rollback (5 Nov 15:34 to 6 Nov ~07:45 UTC), `start` | **0 / 420** |

At 1.43%, about 6 stops would have been expected in 420 requests, so the chance of seeing none is about 0.25%. `xhigh` requests under `end` had 0 stops in 469, but that's a different workload. The baseline of about 1 in 2,000 still exists under `start`.

## 3. Method

- **Prompts:** each was rebuilt token for token from the request logs, using `openai_to_messages` and `Service.encode_prompt` with a stub service. The rebuilt lengths match the logged prompt tokens exactly.
- **Requests:** `/v1/completions` with raw ids against the live regular server (`strata.issue19`, IQ3_S, port 8091). Sampling matched production: temperature 0.7, top-p 0.95, top-k 20, min-p 0. Concurrency was C=2.
- **Classes:**
    - `no_close`: the reply ends on end-of-turn without `</think>` and without a tool call.
    - `call_in_thinking`: a complete tool call inside unclosed thinking.
    - `length`: the reply hit the token limit.
- **Forms of the effort sentence:**
    - `none`: no sentence, which is what `medium` gets under `start`.
    - `top`: the `xhigh` sentence at the top, which is `xhigh` today.
    - `tail`: the `xhigh` sentence as its own system turn right before the reply.
    - `last`: the `xhigh` sentence appended to the last message, usually a tool result.

## 4. Replay of the `end` setting (5 Nov)

This used 12 prompts: 6 from sessions that failed in production (3 of them about 190k tokens) and 6 random ones. The limit was 1,500 tokens per reply.

| Form | Stopped in thinking |
|---|---:|
| As deployed (`xhigh` at the top + `medium` turn at the end) | 21/148 (14.2%) |
| `medium` turn at the end only | 9/162 (5.6%) |
| `xhigh` at the top only | 0/162 |
| Neither (`start`) | 0/148 |

The rates are inflated, because half of the prompts came from failed sessions.

## 5. Placement: thinking length and stops

The first test used 39 random production prompts, one per conversation, in 4 buckets from 20k to 167k tokens. Each form ran 4 times, with a limit of 8,000 tokens per reply. The stress test used the 12 prompts from section 4, with 8 runs of each form. The two together give 252 replies per form.

| Form | Thinking vs `none`, median per-prompt ratio (95% bootstrap) | At 130k+ tokens | Stopped in thinking |
|---|---:|---:|---:|
| `none` | 1 | 1 | 1/252 (+1 `call_in_thinking`) |
| `top` | 1.38× (1.17–1.72) | 1.20× | 0/252 |
| `tail` | 1.56× (1.25–2.29) | 2.84× | **19/252** (10/16 on prompts from failed sessions, including a 190k one) |
| `last` | 2.48× (1.82–2.99) | 3.60× | 0/252 |

- **Short replies:** most of these are tool steps, with a median of 91–219 thinking tokens, so length here doesn't show quality.
- **`last` side effects:** in 2 of 156 random-prompt replies the model reacted to the injected sentence. It copied the sentence into its thinking once, and once said that the last message was only an instruction.

## 6. Placement: graded accuracy (GPQA-Diamond)

- **Questions:** the same 40 GPQA-Diamond questions as `bench/runs/quality/iq3s-prod-gpqa-c3.json`, with a limit of 40,000 tokens per reply.
- **Contexts:** the questions were asked as a new user message after real Hermes conversations: d#1 (142k), b#2 (78k) and d#2 (167k), as picked by `r3/build.py`. They were also asked with no conversation before them.
- **Scoring:** a reply with no answer counts as wrong.

| Before the questions | Rounds | `none` | `top` | `last` | `top` vs `none` (+better / −worse) | `last` vs `none` | Median tokens `none` / `top` / `last` |
|---|---:|---:|---:|---:|---|---|---|
| No conversation | 1 | 85% | 88% | 85% | +2/−1, p=1.00 | +4/−4, p=1.00 | 2,082 / 6,362 / 9,591 |
| 78k-token conversation | 3 | 86% | 88% | 90% | +7/−5, p=0.77 | +7/−3, p=0.34 | 848 / 1,386 / 1,446 |
| 142k-token conversation | 3 | **92%** | 83% | 82% | +2/−10, **p=0.04** | +1/−12, **p=0.003** | 1,362 / 1,853 / 1,802 |
| 167k-token conversation | 3 | 90% | 89% | 93% | +6/−8, p=0.79 | +6/−4, p=0.75 | 1,183 / 2,307 / 1,930 |
| **Pooled** (400 replies per form) | | **89.0%** | **86.8%** | **88.2%** | 17/24 pairs, p=0.35 | 18/23 pairs, p=0.53 | |

- **No measurable gain:** `xhigh` never helped significantly, in either position or at any context size. With no conversation it used 3–4.6× the tokens for the same score.
- **The 142k loss:** this is the only significant result. There, `xhigh` replies often resumed the conversation's earlier task or made a tool call, saying for example "I need to look into this further. Let me confirm the details." That didn't recur at 78k or 167k, so it depends on the conversation, not on context size alone.
- **`top` vs `last`:** indistinguishable in every setting.
- **Run-to-run noise:** after the 78k conversation, `none` scored 75% in round 1 and 92.5% in round 2. Single rounds are unreliable.

## 7. Limits

- **One benchmark:** GPQA is multiple-choice science. It may not show whether `xhigh` helps on real agent work, such as long coding tasks or reviews.
- **Easy questions:** the model scores close to 90% on them, so a small gain from `xhigh` could go unnoticed. A large one would show up.
- **Coverage:** only the regular model was tested, not the abliterated one, and only 3 conversations were used as context.
- **Side effects of the tests:** the replays ran on the live server, so live requests were slower and lost some prompt cache. They also left `cmpl-*` request logs in `/d/hdd_800/logs/strata-requests/`, from 5 Nov 09:5x–10:5x and 15:4x–17:05 UTC, and 6 Nov 00:12–07:46 UTC.

## 8. Recommendations

1. **Keep `"effort_position": "start"`** in both production configs. Don't re-enable `end`, and don't build a turn before the reply for any effort level.
2. **Leave `xhigh` at the top.** `last` is the only safe alternative, and it showed no gain.
3. **Pinning Gitea mentions to `xhigh`:** on this evidence, `xhigh` costs time without measurably better answers, so reconsider the pin. A graded agent-task test would settle it.
4. **Hermes:** still fix the handling of replies that stop inside their thinking (hermes-agent#19), for the rare remaining stops.
5. **`reasoning_close_retry` (upstream #1053):** leave it off unless stops stay above the old baseline.

## 9. Files

All under `bench/results/2026-11-06-effort-placement/`.

- **Removed for privacy:** the reply text, which comes from private Hermes sessions, and the prompt id files. Each row keeps its class, finish reason, token counts, `think_tokens` (counted with the IQ3_S tokenizer before removal), `mentions_effort` and, for GPQA, the answer and whether it was correct.
- **Rebuilding the prompts:** the `build.py` files rebuild them from the request logs on ada (`/d/hdd_800/logs/strata-requests/`).
- **Analysis scripts:** `an.py` needs the reply text, so it only runs on the full copy at `~/git/strata/bench/runs/effort-placement-2026-11-06/` on ada. `pool.py` runs on these files.

| Path | Contents |
|---|---|
| `eff.json` | Per-request effort and outcome scan behind section 2 |
| `replay/` | Section 4: `build.py`, `split.py`, `run2.py`, `results.jsonl`, `ctl.txt` (the random prompts) |
| `r3/` | Section 5, first test: `build.py`, `run.py`, `an.py`, `results.jsonl` |
| `r4/` | Section 5, stress test |
| `r5/` | Section 6, 142k conversation. `r6/build.py` builds the no-conversation (`zero`), 78k (`b#2`) and 167k (`d#2`) stages, which are in `r6/`, `r7/` and `r8/` |
| `pool.py`, `pooled.txt` | The table in section 6 |
