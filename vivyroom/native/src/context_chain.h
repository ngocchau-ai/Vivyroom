/**
 * context_chain.h — Context Chain Engine (CCE) Public API
 *
 * Module C native trong CAUTREO Server.
 * Phân rã context lớn thành chuỗi context nhỏ, giữ nguyên chính xác
 * qua ChainState accumulation.
 *
 * Không cần Python, không cần Ollama adapter.
 */

#ifndef CT_CONTEXT_CHAIN_H
#define CT_CONTEXT_CHAIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Forward declarations ──────────────────────────────────────────── */
struct ct_engine;
typedef struct ct_engine ct_engine;

/* ── Limits (compile-time, stack-friendly) ─────────────────────────── */
#define CT_CHAIN_MAX_SEGMENTS    64
#define CT_CHAIN_MAX_CHAINS      16
#define CT_CHAIN_MAX_ENTITIES   256
#define CT_CHAIN_MAX_FACTS      128
#define CT_CHAIN_MAX_DECISIONS   64
#define CT_CHAIN_MAX_QUESTIONS   32
#define CT_CHAIN_MAX_OUTPUTS     16
#define CT_CHAIN_MAX_PROMPT   65536   /* 64 KB per chain prompt */
#define CT_CHAIN_ID_LEN          16
#define CT_CHAIN_ROLE_LEN        16
#define CT_CHAIN_PRIORITY_LEN     8

/* ── Chain State ───────────────────────────────────────────────────── */

typedef struct ct_chain_entity {
    char name[64];
    char info[128];
} ct_chain_entity;

typedef struct ct_chain_state {
    ct_chain_entity entities[CT_CHAIN_MAX_ENTITIES];
    int             entity_count;

    char decisions[CT_CHAIN_MAX_DECISIONS][256];
    int  decision_count;

    char facts[CT_CHAIN_MAX_FACTS][256];
    int  fact_count;

    char open_questions[CT_CHAIN_MAX_QUESTIONS][256];
    int  question_count;

    char chain_outputs[CT_CHAIN_MAX_OUTPUTS][512];
    int  chain_output_count;
} ct_chain_state;

/* ── Segment ───────────────────────────────────────────────────────── */

typedef struct ct_context_segment {
    char  id[CT_CHAIN_ID_LEN];
    char  role[CT_CHAIN_ROLE_LEN];
    char *content;
    int   content_len;
    int   token_estimate;
    int   position;
    float weight;                       /* composite score [0,1] */
    char  priority[CT_CHAIN_PRIORITY_LEN]; /* HIGH / MEDIUM / LOW / DEAD */
} ct_context_segment;

/* ── Chain ─────────────────────────────────────────────────────────── */

typedef struct ct_chain {
    int                  id;
    ct_context_segment  *segments;
    int                  segment_count;
    int                  token_estimate;
    char                *prompt;
    int                  prompt_len;
} ct_chain;

/* ── Scoring coefficients ──────────────────────────────────────────── */

typedef struct ct_scoring_weights {
    float alpha;    /* task relevance      (default 0.40) */
    float beta;     /* recency             (default 0.20) */
    float gamma;    /* dependency          (default 0.20) */
    float delta;    /* information density (default 0.15) */
    float epsilon;  /* agent context       (default 0.05) */
} ct_scoring_weights;

/* ── Main context ──────────────────────────────────────────────────── */

typedef struct ct_context_chain_ctx {
    ct_chain             *chains;
    int                   chain_count;
    ct_chain_state        state;

    ct_scoring_weights    weights;
    int                   max_chain_tokens;
    int                   state_overhead;     /* reserved tokens for state injection */

    /* Raw segments (owned) */
    ct_context_segment   *segments;
    int                   segment_count;

    /* Stats */
    int                   total_input_tokens;
    int                   total_chains;
    float                 peak_memory_ratio;
} ct_context_chain_ctx;

/* ── Lifecycle ─────────────────────────────────────────────────────── */

ct_context_chain_ctx *ct_context_chain_create(int max_chain_tokens);
void                  ct_context_chain_free(ct_context_chain_ctx *ctx);

/* ── Phase 1: Decompose ────────────────────────────────────────────── */
/* Parse OpenAI-format JSON context → score → pack into chains.        */

int ct_context_chain_decompose(
    ct_context_chain_ctx *ctx,
    const char          *json_context,       /* JSON array of messages */
    int                  json_len,
    const char          *task_prompt,
    int                  task_len
);

/* ── Phase 2: Progressive Inference ────────────────────────────────── */
/* DEPRECATED: Use ct_cce_run() from runner.h instead.
 * This function returns -2 (CT_CCE_DEPRECATED) and does nothing.
 * The new ct_cce_run() in runner.c fully replaces this with real engine execution. */

int ct_context_chain_infer(
    ct_context_chain_ctx *ctx,
    ct_engine           *engine
);

/* ── Phase 3: Synthesize ───────────────────────────────────────────── */
/* Build final prompt combining all chain outputs + state.             */

int ct_context_chain_synthesize(
    ct_context_chain_ctx *ctx,
    char                *output,
    int                  output_max
);

/* ── Scoring ───────────────────────────────────────────────────────── */

float ct_chain_score_segment(
    const ct_context_segment  *seg,
    const char                *task_prompt,
    int                        task_len,
    int                        total_segments,
    const ct_scoring_weights  *w
);

/* ── State Management ──────────────────────────────────────────────── */

void ct_chain_state_init(ct_chain_state *state);
void ct_chain_state_merge(ct_chain_state *dst, const ct_chain_state *src);
int  ct_chain_state_serialize(const ct_chain_state *state, char *out, int out_max);

/* ── Segmenter ─────────────────────────────────────────────────────── */

int ct_chain_segment_context(
    const char            *json_context,
    int                    json_len,
    ct_context_segment    *out_segments,
    int                    max_segments,
    int                   *out_count,
    int                    max_tokens_per_segment
);

/* ── Packer ────────────────────────────────────────────────────────── */

int ct_chain_pack(
    ct_context_segment   *scored,
    int                   scored_count,
    ct_chain             *out_chains,
    int                   max_chains,
    int                  *out_chain_count,
    int                   max_chain_tokens,
    int                   state_overhead
);

/* ── State Extractor (rule-based, from engine response) ────────────── */

int ct_chain_state_extract(
    ct_chain_state       *state,
    const char           *response,
    int                   response_len
);

#ifdef __cplusplus
}
#endif

#endif /* CT_CONTEXT_CHAIN_H */
