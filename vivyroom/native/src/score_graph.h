#ifndef CT_SCORE_GRAPH_H
#define CT_SCORE_GRAPH_H

/*
 * score_graph.h — Runtime Score Graph (CAUTREO)
 *
 * Mọi quyết định runtime quan trọng đều có score + confidence.
 * Không heuristic rời rạc — tất cả score nằm trong một graph thống nhất.
 *
 * Score Graph là nền tảng cho:
 *   - Runtime Governor (quyết định policy)
 *   - Adaptation State Machine (COLD→WARMING→ADAPTED)
 *   - Self-Improvement (candidate policy evaluation)
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Score Types ─────────────────────────────────────────────────────── */

typedef enum {
    CT_SCORE_MODEL_CAPABILITY = 0,  /* model capability score */
    CT_SCORE_AGENT_FIT,             /* agent-task compatibility */
    CT_SCORE_TASK_PROGRESS,         /* task completion likelihood */
    CT_SCORE_MEMORY_QUALITY,        /* memory retrieval relevance */
    CT_SCORE_CONTEXT_EFFICIENCY,    /* context utilization */
    CT_SCORE_PRECISION_FIT,         /* precision selection quality */
    CT_SCORE_STREAM_EFFICIENCY,     /* SSD streaming efficiency */
    CT_SCORE_OUTCOME,               /* task outcome quality */
    CT_SCORE_COUNT,
} ct_score_type_t;

/* ── Score Entry ─────────────────────────────────────────────────────── */

typedef struct {
    ct_score_type_t type;
    float           score;          /* 0.0 - 1.0 */
    float           confidence;     /* 0.0 - 1.0 (bằng chứng nhiều hay ít) */
    uint64_t        evidence_count; /* số lần cập nhật */
    uint64_t        last_update_ms; /* timestamp cập nhật cuối */
} ct_score_entry_t;

/* ── Score Graph ─────────────────────────────────────────────────────── */

typedef struct ct_score_graph_s ct_score_graph_t;

/* Tạo score graph. */
ct_score_graph_t *ct_score_graph_create(void);

/* Giải phóng score graph. */
void ct_score_graph_destroy(ct_score_graph_t *graph);

/* ── Update ──────────────────────────────────────────────────────────── */

/* Cập nhật score cho một score type.
 * evidence_weight: 0.0-1.0, càng cao thì confidence tăng càng nhanh.
 * Returns 0 on success. */
int ct_score_graph_update(
    ct_score_graph_t *graph,
    ct_score_type_t   type,
    float             score,
    float             evidence_weight
);

/* Cập nhật score từ outcome (task success/failure).
 * success: true = task thành công, false = task thất bại.
 * Tự động cập nhật CT_SCORE_OUTCOME và các score liên quan. */
int ct_score_graph_update_outcome(
    ct_score_graph_t *graph,
    bool              success,
    float             quality    /* 0.0-1.0, mức độ thành công */
);

/* ── Query ───────────────────────────────────────────────────────────── */

/* Lấy score entry cho một score type.
 * Returns NULL nếu type không hợp lệ. */
const ct_score_entry_t *ct_score_graph_get(
    const ct_score_graph_t *graph,
    ct_score_type_t         type
);

/* Lấy score value (0.0-1.0). Returns 0.0 nếu chưa có. */
float ct_score_graph_score(
    const ct_score_graph_t *graph,
    ct_score_type_t         type
);

/* Lấy confidence value (0.0-1.0). Returns 0.0 nếu chưa có. */
float ct_score_graph_confidence(
    const ct_score_graph_t *graph,
    ct_score_type_t         type
);

/* Kiểm tra xem đã có đủ confidence trên ít nhất N score types chưa. */
bool ct_score_graph_has_minimum_confidence(
    const ct_score_graph_t *graph,
    uint32_t                min_types,
    float                   min_confidence
);

/* ── Core/Learned Blending ───────────────────────────────────────────── */

/* Tính blended score = alpha * core_score + (1-alpha) * learned_score.
 * alpha tự động giảm khi confidence tăng:
 *   - confidence < 0.3: alpha = 0.8 (80% core)
 *   - confidence > 0.7: alpha = 0.25 (25% core)
 *   - giữa: linear interpolation */
float ct_score_graph_blended(
    const ct_score_graph_t *graph,
    ct_score_type_t         type,
    float                   core_prior  /* prior từ bootstrap core map */
);

/* ── Aggregate ───────────────────────────────────────────────────────── */

/* Tính overall system health score (weighted average của tất cả scores). */
float ct_score_graph_health(const ct_score_graph_t *graph);

/* Đếm số score types có confidence >= threshold. */
uint32_t ct_score_graph_confident_count(
    const ct_score_graph_t *graph,
    float                   threshold
);

/* ── Persist / Load ──────────────────────────────────────────────────── */

/* Lưu score graph ra file (binary). Returns 0 on success. */
int ct_score_graph_save(const ct_score_graph_t *graph, const char *path);

/* Load score graph từ file. Returns NULL on failure. */
ct_score_graph_t *ct_score_graph_load(const char *path);

/* ── Utilities ───────────────────────────────────────────────────────── */

/* Lấy tên score type. */
const char *ct_score_type_name(ct_score_type_t type);

/* In score graph ra stdout (debug). */
void ct_score_graph_print(const ct_score_graph_t *graph);

/* Reset tất cả scores. */
void ct_score_graph_reset(ct_score_graph_t *graph);

#ifdef __cplusplus
}
#endif

#endif /* CT_SCORE_GRAPH_H */
