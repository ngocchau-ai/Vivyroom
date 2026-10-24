#ifndef CT_WVS_H
#define CT_WVS_H

/*
 * wvs.h — Weight Value Scoreboard (CAUTREO v2)
 *
 * Trái tim hệ thống: đánh giá giá trị trọng số theo tập tính người dùng.
 * Granularity tự điều phối (EXPERT / TENSOR / HYBRID / AUTO).
 *
 * 5 mức residency priority:
 *   hot       (>80%)  → RAM pin
 *   semi-hot  (≤80%)  → RAM prefer
 *   warm      (≤60%)  → RAM if budget
 *   cold      (≤25%)  → SSD prefetch
 *   rare      (<10%)  → SSD stream
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Granularity ─────────────────────────────────────────────────────── */

typedef enum {
    CT_WVS_GRAN_EXPERT,   /* MoE expert-level (coarse, 64-256 entries) */
    CT_WVS_GRAN_TENSOR,   /* per-tensor (fine, 300-500 entries) */
    CT_WVS_GRAN_HYBRID,   /* expert coarse + tensor refine cho hot */
    CT_WVS_GRAN_AUTO,     /* tự chọn dựa trên HAL scorecard */
} ct_wvs_granularity_t;

extern const char *ct_wvs_gran_name(ct_wvs_granularity_t g);

/* ── Hotness / Residency ─────────────────────────────────────────────── */

typedef enum {
    CT_HOTNESS_HOT       = 0,  /* >80%  → RAM pin */
    CT_HOTNESS_SEMI_HOT  = 1,  /* ≤80%  → RAM prefer */
    CT_HOTNESS_WARM      = 2,  /* ≤60%  → RAM if budget */
    CT_HOTNESS_COLD      = 3,  /* ≤25%  → SSD prefetch */
    CT_HOTNESS_RARE      = 4,  /* <10%  → SSD stream */
    CT_HOTNESS_COUNT,
} ct_hotness_t;

extern const char *ct_hotness_name(ct_hotness_t h);
/* Legacy name retained for API compatibility; returns residency hint. */
extern const char *ct_hotness_precision(ct_hotness_t h);

/* ── WVS Entry ───────────────────────────────────────────────────────── */

typedef struct {
    char    key[64];          /* expert_N or blk.N.xxx.weight */
    uint64_t access_count;    /* tổng số lần access */
    uint64_t last_access_ms;  /* timestamp (ms since epoch) */
    float    hotness_score;   /* 0.0 – 1.0 (tần suất gần đây) */
    ct_hotness_t hotness;     /* phân loại hiện tại */
    uint32_t slot;            /* vị trí trong bảng (hash index) */
} ct_wvs_entry_t;

/* ── WVS Scoreboard ──────────────────────────────────────────────────── */

typedef struct ct_wvs_s ct_wvs_t;

/* Tạo scoreboard với sức chứa tối đa `max_entries`.
 * granularity = CT_WVS_GRAN_AUTO → tự chọn sau khi detect hardware. */
ct_wvs_t *ct_wvs_create(uint32_t max_entries, ct_wvs_granularity_t granularity);
void      ct_wvs_destroy(ct_wvs_t *wvs);

/* Ghi nhận một access: tìm entry theo key, tăng counter, update hotness.
 * Nếu entry chưa tồn tại và còn slot → tạo mới.
 * Returns entry index hoặc -1 nếu full. */
int ct_wvs_record_access(ct_wvs_t *wvs, const char *key);

/* Tra cứu hotness của một key. Returns CT_HOTNESS_RARE nếu không tìm thấy. */
ct_hotness_t ct_wvs_get_hotness(const ct_wvs_t *wvs, const char *key);
bool ct_wvs_get_hotness_score(const ct_wvs_t *wvs, const char *key, float *score);

/* Lấy precision đề xuất cho key. */
/* Legacy name retained for API compatibility; returns residency hint. */
const char *ct_wvs_get_precision(const ct_wvs_t *wvs, const char *key);

/* ── Granularity ─────────────────────────────────────────────────────── */

/* Tự chọn granularity dựa trên hardware scorecard + model info.
 * Gọi sau ct_hal_detect(). */
ct_wvs_granularity_t ct_wvs_select_granularity(uint64_t ram_total_bytes,
                                                uint64_t ram_avail_bytes,
                                                uint64_t model_size_bytes,
                                                uint64_t num_params,
                                                bool is_moe);

/* Set granularity manually (override auto). */
void ct_wvs_set_granularity(ct_wvs_t *wvs, ct_wvs_granularity_t g);
ct_wvs_granularity_t ct_wvs_get_granularity(const ct_wvs_t *wvs);

/* ── Persist / Load ──────────────────────────────────────────────────── */

/* Lưu scoreboard ra file (binary). Returns 0 on success. */
int ct_wvs_save(const ct_wvs_t *wvs, const char *path);

/* Load scoreboard từ file. Returns NULL on failure. */
ct_wvs_t *ct_wvs_load(const char *path, uint32_t max_entries);

/* ── Utilities ───────────────────────────────────────────────────────── */

/* Reset tất cả entries (xóa lịch sử). */
void ct_wvs_reset(ct_wvs_t *wvs);

/* In scoreboard ra stdout. */
void ct_wvs_print(const ct_wvs_t *wvs);

/* Lấy số entries hiện tại. */
uint32_t ct_wvs_count(const ct_wvs_t *wvs);

/* Lấy entry theo index (0..count-1). Returns NULL nếu out of range. */
const ct_wvs_entry_t *ct_wvs_entry(const ct_wvs_t *wvs, uint32_t idx);

/* Cập nhật hotness cho tất cả entries (gọi định kỳ).
 * Dùng decay factor để giảm dần hotness của entries ít dùng. */
void ct_wvs_update_all(ct_wvs_t *wvs);

#ifdef __cplusplus
}
#endif

#endif /* CT_WVS_H */
