/*
 * ct_weight_pager.h — Cautreo Weight Slice Registry & Sequential FFN Streaming.
 *
 * Spec: SPEC-CAUTREO-WEIGHT-PAGING-R4 (Sprint R4)
 *
 * Manages Zero-RAM-Waste Task-Sequential FFN Streaming for arbitrary LLMs.
 * Reads real tensor bytes from GGUF files, enforces a fixed RAM budget,
 * supports sparse (top-k) activation, and performs real matrix-vector
 * computation through loaded slices.
 *
 * Gate 9: no performance claims — measurements go to benchmark receipts.
 */
#ifndef CT_WEIGHT_PAGER_H
#define CT_WEIGHT_PAGER_H

#include "ct_gguf_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Error codes
 * ----------------------------------------------------------------------- */
typedef enum {
    CT_WP_OK            =  0,
    CT_WP_ERR_IO        = -1,
    CT_WP_ERR_MAGIC     = -2,
    CT_WP_ERR_VERSION   = -3,
    CT_WP_ERR_CORRUPT   = -4,
    CT_WP_ERR_NOMEM     = -5,
    CT_WP_ERR_NOT_FOUND = -6,
    CT_WP_ERR_BOUNDS    = -7,
    CT_WP_ERR_BUDGET    = -8,
    CT_WP_ERR_STATE     = -9,
    CT_WP_ERR_UNSUPPORTED = -10,
} ct_wp_status_t;

/* -----------------------------------------------------------------------
 * Slice state
 * ----------------------------------------------------------------------- */
typedef enum {
    CT_SLICE_ON_DISK  = 0,
    CT_SLICE_PAGING   = 1,
    CT_SLICE_RESIDENT = 2,
    CT_SLICE_DIRTY    = 3,
} ct_slice_state_t;

/* -----------------------------------------------------------------------
 * Weight slice entry
 * ----------------------------------------------------------------------- */
#define CT_SLICE_NAME_MAX 128

typedef struct {
    char     name[CT_SLICE_NAME_MAX];
    uint64_t file_offset;
    uint64_t size_bytes;
    uint64_t loaded_bytes;
    uint32_t layer_index;
    ct_ggml_type_t quant_type;
    ct_slice_state_t state;
    float    task_affinity;
    uint64_t last_accessed_ms;
    void    *buffer;
    uint64_t buffer_capacity;
    /* Shape for stream_compute: [rows, cols] or [n] for 1-D */
    uint32_t n_dims;
    uint64_t dims[4];
} ct_weight_slice_t;

/* -----------------------------------------------------------------------
 * Registry handle
 * ----------------------------------------------------------------------- */
typedef struct {
    ct_gguf_file_t *gguf;
    char            model_id[64];
    uint64_t        max_ram_bytes;
    uint64_t        current_ram_bytes;
    uint32_t        n_slices;
    uint32_t        capacity;
    ct_weight_slice_t *slices;
} ct_weight_registry_t;

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */
CT_API ct_wp_status_t ct_weight_registry_init(const char *model_path,
                                               uint64_t max_ram_bytes,
                                               ct_weight_registry_t **out);
CT_API void ct_weight_registry_free(ct_weight_registry_t *reg);

/* Auto-register weight slices from GGUF tensor index. Returns count or <0 error. */
CT_API int ct_weight_registry_build_from_gguf(ct_weight_registry_t *reg);

/* Register a single slice manually (for tests / synthetic fixtures). */
CT_API ct_wp_status_t ct_weight_registry_register_slice(
    ct_weight_registry_t *reg, const char *name, uint64_t file_offset,
    uint64_t size_bytes, uint32_t layer_index, ct_ggml_type_t quant_type);

/* -----------------------------------------------------------------------
 * Sequential paging
 * ----------------------------------------------------------------------- */
CT_API ct_wp_status_t ct_weight_slice_page_in(ct_weight_registry_t *reg,
                                               const char *slice_name,
                                               void **out_buffer_ptr);
CT_API ct_wp_status_t ct_weight_slice_page_out(ct_weight_registry_t *reg,
                                                const char *slice_name);
CT_API ct_wp_status_t ct_weight_slice_sparse_page_in(
    ct_weight_registry_t *reg, const char *slice_name, float top_k_ratio,
    void **out_buffer_ptr);

/* -----------------------------------------------------------------------
 * Sequential streaming compute (real matvec)
 * ----------------------------------------------------------------------- */
CT_API ct_wp_status_t ct_weight_slice_stream_compute(
    ct_weight_registry_t *reg, const char *slice_name,
    const float *input, size_t input_len,
    float *output, size_t output_len);

/* -----------------------------------------------------------------------
 * Introspection
 * ----------------------------------------------------------------------- */
CT_API uint64_t ct_weight_registry_ram_usage(const ct_weight_registry_t *reg);
CT_API int      ct_weight_registry_resident_count(const ct_weight_registry_t *reg);
CT_API int      ct_weight_registry_slice_count(const ct_weight_registry_t *reg);
CT_API int      ct_weight_slice_state(const ct_weight_registry_t *reg,
                                       const char *slice_name);
CT_API int      ct_weight_slice_loaded_bytes(const ct_weight_registry_t *reg,
                                              const char *slice_name);

#ifdef __cplusplus
}
#endif
#endif /* CT_WEIGHT_PAGER_H */
