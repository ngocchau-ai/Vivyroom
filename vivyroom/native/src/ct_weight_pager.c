/*
 * ct_weight_pager.c — Weight Slice Registry & Sequential FFN Streaming.
 *
 * Reads real tensor bytes from GGUF, enforces RAM budget via LRU eviction,
 * supports sparse (top-k) activation, and performs real matvec computation.
 *
 * Gate 9: no performance claims in this file.
 */
#ifndef CT_PAGER_BUILD
#define CT_PAGER_BUILD 1
#endif
#include "ct_weight_pager.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */
static uint64_t now_ms(void) {
    return (uint64_t)(clock() / (CLOCKS_PER_SEC / 1000));
}

static ct_weight_slice_t *find_slice(ct_weight_registry_t *reg, const char *name) {
    for (uint32_t i = 0; i < reg->n_slices; i++) {
        if (strcmp(reg->slices[i].name, name) == 0) return &reg->slices[i];
    }
    return NULL;
}

static const ct_weight_slice_t *find_slice_const(const ct_weight_registry_t *reg,
                                                  const char *name) {
    for (uint32_t i = 0; i < reg->n_slices; i++) {
        if (strcmp(reg->slices[i].name, name) == 0) return &reg->slices[i];
    }
    return NULL;
}

/* LRU eviction: lowest (task_affinity, last_accessed_ms) first. */
static bool evict_lru_slice(ct_weight_registry_t *reg) {
    ct_weight_slice_t *victim = NULL;
    for (uint32_t i = 0; i < reg->n_slices; i++) {
        ct_weight_slice_t *s = &reg->slices[i];
        if (s->state != CT_SLICE_RESIDENT) continue;
        if (!victim) { victim = s; continue; }
        if (s->task_affinity < victim->task_affinity ||
            (s->task_affinity == victim->task_affinity &&
             s->last_accessed_ms < victim->last_accessed_ms)) {
            victim = s;
        }
    }
    if (!victim) return false;
    return ct_weight_slice_page_out(reg, victim->name) == CT_WP_OK;
}

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */

CT_API ct_wp_status_t ct_weight_registry_init(const char *model_path,
                                               uint64_t max_ram_bytes,
                                               ct_weight_registry_t **out) {
    if (!model_path || !out) return CT_WP_ERR_IO;
    *out = NULL;

    ct_gguf_file_t *gguf = NULL;
    ct_gguf_status_t gs = ct_gguf_open(model_path, &gguf);
    if (gs != CT_GGUF_OK) return (ct_wp_status_t)gs;

    ct_weight_registry_t *reg = (ct_weight_registry_t *)calloc(1, sizeof(*reg));
    if (!reg) { ct_gguf_close(gguf); return CT_WP_ERR_NOMEM; }

    reg->gguf = gguf;
    reg->max_ram_bytes = max_ram_bytes;
    reg->capacity = 256;
    reg->slices = (ct_weight_slice_t *)calloc(reg->capacity, sizeof(ct_weight_slice_t));
    if (!reg->slices) {
        ct_gguf_close(gguf);
        free(reg);
        return CT_WP_ERR_NOMEM;
    }

    *out = reg;
    return CT_WP_OK;
}

CT_API void ct_weight_registry_free(ct_weight_registry_t *reg) {
    if (!reg) return;
    for (uint32_t i = 0; i < reg->n_slices; i++) {
        if (reg->slices[i].buffer) free(reg->slices[i].buffer);
    }
    if (reg->slices) free(reg->slices);
    if (reg->gguf) ct_gguf_close(reg->gguf);
    free(reg);
}

CT_API int ct_weight_registry_build_from_gguf(ct_weight_registry_t *reg) {
    if (!reg || !reg->gguf) return (int)CT_WP_ERR_IO;

    uint64_t n = ct_gguf_tensor_count(reg->gguf);
    for (uint64_t i = 0; i < n; i++) {
        const ct_gguf_tensor_t *t = ct_gguf_tensor_at(reg->gguf, i);
        if (!t) continue;

        uint64_t abs_offset = ct_gguf_data_offset(reg->gguf) + t->offset;

        uint32_t layer_idx = 0;
        const char *blk = strstr(t->name, "blk.");
        if (blk) layer_idx = (uint32_t)atoi(blk + 4);
        else {
            const char *layers = strstr(t->name, "layers.");
            if (layers) layer_idx = (uint32_t)atoi(layers + 7);
        }

        ct_wp_status_t st = ct_weight_registry_register_slice(
            reg, t->name, abs_offset, t->size_bytes, layer_idx, t->type);
        if (st != CT_WP_OK) return (int)st;

        ct_weight_slice_t *s = &reg->slices[reg->n_slices - 1];
        s->n_dims = t->n_dims;
        memcpy(s->dims, t->dims, sizeof(t->dims));
    }
    return (int)reg->n_slices;
}

CT_API ct_wp_status_t ct_weight_registry_register_slice(
    ct_weight_registry_t *reg, const char *name, uint64_t file_offset,
    uint64_t size_bytes, uint32_t layer_index, ct_ggml_type_t quant_type) {
    if (!reg || !name) return CT_WP_ERR_IO;

    if (reg->n_slices >= reg->capacity) {
        uint32_t newcap = reg->capacity * 2;
        ct_weight_slice_t *ns = (ct_weight_slice_t *)realloc(
            reg->slices, newcap * sizeof(ct_weight_slice_t));
        if (!ns) return CT_WP_ERR_NOMEM;
        memset(ns + reg->capacity, 0,
               (newcap - reg->capacity) * sizeof(ct_weight_slice_t));
        reg->slices = ns;
        reg->capacity = newcap;
    }

    ct_weight_slice_t *s = &reg->slices[reg->n_slices];
    memset(s, 0, sizeof(*s));
    strncpy(s->name, name, CT_SLICE_NAME_MAX - 1);
    s->file_offset = file_offset;
    s->size_bytes = size_bytes;
    s->layer_index = layer_index;
    s->quant_type = quant_type;
    s->state = CT_SLICE_ON_DISK;
    s->task_affinity = 0.5f;
    s->n_dims = 1;
    s->dims[0] = size_bytes;

    reg->n_slices++;
    return CT_WP_OK;
}

/* -----------------------------------------------------------------------
 * Sequential paging
 * ----------------------------------------------------------------------- */

CT_API ct_wp_status_t ct_weight_slice_page_in(ct_weight_registry_t *reg,
                                               const char *slice_name,
                                               void **out_buffer_ptr) {
    if (!reg || !slice_name) return CT_WP_ERR_IO;

    ct_weight_slice_t *s = find_slice(reg, slice_name);
    if (!s) return CT_WP_ERR_NOT_FOUND;

    /* Idempotent: already resident */
    if (s->state == CT_SLICE_RESIDENT && s->buffer) {
        s->last_accessed_ms = now_ms();
        if (out_buffer_ptr) *out_buffer_ptr = s->buffer;
        return CT_WP_OK;
    }

    /* Enforce RAM budget via LRU eviction */
    while (reg->current_ram_bytes + s->size_bytes > reg->max_ram_bytes) {
        if (!evict_lru_slice(reg)) return CT_WP_ERR_BUDGET;
    }

    void *buf = malloc(s->size_bytes);
    if (!buf) return CT_WP_ERR_NOMEM;

    /* Read raw bytes from GGUF at known file_offset */
    if (ct_fseek(reg->gguf->fp, (ct_off_t)s->file_offset, SEEK_SET) != 0) {
        free(buf);
        return CT_WP_ERR_IO;
    }
    if (fread(buf, 1, s->size_bytes, reg->gguf->fp) != s->size_bytes) {
        free(buf);
        return CT_WP_ERR_IO;
    }

    s->buffer = buf;
    s->buffer_capacity = s->size_bytes;
    s->loaded_bytes = s->size_bytes;
    s->state = CT_SLICE_RESIDENT;
    s->last_accessed_ms = now_ms();
    reg->current_ram_bytes += s->size_bytes;

    if (out_buffer_ptr) *out_buffer_ptr = s->buffer;
    return CT_WP_OK;
}

CT_API ct_wp_status_t ct_weight_slice_page_out(ct_weight_registry_t *reg,
                                                const char *slice_name) {
    if (!reg || !slice_name) return CT_WP_ERR_IO;

    ct_weight_slice_t *s = find_slice(reg, slice_name);
    if (!s) return CT_WP_ERR_NOT_FOUND;

    if (s->state == CT_SLICE_RESIDENT && s->buffer) {
        free(s->buffer);
        s->buffer = NULL;
        s->buffer_capacity = 0;
        reg->current_ram_bytes -= s->loaded_bytes;
        s->loaded_bytes = 0;
        s->state = CT_SLICE_ON_DISK;
        return CT_WP_OK;
    }
    return CT_WP_ERR_STATE;
}

CT_API ct_wp_status_t ct_weight_slice_sparse_page_in(
    ct_weight_registry_t *reg, const char *slice_name, float top_k_ratio,
    void **out_buffer_ptr) {
    if (!reg || !slice_name) return CT_WP_ERR_IO;
    if (top_k_ratio <= 0.0f || top_k_ratio > 1.0f) return CT_WP_ERR_BOUNDS;

    ct_weight_slice_t *s = find_slice(reg, slice_name);
    if (!s) return CT_WP_ERR_NOT_FOUND;

    if (s->state == CT_SLICE_RESIDENT && s->buffer) {
        s->last_accessed_ms = now_ms();
        if (out_buffer_ptr) *out_buffer_ptr = s->buffer;
        return CT_WP_OK;
    }

    uint64_t sparse_bytes = (uint64_t)ceil((double)s->size_bytes * (double)top_k_ratio);
    if (sparse_bytes == 0) sparse_bytes = 1;
    if (sparse_bytes > s->size_bytes) sparse_bytes = s->size_bytes;

    while (reg->current_ram_bytes + sparse_bytes > reg->max_ram_bytes) {
        if (!evict_lru_slice(reg)) return CT_WP_ERR_BUDGET;
    }

    void *buf = malloc(sparse_bytes);
    if (!buf) return CT_WP_ERR_NOMEM;

    if (ct_fseek(reg->gguf->fp, (ct_off_t)s->file_offset, SEEK_SET) != 0) {
        free(buf);
        return CT_WP_ERR_IO;
    }
    if (fread(buf, 1, sparse_bytes, reg->gguf->fp) != sparse_bytes) {
        free(buf);
        return CT_WP_ERR_IO;
    }

    s->buffer = buf;
    s->buffer_capacity = sparse_bytes;
    s->loaded_bytes = sparse_bytes;
    s->state = CT_SLICE_RESIDENT;
    s->last_accessed_ms = now_ms();
    reg->current_ram_bytes += sparse_bytes;

    if (out_buffer_ptr) *out_buffer_ptr = s->buffer;
    return CT_WP_OK;
}

/* -----------------------------------------------------------------------
 * Streaming compute (real matvec)
 * ----------------------------------------------------------------------- */

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) { f32 = sign; }
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f32 = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = sign | 0x7F800000u | (mant << 13);
    } else {
        f32 = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f32, 4);
    return out;
}

static void dequant_q8_0(const uint8_t *blk, float *out) {
    float scale = f16_to_f32(*(const uint16_t *)blk);
    const int8_t *qs = (const int8_t *)(blk + 2);
    for (int i = 0; i < 32; i++) out[i] = scale * (float)qs[i];
}

static void dequant_q4_0(const uint8_t *blk, float *out) {
    float scale = f16_to_f32(*(const uint16_t *)blk);
    const uint8_t *qs = blk + 2;
    for (int i = 0; i < 16; i++) {
        int q0 = (qs[i] & 0x0F) - 8;
        int q1 = ((qs[i] >> 4) & 0x0F) - 8;
        out[2 * i] = scale * (float)q0;
        out[2 * i + 1] = scale * (float)q1;
    }
}

/* Q4_K: 144 bytes per 256 elements. Super-block with 8 sub-blocks of 32. */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void dequant_q4_k(const uint8_t *blk, float *out) {
    float d = f16_to_f32(*(const uint16_t *)blk);
    float dmin = f16_to_f32(*(const uint16_t *)(blk + 2));
    const uint8_t *scales = blk + 4;
    const uint8_t *qs = blk + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, mn;
        get_scale_min_k4(is, scales, &sc, &mn);
        float d1 = d * (float)sc;
        float m1 = dmin * (float)mn;
        get_scale_min_k4(is + 1, scales, &sc, &mn);
        float d2 = d * (float)sc;
        float m2 = dmin * (float)mn;
        for (int l = 0; l < 32; l++) out[j + l] = d1 * (float)(qs[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) out[j + 32 + l] = d2 * (float)(qs[l] >> 4) - m2;
        qs += 32;
        is += 2;
    }
}

/* Q6_K: 210 bytes per 256 elements. */
static void dequant_q6_k(const uint8_t *blk, float *out) {
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + 128;
    const int8_t *scales = (const int8_t *)(blk + 192);
    float d = f16_to_f32(*(const uint16_t *)(blk + 208));
    for (int i = 0; i < 256; i += 16) {
        int sidx = (i / 16) / 2;
        float sc = d * (float)scales[sidx];
        for (int j = 0; j < 16; j++) {
            int l = (ql[i + j] & 0xF) | (((qh[i / 4 + j / 4] >> (2 * (j % 4))) & 3) << 4);
            out[i + j] = sc * (float)(l - 32);
        }
    }
}

CT_API ct_wp_status_t ct_weight_slice_stream_compute(
    ct_weight_registry_t *reg, const char *slice_name,
    const float *input, size_t input_len,
    float *output, size_t output_len) {
    if (!reg || !slice_name || !input || !output) return CT_WP_ERR_IO;

    ct_weight_slice_t *s = find_slice(reg, slice_name);
    if (!s) return CT_WP_ERR_NOT_FOUND;
    if (s->state != CT_SLICE_RESIDENT || !s->buffer) {
        ct_wp_status_t st = ct_weight_slice_page_in(reg, slice_name, NULL);
        if (st != CT_WP_OK) return st;
        s = find_slice(reg, slice_name);
        if (!s || s->state != CT_SLICE_RESIDENT) return CT_WP_ERR_STATE;
    }

    /* Shape [rows, cols] */
    uint64_t rows, cols;
    if (s->n_dims >= 2) {
        rows = s->dims[s->n_dims - 2];
        cols = s->dims[s->n_dims - 1];
    } else {
        rows = 1;
        cols = s->dims[0];
    }

    if (input_len < cols) return CT_WP_ERR_BOUNDS;
    if (output_len < rows) return CT_WP_ERR_BOUNDS;

    /* Sparse buffers hold partial data — full matvec needs the complete tensor */
    if (s->loaded_bytes < s->size_bytes) return CT_WP_ERR_BOUNDS;

    const uint8_t *raw = (const uint8_t *)s->buffer;

    switch (s->quant_type) {
    case CT_GGML_F32: {
        const float *w = (const float *)raw;
        for (uint64_t i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (uint64_t j = 0; j < cols; j++) {
                sum += input[j] * w[i * cols + j];
            }
            output[i] = sum;
        }
        return CT_WP_OK;
    }
    case CT_GGML_F16: {
        const uint16_t *w = (const uint16_t *)raw;
        for (uint64_t i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (uint64_t j = 0; j < cols; j++) {
                sum += input[j] * f16_to_f32(w[i * cols + j]);
            }
            output[i] = sum;
        }
        return CT_WP_OK;
    }
    case CT_GGML_Q8_0: {
        uint64_t n_blocks = (cols + 31) / 32;
        for (uint64_t i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (uint64_t b = 0; b < n_blocks; b++) {
                float deq[32];
                dequant_q8_0(raw + (i * n_blocks + b) * 34, deq);
                uint64_t base = b * 32;
                for (uint64_t k = 0; k < 32 && (base + k) < cols; k++) {
                    sum += input[base + k] * deq[k];
                }
            }
            output[i] = sum;
        }
        return CT_WP_OK;
    }
    case CT_GGML_Q4_0: {
        uint64_t n_blocks = (cols + 31) / 32;
        for (uint64_t i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (uint64_t b = 0; b < n_blocks; b++) {
                float deq[32];
                dequant_q4_0(raw + (i * n_blocks + b) * 18, deq);
                uint64_t base = b * 32;
                for (uint64_t k = 0; k < 32 && (base + k) < cols; k++) {
                    sum += input[base + k] * deq[k];
                }
            }
            output[i] = sum;
        }
        return CT_WP_OK;
    }
    case CT_GGML_Q4_K: {
        uint64_t n_blocks = (cols + 255) / 256;
        for (uint64_t i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (uint64_t b = 0; b < n_blocks; b++) {
                float deq[256];
                dequant_q4_k(raw + (i * n_blocks + b) * 144, deq);
                uint64_t base = b * 256;
                for (uint64_t k = 0; k < 256 && (base + k) < cols; k++) {
                    sum += input[base + k] * deq[k];
                }
            }
            output[i] = sum;
        }
        return CT_WP_OK;
    }
    case CT_GGML_Q6_K: {
        uint64_t n_blocks = (cols + 255) / 256;
        for (uint64_t i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (uint64_t b = 0; b < n_blocks; b++) {
                float deq[256];
                dequant_q6_k(raw + (i * n_blocks + b) * 210, deq);
                uint64_t base = b * 256;
                for (uint64_t k = 0; k < 256 && (base + k) < cols; k++) {
                    sum += input[base + k] * deq[k];
                }
            }
            output[i] = sum;
        }
        return CT_WP_OK;
    }
    default:
        return CT_WP_ERR_UNSUPPORTED;
    }
}

/* -----------------------------------------------------------------------
 * Introspection
 * ----------------------------------------------------------------------- */

CT_API uint64_t ct_weight_registry_ram_usage(const ct_weight_registry_t *reg) {
    return reg ? reg->current_ram_bytes : 0;
}

CT_API int ct_weight_registry_resident_count(const ct_weight_registry_t *reg) {
    if (!reg) return 0;
    int n = 0;
    for (uint32_t i = 0; i < reg->n_slices; i++) {
        if (reg->slices[i].state == CT_SLICE_RESIDENT) n++;
    }
    return n;
}

CT_API int ct_weight_registry_slice_count(const ct_weight_registry_t *reg) {
    return reg ? (int)reg->n_slices : 0;
}

CT_API int ct_weight_slice_state(const ct_weight_registry_t *reg,
                                  const char *slice_name) {
    if (!reg || !slice_name) return (int)CT_SLICE_ON_DISK;
    const ct_weight_slice_t *s = find_slice_const(reg, slice_name);
    return s ? (int)s->state : (int)CT_SLICE_ON_DISK;
}

CT_API int ct_weight_slice_loaded_bytes(const ct_weight_registry_t *reg,
                                         const char *slice_name) {
    if (!reg || !slice_name) return 0;
    const ct_weight_slice_t *s = find_slice_const(reg, slice_name);
    return s ? (int)s->loaded_bytes : 0;
}
