/*
 * ct_gguf_reader.c — GGUF Lazy Tensor Reader implementation.
 *
 * Parses GGUF v2/v3 header, metadata KV pairs, and tensor index.
 * Provides lazy raw-byte access to individual tensors by file offset.
 *
 * Gate 9: no performance claims in this file.
 */
#ifndef CT_PAGER_BUILD
#define CT_PAGER_BUILD 1
#endif
#include "ct_gguf_reader.h"

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static FILE *open_file_utf8(const char *path) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    if (wlen <= 0) return NULL;
    wchar_t *wpath = (wchar_t *)malloc((size_t)wlen * sizeof(wchar_t));
    if (!wpath) return NULL;
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen);
    FILE *fp = _wfopen(wpath, L"rb");
    free(wpath);
    return fp;
}
#else
static FILE *open_file_utf8(const char *path) {
    return fopen(path, "rb");
}
#endif

/* -----------------------------------------------------------------------
 * Internal helpers
 * ----------------------------------------------------------------------- */

#define GGUF_MAGIC 0x46554747u  /* "GGUF" little-endian */
#define GGUF_DATA_ALIGNMENT 32u

static uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

static bool read_u32(FILE *fp, uint32_t *out) {
    return fread(out, sizeof(uint32_t), 1, fp) == 1;
}

static bool read_u64(FILE *fp, uint64_t *out) {
    return fread(out, sizeof(uint64_t), 1, fp) == 1;
}

/* Read a GGUF string (always u64 length per spec). Caller must free. */
static char *read_string(FILE *fp, uint32_t version) {
    (void)version;
    uint64_t len;
    if (!read_u64(fp, &len)) return NULL;
    if (len > 65536) return NULL;
    char *buf = (char *)malloc(len + 1);
    if (!buf) return NULL;
    if (len > 0 && fread(buf, 1, len, fp) != len) {
        free(buf);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

/* Skip a GGUF value of the given type without storing it. */
static bool skip_value(FILE *fp, uint32_t version, uint32_t vtype) {
    switch (vtype) {
    case 0: case 1: case 7:  /* UINT8, INT8, BOOL */
        return ct_fseek(fp, 1, SEEK_CUR) == 0;
    case 2: case 3:          /* UINT16, INT16 */
        return ct_fseek(fp, 2, SEEK_CUR) == 0;
    case 4: case 5: case 6:  /* UINT32, INT32, FLOAT32 */
        return ct_fseek(fp, 4, SEEK_CUR) == 0;
    case 10: case 11: case 12: /* UINT64, INT64, FLOAT64 */
        return ct_fseek(fp, 8, SEEK_CUR) == 0;
    case 8: { /* STRING — skip without reading (may be very long) */
        uint64_t len;
        if (!read_u64(fp, &len)) return false;
        return ct_fseek(fp, (ct_off_t)len, SEEK_CUR) == 0;
    }
    case 9: { /* ARRAY */
        uint32_t atype;
        uint64_t n;
        if (!read_u32(fp, &atype)) return false;
        if (!read_u64(fp, &n)) return false;
        for (uint64_t i = 0; i < n; i++) {
            if (!skip_value(fp, version, atype)) return false;
        }
        return true;
    }
    default:
        return false;
    }
}

/* Compute raw byte size for a tensor. */
static uint64_t compute_tensor_size(ct_ggml_type_t type, const uint64_t *dims,
                                     uint32_t n_dims) {
    if (n_dims == 0 || n_dims > 4) return 0;
    uint64_t n_elements = 1;
    for (uint32_t i = 0; i < n_dims; i++) n_elements *= dims[i];

    switch (type) {
    case CT_GGML_F32:  return n_elements * 4;
    case CT_GGML_F16:  return n_elements * 2;
    case CT_GGML_I8:   return n_elements;
    case CT_GGML_I16:  return n_elements * 2;
    case CT_GGML_I32:  return n_elements * 4;
    case CT_GGML_I64:  return n_elements * 8;
    case CT_GGML_F64:  return n_elements * 8;
    case CT_GGML_Q4_0: return (n_elements / 32) * 18;
    case CT_GGML_Q4_1: return (n_elements / 32) * 20;
    case CT_GGML_Q5_0: return (n_elements / 32) * 22;
    case CT_GGML_Q5_1: return (n_elements / 32) * 24;
    case CT_GGML_Q8_0: return (n_elements / 32) * 34;
    case CT_GGML_Q8_1: return (n_elements / 32) * 36;
    case CT_GGML_Q2_K: return (n_elements / 256) * 84;
    case CT_GGML_Q3_K: return (n_elements / 256) * 110;
    case CT_GGML_Q4_K: return (n_elements / 256) * 144;
    case CT_GGML_Q5_K: return (n_elements / 256) * 176;
    case CT_GGML_Q6_K: return (n_elements / 256) * 210;
    case CT_GGML_Q8_K: return (n_elements / 256) * 292;
    case CT_GGML_IQ4_NL: return (n_elements / 32) * 18;
    case CT_GGML_IQ1_M:  return (n_elements / 256) * 52;
    case CT_GGML_BF16:   return n_elements * 2;
    default: return 0;
    }
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

CT_API ct_gguf_status_t ct_gguf_open(const char *path, ct_gguf_file_t **out) {
    if (!path || !out) return CT_GGUF_ERR_IO;
    *out = NULL;

    FILE *fp = open_file_utf8(path);
    if (!fp) return CT_GGUF_ERR_IO;

    if (ct_fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return CT_GGUF_ERR_IO; }
    ct_off_t fsz = ct_ftell(fp);
    if (fsz < 24) { fclose(fp); return CT_GGUF_ERR_CORRUPT; }
    uint64_t file_size = (uint64_t)fsz;
    if (ct_fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return CT_GGUF_ERR_IO; }

    uint32_t magic;
    if (!read_u32(fp, &magic) || magic != GGUF_MAGIC) {
        fclose(fp);
        return CT_GGUF_ERR_MAGIC;
    }

    uint32_t version;
    if (!read_u32(fp, &version) || version < 2 || version > 3) {
        fclose(fp);
        return CT_GGUF_ERR_VERSION;
    }

    uint64_t n_tensors, n_kv;
    if (!read_u64(fp, &n_tensors) || !read_u64(fp, &n_kv)) {
        fclose(fp);
        return CT_GGUF_ERR_CORRUPT;
    }
    if (n_tensors > 1000000 || n_kv > 100000) {
        fclose(fp);
        return CT_GGUF_ERR_CORRUPT;
    }

    /* Skip KV pairs */
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = read_string(fp, version);
        if (!key) { fclose(fp); return CT_GGUF_ERR_CORRUPT; }
        free(key);
        uint32_t vtype;
        if (!read_u32(fp, &vtype)) { fclose(fp); return CT_GGUF_ERR_CORRUPT; }
        if (!skip_value(fp, version, vtype)) { fclose(fp); return CT_GGUF_ERR_CORRUPT; }
    }

    ct_gguf_file_t *g = (ct_gguf_file_t *)calloc(1, sizeof(ct_gguf_file_t));
    if (!g) { fclose(fp); return CT_GGUF_ERR_NOMEM; }
    g->fp = fp;
    g->version = version;
    g->n_tensors = n_tensors;
    g->n_kv = n_kv;
    g->file_size = file_size;

    if (n_tensors > 0) {
        g->tensors = (ct_gguf_tensor_t *)calloc(n_tensors, sizeof(ct_gguf_tensor_t));
        if (!g->tensors) { free(g); fclose(fp); return CT_GGUF_ERR_NOMEM; }
    }

    /* Parse tensor infos */
    for (uint64_t i = 0; i < n_tensors; i++) {
        ct_gguf_tensor_t *t = &g->tensors[i];

        char *name = read_string(fp, version);
        if (!name) { ct_gguf_close(g); return CT_GGUF_ERR_CORRUPT; }
        strncpy(t->name, name, CT_GGUF_NAME_MAX - 1);
        t->name[CT_GGUF_NAME_MAX - 1] = '\0';
        free(name);

        uint32_t n_dims;
        if (!read_u32(fp, &n_dims) || n_dims == 0 || n_dims > 4) {
            ct_gguf_close(g);
            return CT_GGUF_ERR_CORRUPT;
        }
        t->n_dims = n_dims;

        for (uint32_t d = 0; d < n_dims; d++) {
            if (!read_u64(fp, &t->dims[d])) {
                ct_gguf_close(g);
                return CT_GGUF_ERR_CORRUPT;
            }
        }

        uint32_t dtype;
        if (!read_u32(fp, &dtype)) {
            ct_gguf_close(g);
            return CT_GGUF_ERR_CORRUPT;
        }
        t->type = (ct_ggml_type_t)dtype;

        if (!read_u64(fp, &t->offset)) {
            ct_gguf_close(g);
            return CT_GGUF_ERR_CORRUPT;
        }

        t->size_bytes = compute_tensor_size(t->type, t->dims, t->n_dims);
        if (t->size_bytes == 0) {
            ct_gguf_close(g);
            return CT_GGUF_ERR_CORRUPT;
        }
    }

    /* Data section starts at next 32-byte aligned offset */
    ct_off_t cur_off = ct_ftell(fp);
    g->data_offset = align_up((uint64_t)cur_off, GGUF_DATA_ALIGNMENT);

    *out = g;
    return CT_GGUF_OK;
}

CT_API void ct_gguf_close(ct_gguf_file_t *g) {
    if (!g) return;
    if (g->fp) fclose(g->fp);
    if (g->tensors) free(g->tensors);
    free(g);
}

CT_API const ct_gguf_tensor_t *ct_gguf_find_tensor(const ct_gguf_file_t *g,
                                                    const char *name) {
    if (!g || !name) return NULL;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return NULL;
}

CT_API const ct_gguf_tensor_t *ct_gguf_tensor_at(const ct_gguf_file_t *g,
                                                   uint64_t index) {
    if (!g || index >= g->n_tensors) return NULL;
    return &g->tensors[index];
}

CT_API ct_gguf_status_t ct_gguf_read_tensor_data(ct_gguf_file_t *g,
                                                  const ct_gguf_tensor_t *info,
                                                  void *buf, size_t buf_size) {
    if (!g || !info || !buf) return CT_GGUF_ERR_IO;
    if (buf_size < info->size_bytes) return CT_GGUF_ERR_BOUNDS;

    uint64_t file_pos = g->data_offset + info->offset;
    if (file_pos + info->size_bytes > g->file_size) return CT_GGUF_ERR_BOUNDS;

    if (ct_fseek(g->fp, (ct_off_t)file_pos, SEEK_SET) != 0) return CT_GGUF_ERR_IO;
    if (fread(buf, 1, info->size_bytes, g->fp) != info->size_bytes)
        return CT_GGUF_ERR_IO;

    return CT_GGUF_OK;
}

CT_API uint64_t ct_gguf_tensor_count(const ct_gguf_file_t *g) {
    return g ? g->n_tensors : 0;
}

CT_API uint64_t ct_gguf_data_offset(const ct_gguf_file_t *g) {
    return g ? g->data_offset : 0;
}

CT_API uint64_t ct_gguf_type_size(ct_ggml_type_t type, const uint64_t *dims,
                                   uint32_t n_dims) {
    return compute_tensor_size(type, dims, n_dims);
}
