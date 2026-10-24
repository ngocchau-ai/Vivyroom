/*
 * ct_gguf_reader.h — GGUF Lazy Tensor Reader for Cautreo Weight Pager.
 *
 * Parses GGUF header/metadata/tensor-index WITHOUT loading weight data.
 * Provides lazy tensor access: read raw bytes at any tensor's file offset.
 *
 * Spec: SPEC-CAUTREO-WEIGHT-PAGING-R4 (Sprint R4)
 * Gate 9: no performance claims — measurements go to benchmark receipts.
 */
#ifndef CT_GGUF_READER_H
#define CT_GGUF_READER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Export macro
 * ----------------------------------------------------------------------- */
#ifdef _WIN32
  #ifdef CT_PAGER_BUILD
    #define CT_API __declspec(dllexport)
  #else
    #define CT_API __declspec(dllimport)
  #endif
#else
  #define CT_API __attribute__((visibility("default")))
#endif

/* -----------------------------------------------------------------------
 * Large-file-safe seek/tell (Windows: long is 32-bit)
 * ----------------------------------------------------------------------- */
#ifdef _WIN32
  #define ct_ftell(f)        _ftelli64(f)
  #define ct_fseek(f, off, w) _fseeki64(f, off, w)
  typedef long long ct_off_t;
#else
  #define ct_ftell(f)        ftello(f)
  #define ct_fseek(f, off, w) fseeko(f, off, w)
  typedef off_t ct_off_t;
#endif

/* -----------------------------------------------------------------------
 * Error codes
 * ----------------------------------------------------------------------- */
typedef enum {
    CT_GGUF_OK            =  0,
    CT_GGUF_ERR_IO        = -1,
    CT_GGUF_ERR_MAGIC     = -2,
    CT_GGUF_ERR_VERSION   = -3,
    CT_GGUF_ERR_CORRUPT   = -4,
    CT_GGUF_ERR_NOMEM     = -5,
    CT_GGUF_ERR_NOT_FOUND = -6,
    CT_GGUF_ERR_BOUNDS    = -7,
} ct_gguf_status_t;

/* -----------------------------------------------------------------------
 * GGML type identifiers (subset used for weight tensors)
 * ----------------------------------------------------------------------- */
typedef enum {
    CT_GGML_F32     = 0,
    CT_GGML_F16     = 1,
    CT_GGML_Q4_0    = 2,
    CT_GGML_Q4_1    = 3,
    CT_GGML_Q5_0    = 6,
    CT_GGML_Q5_1    = 7,
    CT_GGML_Q8_0    = 8,
    CT_GGML_Q8_1    = 9,
    CT_GGML_Q2_K    = 10,
    CT_GGML_Q3_K    = 11,
    CT_GGML_Q4_K    = 12,
    CT_GGML_Q5_K    = 13,
    CT_GGML_Q6_K    = 14,
    CT_GGML_Q8_K    = 15,
    CT_GGML_IQ2_XXS = 16,
    CT_GGML_IQ2_XS  = 17,
    CT_GGML_IQ3_XXS = 18,
    CT_GGML_IQ1_S   = 19,
    CT_GGML_IQ4_NL  = 20,
    CT_GGML_IQ3_S   = 21,
    CT_GGML_IQ2_S   = 22,
    CT_GGML_IQ4_XS  = 23,
    CT_GGML_I8      = 24,
    CT_GGML_I16     = 25,
    CT_GGML_I32     = 26,
    CT_GGML_I64     = 27,
    CT_GGML_F64     = 28,
    CT_GGML_IQ1_M   = 29,
    CT_GGML_BF16    = 30,
} ct_ggml_type_t;

/* -----------------------------------------------------------------------
 * Tensor info entry
 * ----------------------------------------------------------------------- */
#define CT_GGUF_NAME_MAX 256

typedef struct {
    char     name[CT_GGUF_NAME_MAX];
    uint32_t n_dims;
    uint64_t dims[4];
    ct_ggml_type_t type;
    uint64_t offset;       /* byte offset from data section start */
    uint64_t size_bytes;   /* computed raw byte size of this tensor */
} ct_gguf_tensor_t;

/* -----------------------------------------------------------------------
 * GGUF file handle (lazy: only header + index in memory)
 * ----------------------------------------------------------------------- */
typedef struct {
    FILE            *fp;
    uint32_t         version;
    uint64_t         n_tensors;
    uint64_t         n_kv;
    ct_gguf_tensor_t *tensors;
    uint64_t         data_offset;  /* file offset where tensor data begins */
    uint64_t         file_size;
} ct_gguf_file_t;

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */

/* Open GGUF file, parse header + tensor index. Weights are NOT loaded. */
CT_API ct_gguf_status_t ct_gguf_open(const char *path, ct_gguf_file_t **out);

/* Close and free all resources. */
CT_API void ct_gguf_close(ct_gguf_file_t *g);

/* -----------------------------------------------------------------------
 * Tensor access
 * ----------------------------------------------------------------------- */

/* Find tensor by exact name. Returns NULL if not found. */
CT_API const ct_gguf_tensor_t *ct_gguf_find_tensor(const ct_gguf_file_t *g,
                                                    const char *name);

/* Get tensor by index. Returns NULL if index out of range. */
CT_API const ct_gguf_tensor_t *ct_gguf_tensor_at(const ct_gguf_file_t *g,
                                                   uint64_t index);

/*
 * Read raw tensor data into caller-supplied buffer.
 * Validates: offset + size_bytes <= file_size, buf_size >= size_bytes.
 * Returns CT_GGUF_OK on success.
 */
CT_API ct_gguf_status_t ct_gguf_read_tensor_data(ct_gguf_file_t *g,
                                                  const ct_gguf_tensor_t *info,
                                                  void *buf, size_t buf_size);

/* -----------------------------------------------------------------------
 * Introspection
 * ----------------------------------------------------------------------- */
CT_API uint64_t ct_gguf_tensor_count(const ct_gguf_file_t *g);
CT_API uint64_t ct_gguf_data_offset(const ct_gguf_file_t *g);

/*
 * Compute raw byte size for a tensor given its dims and ggml type.
 * Returns 0 for unsupported types.
 */
CT_API uint64_t ct_gguf_type_size(ct_ggml_type_t type, const uint64_t *dims,
                                   uint32_t n_dims);

#ifdef __cplusplus
}
#endif
#endif /* CT_GGUF_READER_H */
