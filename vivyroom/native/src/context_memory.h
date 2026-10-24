#ifndef CAUTREO_CONTEXT_MEMORY_H
#define CAUTREO_CONTEXT_MEMORY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CT_MEMORY_ID_MAX 96
#define CT_MEMORY_HASH_MAX 96 /* algorithm prefix plus a 64-byte hex digest */
#define CT_MEMORY_CONTENT_MAX (1024u * 1024u)

typedef enum {
    CT_MEMORY_HARD_FACT = 0,
    CT_MEMORY_TASK,
    CT_MEMORY_CONSTRAINT,
    CT_MEMORY_SUMMARY
} ct_memory_kind_t;

typedef enum { CT_MEMORY_PERSIST_NONE = 0, CT_MEMORY_PERSIST_COLD_LOG } ct_memory_persistence_t;
typedef enum {
    CT_MEMORY_OK = 0, CT_MEMORY_INVALID = -1, CT_MEMORY_NOMEM = -2,
    CT_MEMORY_IO = -3, CT_MEMORY_CORRUPT = -4, CT_MEMORY_CONFLICT = -5,
    CT_MEMORY_NOT_FOUND = -6
} ct_memory_status_t;

typedef struct {
    const char *message_id;
    const char *tool_call_id;
    uint64_t source_offset;
    uint64_t source_length;
    const char *source_hash; /* caller-computed exact-source hash, e.g. SHA-256 */
} ct_memory_provenance_t;

typedef struct {
    const char *id;
    ct_memory_kind_t kind;
    const char *content;
    size_t content_len; /* SIZE_MAX means strlen */
    ct_memory_provenance_t provenance;
    uint64_t created_at_ms;
    uint64_t expires_at_ms; /* 0 means no TTL */
} ct_memory_put_t;

typedef struct {
    char id[CT_MEMORY_ID_MAX];
    ct_memory_kind_t kind;
    char *content;
    size_t content_len;
    char message_id[CT_MEMORY_ID_MAX];
    char tool_call_id[CT_MEMORY_ID_MAX];
    uint64_t source_offset, source_length;
    char source_hash[CT_MEMORY_HASH_MAX];
    uint64_t version, created_at_ms, updated_at_ms, expires_at_ms;
    bool deleted;
} ct_memory_item_t;

typedef struct {
    ct_memory_persistence_t persistence;
    const char *cold_log_path; /* required only for COLD_LOG; borrowed during open */
    size_t max_items;           /* 0 => 1024 */
} ct_memory_options_t;

typedef struct ct_context_memory ct_context_memory_t;

ct_memory_status_t ct_context_memory_open(const ct_memory_options_t *options,
                                          ct_context_memory_t **out);
void ct_context_memory_close(ct_context_memory_t *memory);

ct_memory_status_t ct_context_memory_put(ct_context_memory_t *memory,
                                         const ct_memory_put_t *put,
                                         uint64_t expected_version,
                                         uint64_t *out_version);
ct_memory_status_t ct_context_memory_delete(ct_context_memory_t *memory,
                                            const char *id,
                                            uint64_t expected_version,
                                            uint64_t now_ms);
const ct_memory_item_t *ct_context_memory_get(const ct_context_memory_t *memory,
                                              const char *id, uint64_t now_ms);

/* Deterministic lexical baseline, sorted by score then id. Borrowed item pointers. */
size_t ct_context_memory_retrieve(const ct_context_memory_t *memory,
                                  const char *query, uint64_t now_ms,
                                  const ct_memory_item_t **out, size_t out_cap);
size_t ct_context_memory_count(const ct_context_memory_t *memory, uint64_t now_ms);

/* Flushes the append-only log; replay validates every record checksum. */
ct_memory_status_t ct_context_memory_checkpoint(ct_context_memory_t *memory);

#ifdef __cplusplus
}
#endif
#endif
