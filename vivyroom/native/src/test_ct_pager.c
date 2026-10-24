/*
 * test_ct_pager.c — C-level unit tests for the weight pager.
 *
 * Creates synthetic GGUF fixtures, tests page_in/page_out, RAM budget,
 * sparse activation, stream_compute (F32 matvec), and LRU eviction.
 *
 * Run: make test
 */
#include "ct_weight_pager.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); } \
} while(0)

/* -----------------------------------------------------------------------
 * Helper: write a minimal synthetic GGUF file with F32 tensors
 * ----------------------------------------------------------------------- */
static void write_u32(FILE *fp, uint32_t v) { fwrite(&v, 4, 1, fp); }
static void write_u64(FILE *fp, uint64_t v) { fwrite(&v, 8, 1, fp); }
static void write_string_v2(FILE *fp, const char *s) {
    uint64_t len = strlen(s);
    write_u64(fp, len);
    fwrite(s, 1, (size_t)len, fp);
}

typedef struct {
    const char *name;
    uint64_t dims[2];
    uint32_t n_dims;
    uint32_t type;
    float *data;
} test_tensor_t;

static void write_synthetic_gguf(const char *path, test_tensor_t *tensors,
                                  uint64_t n_tensors) {
    FILE *fp = fopen(path, "wb");
    assert(fp);

    write_u32(fp, 0x46554747u);
    write_u32(fp, 2);
    write_u64(fp, n_tensors);
    write_u64(fp, 0);

    uint64_t data_offset = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        write_string_v2(fp, tensors[i].name);
        write_u32(fp, tensors[i].n_dims);
        for (uint32_t d = 0; d < tensors[i].n_dims; d++)
            write_u64(fp, tensors[i].dims[d]);
        write_u32(fp, tensors[i].type);
        write_u64(fp, data_offset);

        uint64_t n_el = 1;
        for (uint32_t d = 0; d < tensors[i].n_dims; d++)
            n_el *= tensors[i].dims[d];
        data_offset += n_el * 4;
    }

    long cur = ftell(fp);
    uint64_t aligned = ((uint64_t)cur + 31) / 32 * 32;
    for (uint64_t i = (uint64_t)cur; i < aligned; i++) fputc(0, fp);

    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t n_el = 1;
        for (uint32_t d = 0; d < tensors[i].n_dims; d++)
            n_el *= tensors[i].dims[d];
        fwrite(tensors[i].data, 4, n_el, fp);
    }
    fclose(fp);
}

/* -----------------------------------------------------------------------
 * Tests
 * ----------------------------------------------------------------------- */

static void test_gguf_open_and_find(void) {
    float data_a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data_b[6] = {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    test_tensor_t tensors[] = {
        {"blk.0.ffn_gate", {2, 2}, 2, 0, data_a},
        {"blk.0.ffn_up",   {3, 2}, 2, 0, data_b},
    };
    write_synthetic_gguf("test_fixture.gguf", tensors, 2);

    ct_gguf_file_t *g = NULL;
    ct_gguf_status_t st = ct_gguf_open("test_fixture.gguf", &g);
    CHECK(st == CT_GGUF_OK, "gguf_open");
    CHECK(g != NULL, "gguf_open returns non-null");
    CHECK(ct_gguf_tensor_count(g) == 2, "tensor count = 2");

    const ct_gguf_tensor_t *t0 = ct_gguf_find_tensor(g, "blk.0.ffn_gate");
    CHECK(t0 != NULL, "find blk.0.ffn_gate");
    CHECK(t0->size_bytes == 16, "tensor 0 size = 16 (4 floats)");

    const ct_gguf_tensor_t *t1 = ct_gguf_find_tensor(g, "blk.0.ffn_up");
    CHECK(t1 != NULL, "find blk.0.ffn_up");
    CHECK(t1->size_bytes == 24, "tensor 1 size = 24 (6 floats)");

    CHECK(ct_gguf_find_tensor(g, "nonexistent") == NULL, "missing tensor returns NULL");

    ct_gguf_close(g);
    printf("  test_gguf_open_and_find: OK\n");
}

static void test_registry_init_and_free(void) {
    float data[4] = {1, 2, 3, 4};
    test_tensor_t tensors[] = {
        {"blk.0.ffn", {2, 2}, 2, 0, data},
    };
    write_synthetic_gguf("test_fixture.gguf", tensors, 1);

    ct_weight_registry_t *reg = NULL;
    ct_wp_status_t st = ct_weight_registry_init("test_fixture.gguf", 1024 * 1024, &reg);
    CHECK(st == CT_WP_OK, "registry_init");
    CHECK(reg != NULL, "registry non-null");

    int n = ct_weight_registry_build_from_gguf(reg);
    CHECK(n == 1, "build_from_gguf returns 1 slice");
    CHECK(ct_weight_registry_slice_count(reg) == 1, "slice_count = 1");

    ct_weight_registry_free(reg);
    printf("  test_registry_init_and_free: OK\n");
}

static void test_page_in_reads_real_bytes(void) {
    float data[4] = {1.5f, -2.5f, 3.75f, 0.25f};
    test_tensor_t tensors[] = {
        {"test_tensor", {2, 2}, 2, 0, data},
    };
    write_synthetic_gguf("test_fixture.gguf", tensors, 1);

    ct_weight_registry_t *reg = NULL;
    ct_weight_registry_init("test_fixture.gguf", 1024 * 1024, &reg);
    ct_weight_registry_build_from_gguf(reg);

    void *buf = NULL;
    ct_wp_status_t st = ct_weight_slice_page_in(reg, "test_tensor", &buf);
    CHECK(st == CT_WP_OK, "page_in returns OK");
    CHECK(buf != NULL, "page_in returns buffer");

    const float *f = (const float *)buf;
    CHECK(fabsf(f[0] - 1.5f) < 1e-6f, "byte[0] = 1.5");
    CHECK(fabsf(f[1] - (-2.5f)) < 1e-6f, "byte[1] = -2.5");
    CHECK(fabsf(f[2] - 3.75f) < 1e-6f, "byte[2] = 3.75");
    CHECK(fabsf(f[3] - 0.25f) < 1e-6f, "byte[3] = 0.25");

    CHECK(ct_weight_registry_resident_count(reg) == 1, "resident count = 1");
    CHECK(ct_weight_registry_ram_usage(reg) == 16, "RAM usage = 16 bytes");

    ct_weight_registry_free(reg);
    printf("  test_page_in_reads_real_bytes: OK\n");
}

static void test_page_out_frees_memory(void) {
    float data[4] = {1, 2, 3, 4};
    test_tensor_t tensors[] = {
        {"t", {2, 2}, 2, 0, data},
    };
    write_synthetic_gguf("test_fixture.gguf", tensors, 1);

    ct_weight_registry_t *reg = NULL;
    ct_weight_registry_init("test_fixture.gguf", 1024 * 1024, &reg);
    ct_weight_registry_build_from_gguf(reg);

    ct_weight_slice_page_in(reg, "t", NULL);
    CHECK(ct_weight_registry_ram_usage(reg) > 0, "RAM > 0 after page_in");

    ct_wp_status_t st = ct_weight_slice_page_out(reg, "t");
    CHECK(st == CT_WP_OK, "page_out OK");
    CHECK(ct_weight_registry_ram_usage(reg) == 0, "RAM = 0 after page_out");
    CHECK(ct_weight_registry_resident_count(reg) == 0, "resident = 0 after page_out");

    ct_weight_registry_free(reg);
    printf("  test_page_out_frees_memory: OK\n");
}

static void test_ram_budget_eviction(void) {
    float data_a[16], data_b[16], data_c[16];
    memset(data_a, 0, sizeof(data_a));
    memset(data_b, 0, sizeof(data_b));
    memset(data_c, 0, sizeof(data_c));
    test_tensor_t tensors[] = {
        {"a", {4, 4}, 2, 0, data_a},
        {"b", {4, 4}, 2, 0, data_b},
        {"c", {4, 4}, 2, 0, data_c},
    };
    write_synthetic_gguf("test_fixture.gguf", tensors, 3);

    ct_weight_registry_t *reg = NULL;
    ct_weight_registry_init("test_fixture.gguf", 150, &reg);
    ct_weight_registry_build_from_gguf(reg);

    ct_weight_slice_page_in(reg, "a", NULL);
    ct_weight_slice_page_in(reg, "b", NULL);
    CHECK(ct_weight_registry_resident_count(reg) == 2, "2 resident after a,b");
    CHECK(ct_weight_registry_ram_usage(reg) <= 150, "RAM <= 150 after a,b");

    ct_weight_slice_page_in(reg, "c", NULL);
    CHECK(ct_weight_registry_resident_count(reg) <= 2, "at most 2 resident after c");
    CHECK(ct_weight_registry_ram_usage(reg) <= 150, "RAM <= 150 after c");
    CHECK(ct_weight_slice_state(reg, "c") == CT_SLICE_RESIDENT, "c is resident");

    ct_weight_registry_free(reg);
    printf("  test_ram_budget_eviction: OK\n");
}

static void test_sparse_page_in(void) {
    float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    test_tensor_t tensors[] = {
        {"t", {4, 2}, 2, 0, data},
    };
    write_synthetic_gguf("test_fixture.gguf", tensors, 1);

    ct_weight_registry_t *reg = NULL;
    ct_weight_registry_init("test_fixture.gguf", 1024 * 1024, &reg);
    ct_weight_registry_build_from_gguf(reg);

    void *buf = NULL;
    ct_wp_status_t st = ct_weight_slice_sparse_page_in(reg, "t", 0.5f, &buf);
    CHECK(st == CT_WP_OK, "sparse_page_in OK");
    CHECK(buf != NULL, "sparse buffer non-null");
    CHECK(ct_weight_slice_loaded_bytes(reg, "t") == 16, "loaded = 16 (50%)");
    CHECK(ct_weight_registry_ram_usage(reg) == 16, "RAM = 16 bytes");

    ct_weight_registry_free(reg);
    printf("  test_sparse_page_in: OK\n");
}

static void test_stream_compute_f32(void) {
    /* W = [[1,2,3],[4,5,6]], input = [1,1,1] -> output = [6, 15] */
    float w[] = {1, 2, 3, 4, 5, 6};
    test_tensor_t tensors[] = {
        {"W", {2, 3}, 2, 0, w},
    };
    write_synthetic_gguf("test_fixture.gguf", tensors, 1);

    ct_weight_registry_t *reg = NULL;
    ct_weight_registry_init("test_fixture.gguf", 1024 * 1024, &reg);
    ct_weight_registry_build_from_gguf(reg);

    float input[3] = {1.0f, 1.0f, 1.0f};
    float output[2] = {0};
    ct_wp_status_t st = ct_weight_slice_stream_compute(reg, "W", input, 3, output, 2);
    CHECK(st == CT_WP_OK, "stream_compute OK");
    CHECK(fabsf(output[0] - 6.0f) < 1e-5f, "output[0] = 6");
    CHECK(fabsf(output[1] - 15.0f) < 1e-5f, "output[1] = 15");

    ct_weight_registry_free(reg);
    printf("  test_stream_compute_f32: OK\n");
}

static void test_stream_compute_q8_0(void) {
    /* Q8_0 block: f16 scale (=1.0) + 32 int8 values (=1 each) => all 1.0 */
    uint8_t block[34];
    block[0] = 0x00; block[1] = 0x3C;  /* f16 1.0 */
    for (int i = 0; i < 32; i++) block[2 + i] = (uint8_t)1;

    FILE *fp = fopen("test_fixture.gguf", "wb");
    write_u32(fp, 0x46554747u);
    write_u32(fp, 2);
    write_u64(fp, 1);
    write_u64(fp, 0);
    write_string_v2(fp, "q8t");
    write_u32(fp, 1);
    write_u64(fp, 32);
    write_u32(fp, 8);   /* CT_GGML_Q8_0 */
    write_u64(fp, 0);
    long cur = ftell(fp);
    uint64_t aligned = ((uint64_t)cur + 31) / 32 * 32;
    for (uint64_t i = (uint64_t)cur; i < aligned; i++) fputc(0, fp);
    fwrite(block, 1, 34, fp);
    fclose(fp);

    ct_weight_registry_t *reg = NULL;
    ct_weight_registry_init("test_fixture.gguf", 1024 * 1024, &reg);
    ct_weight_registry_build_from_gguf(reg);

    float input[32];
    for (int i = 0; i < 32; i++) input[i] = 1.0f;
    float output[1] = {0};
    ct_wp_status_t st = ct_weight_slice_stream_compute(reg, "q8t", input, 32, output, 1);
    CHECK(st == CT_WP_OK, "q8_0 stream_compute OK");
    CHECK(fabsf(output[0] - 32.0f) < 0.5f, "q8_0 output ~ 32");

    ct_weight_registry_free(reg);
    printf("  test_stream_compute_q8_0: OK\n");
}

static void test_double_page_in_idempotent(void) {
    float data[4] = {1, 2, 3, 4};
    test_tensor_t tensors[] = {
        {"t", {2, 2}, 2, 0, data},
    };
    write_synthetic_gguf("test_fixture.gguf", tensors, 1);

    ct_weight_registry_t *reg = NULL;
    ct_weight_registry_init("test_fixture.gguf", 1024 * 1024, &reg);
    ct_weight_registry_build_from_gguf(reg);

    void *buf1 = NULL;
    ct_weight_slice_page_in(reg, "t", &buf1);
    uint64_t ram1 = ct_weight_registry_ram_usage(reg);

    void *buf2 = NULL;
    ct_weight_slice_page_in(reg, "t", &buf2);
    uint64_t ram2 = ct_weight_registry_ram_usage(reg);

    CHECK(buf1 == buf2, "same buffer on double page_in");
    CHECK(ram1 == ram2, "RAM unchanged on double page_in");
    CHECK(ct_weight_registry_resident_count(reg) == 1, "still 1 resident");

    ct_weight_registry_free(reg);
    printf("  test_double_page_in_idempotent: OK\n");
}

static void test_sequential_cycle(void) {
    float data[16];
    for (int i = 0; i < 16; i++) data[i] = (float)i;
    test_tensor_t tensors[] = {
        {"t", {4, 4}, 2, 0, data},
    };
    write_synthetic_gguf("test_fixture.gguf", tensors, 1);

    ct_weight_registry_t *reg = NULL;
    ct_weight_registry_init("test_fixture.gguf", 1024 * 1024, &reg);
    ct_weight_registry_build_from_gguf(reg);

    for (int i = 0; i < 10; i++) {
        ct_weight_slice_page_in(reg, "t", NULL);
        CHECK(ct_weight_registry_ram_usage(reg) == 64, "RAM stable at 64");
        ct_weight_slice_page_out(reg, "t");
        CHECK(ct_weight_registry_ram_usage(reg) == 0, "RAM = 0 after out");
    }

    ct_weight_registry_free(reg);
    printf("  test_sequential_cycle: OK\n");
}

static void test_corrupt_gguf_rejected(void) {
    FILE *fp = fopen("test_corrupt.gguf", "wb");
    write_u32(fp, 0xDEADBEEFu);
    write_u32(fp, 2);
    write_u64(fp, 0);
    write_u64(fp, 0);
    fclose(fp);

    ct_gguf_file_t *g = NULL;
    ct_gguf_status_t st = ct_gguf_open("test_corrupt.gguf", &g);
    CHECK(st == CT_GGUF_ERR_MAGIC, "corrupt magic rejected");
    CHECK(g == NULL, "corrupt file returns NULL");

    printf("  test_corrupt_gguf_rejected: OK\n");
}

static void test_invalid_slice_name(void) {
    float data[4] = {1, 2, 3, 4};
    test_tensor_t tensors[] = {
        {"t", {2, 2}, 2, 0, data},
    };
    write_synthetic_gguf("test_fixture.gguf", tensors, 1);

    ct_weight_registry_t *reg = NULL;
    ct_weight_registry_init("test_fixture.gguf", 1024 * 1024, &reg);
    ct_weight_registry_build_from_gguf(reg);

    ct_wp_status_t st = ct_weight_slice_page_in(reg, "nonexistent", NULL);
    CHECK(st == CT_WP_ERR_NOT_FOUND, "invalid slice returns NOT_FOUND");

    ct_weight_registry_free(reg);
    printf("  test_invalid_slice_name: OK\n");
}

int main(void) {
    printf("=== Cautreo Weight Pager C Tests ===\n");
    test_gguf_open_and_find();
    test_registry_init_and_free();
    test_page_in_reads_real_bytes();
    test_page_out_frees_memory();
    test_ram_budget_eviction();
    test_sparse_page_in();
    test_stream_compute_f32();
    test_stream_compute_q8_0();
    test_double_page_in_idempotent();
    test_sequential_cycle();
    test_corrupt_gguf_rejected();
    test_invalid_slice_name();

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
