#ifndef CAUTREO_PLUGIN_H
#define CAUTREO_PLUGIN_H

/*
 * plugin.h - CAUTREO Native Tool & Plugin Interface (Sprint C2/C3)
 *
 * Cho phép CAUTREO hoạt động như một AI Agent Engine độc lập không cần Ollama.
 * Hỗ trợ:
 *   - Khai báo tool (name, description, schema/usage)
 *   - Đăng ký plugin (websearch, local file io, system diagnostic)
 *   - Parse tool call từ output của model ([TOOL_CALL: name {args}])
 *   - Tự động thực thi và trả kết quả ([TOOL_RESULT: name] ...)
 *
 * CHANGELOG:
 *   21/09/2026 - Antigravity IDE (HoH Protocol): Khởi tạo Sprint C2 Plugin Architecture.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CT_PLUGIN_MAX_NAME 64
#define CT_PLUGIN_MAX_DESC 256
#define CT_PLUGIN_MAX_TOOLS 32
#define CT_PLUGIN_MAX_PLUGINS 16

/* Hàm thực thi tool. Nhận chuỗi args (JSON hoặc text), trả về kết quả malloc'd (caller free). */
typedef char *(*ct_tool_exec_fn)(const char *args, void *userdata);

typedef struct {
    char            name[CT_PLUGIN_MAX_NAME];
    char            description[CT_PLUGIN_MAX_DESC];
    char            usage_hint[CT_PLUGIN_MAX_DESC];
    ct_tool_exec_fn exec_fn;
    void           *userdata;
} ct_tool_def_t;

typedef struct ct_plugin {
    char            name[CT_PLUGIN_MAX_NAME];
    char            version[32];
    uint32_t        n_tools;
    ct_tool_def_t   tools[CT_PLUGIN_MAX_TOOLS];
    bool          (*init_fn)(struct ct_plugin *self);
    void          (*destroy_fn)(struct ct_plugin *self);
} ct_plugin_t;

typedef struct ct_plugin_registry ct_plugin_registry_t;

/* Khởi tạo & giải phóng registry */
ct_plugin_registry_t *ct_plugin_registry_create(void);
void                  ct_plugin_registry_destroy(ct_plugin_registry_t *reg);

/* Đăng ký plugin */
bool ct_plugin_register(ct_plugin_registry_t *reg, const ct_plugin_t *plugin);

/* Tìm kiếm tool theo tên */
const ct_tool_def_t *ct_plugin_find_tool(const ct_plugin_registry_t *reg, const char *name);

/* Thực thi tool theo tên */
char *ct_plugin_dispatch(const ct_plugin_registry_t *reg, const char *tool_name, const char *args);

/* Sinh đoạn prompt mô tả tất cả các tool hiện có để nhúng vào system prompt */
char *ct_plugin_build_system_prompt(const ct_plugin_registry_t *reg);

/* Parse tool call từ output của model:
 * Hỗ trợ định dạng: [TOOL_CALL: name args] hoặc ```tool_call\nname args\n```
 * Trả về true nếu phát hiện tool call hợp lệ. */
bool ct_plugin_parse_call(const char *text,
                          char *out_tool_name, size_t name_size,
                          char *out_args, size_t args_size);

/* Builtin plugins (Sprint C3) */
const ct_plugin_t *ct_builtin_websearch_plugin_get(void);
const ct_plugin_t *ct_builtin_systools_plugin_get(void);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_PLUGIN_H */
