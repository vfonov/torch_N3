#ifndef N3_SHIM_VERSION_H
#define N3_SHIM_VERSION_H
#define MNI_VERSION "1.12.00"
#define MNI_LONG_VERSION "Package MNI N3, version " MNI_VERSION \
   " (compiled into torch_n3's legacy oracle extension)"
#ifdef __cplusplus
extern "C" {
#endif
void print_version_info(char *version_string);
void set_program_name(char *name);
#ifdef __cplusplus
}
#endif
#endif
