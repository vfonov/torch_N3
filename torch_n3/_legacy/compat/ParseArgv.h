/*
 * ParseArgv.h -- stand-in for MINC's, so SharpenHist/args.{h,cc} compile
 * unmodified without linking libminc2.
 *
 * `args` is constructed only by sharpen_hist.cc's main(), which the build
 * renames away: the shim calls the file's helper functions directly and never
 * parses a command line.  args.cc still has to compile, though, and its
 * static option table needs the ArgvInfo layout and the ARGV_* tags to be the
 * real ones -- so those are reproduced exactly.  Only the parser itself is
 * replaced, by one that aborts if it is ever reached.
 */

#ifndef N3_COMPAT_PARSEARGV_H
#define N3_COMPAT_PARSEARGV_H

#include <stdio.h>
#include <stdlib.h>

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

typedef struct {
  const char *key;
  int type;
  const char *src;
  void *dst;
  const char *help;
} ArgvInfo;

#define ARGV_CONSTANT 15
#define ARGV_INT 16
#define ARGV_STRING 17
#define ARGV_LONG 100
#define ARGV_REST 19
#define ARGV_FLOAT 20
#define ARGV_FUNC 21
#define ARGV_GENFUNC 22
#define ARGV_HELP 23
#define ARGV_VERSION 24
#define ARGV_VERINFO 25
#define ARGV_END 27

#define ARGV_NO_DEFAULTS 0x1
#define ARGV_NO_LEFTOVERS 0x2
#define ARGV_NO_ABBREV 0x4
#define ARGV_DONT_SKIP_FIRST_ARG 0x8
#define ARGV_NO_PRINT 0x10

#if defined(__cplusplus)
extern "C" {
#endif

static int ParseArgv(int *argcPtr, char **argv, ArgvInfo *argTable, int flags)
{
  (void) argcPtr;
  (void) argv;
  (void) argTable;
  (void) flags;
  fprintf(stderr, "torch_n3 legacy shim: ParseArgv is not available\n");
  abort();
  return 1;
}

#if defined(__cplusplus)
}
#endif

#endif /* N3_COMPAT_PARSEARGV_H */
