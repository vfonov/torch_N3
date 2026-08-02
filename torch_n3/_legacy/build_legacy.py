"""Build the CFFI extension that exposes the original N3 C++ code to Python.

Run directly to (re)build in place::

    python3 torch_n3/_legacy/build_legacy.py

The extension compiles the sources vendored under ``n3/`` and ``ebtks/`` --
byte-for-byte copies of the files from ``legacy/N3/src`` and ``legacy/EBTKS``
that are actually needed, see the ``README.md`` in each -- and by default
links no LAPACK/BLAS of its own at all: it shares PyTorch's, the same way a
PyTorch C++ extension would (see the ``LAPACK_LIBS`` comment below for why).
No MINC toolkit, no libminc2, no libEBTKS, and neither original checkout is
required.

``compat/`` supplies the three MINC headers the legacy sources include
(``volume_io.h``, ``time_stamp.h``, ``ParseArgv.h``) so that they can stay
unmodified without libminc2 behind them.

The LAPACK is the one real choice here: EBTKS bundles its own f2c'd copy and
this used to link it.  The swap is measured in the top-level ``README.md``;
it moves the end-to-end result by more than the blocks do.
"""

import os
import shutil
import sys

import cffi

HERE = os.path.dirname(os.path.abspath(__file__))
N3_SRC = os.path.join(HERE, "n3")
BUILD_DIR = os.path.join(HERE, "build")

# N3 and EBTKS both generate config.h with autoconf/cmake.  Between them the
# vendored sources read only these, so one is generated here rather than by
# either legacy build.  Everything named here is unconditional on Linux.
#
# <malloc.h> is the one exception: it exists on Linux but not on macOS (whose
# malloc/free/realloc come from <stdlib.h>, already included everywhere that
# needs them), and every use in the vendored sources is already guarded by
# `#ifdef HAVE_MALLOC_H` for exactly this reason -- so leave it unset there.
CONFIG_H = """\
#ifndef N3_SHIM_CONFIG_H
#define N3_SHIM_CONFIG_H
#define HAVE_ISFINITE 1
#define HAVE_FLOAT_H 1
#define HAVE_MKSTEMP 1
#define HAVE_DIRENT_H 1
#define HAVE_FCNTL_H 1
%(HAVE_MALLOC_H)s
#define HAVE_MEMORY_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_STRINGS_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_WAIT_H 1
#define HAVE_UNISTD_H 1
#endif
""" % {"HAVE_MALLOC_H": "" if sys.platform == "darwin" else "#define HAVE_MALLOC_H 1"}

# Likewise version.h, normally produced from legacy/N3/include/version.h.in.
# Only sharpen_hist's argument parser uses it, and only to print a banner.
VERSION_H = """\
#ifndef N3_SHIM_VERSION_H
#define N3_SHIM_VERSION_H
#define MNI_VERSION "1.12.00"
#define MNI_LONG_VERSION "Package MNI N3, version " MNI_VERSION \\
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
"""

EBTKS = os.path.join(HERE, "ebtks")

# EBTKS's own build compiles these plus a bundled f2c'd LAPACK; this build
# links the system LAPACK/BLAS instead, so `clapack/` is not vendored.  Pruned
# to what the extension actually requires -- see ebtks/README.md.
EBTKS_SOURCES = [
    os.path.join(EBTKS, "src", "FileIO.cc"),
    os.path.join(EBTKS, "src", "MString.cc"),
    os.path.join(EBTKS, "src", "OrderedCltn.cc"),
    os.path.join(EBTKS, "src", "Path.cc"),
    os.path.join(EBTKS, "templates", "Matrix.cc"),
    os.path.join(EBTKS, "templates", "MatrixSupport.cc"),
    os.path.join(EBTKS, "templates", "ValueMap.cc"),
]

LEGACY_SOURCES = [
    os.path.join(N3_SRC, "Splines", "Spline.cc"),
    os.path.join(N3_SRC, "Splines", "TBSpline.cc"),
    os.path.join(N3_SRC, "VolumeHist", "DHistogram.cc"),
    os.path.join(N3_SRC, "VolumeHist", "WHistogram.cc"),
    os.path.join(N3_SRC, "SharpenHist", "sharpen_hist.cc"),
    # sharpen_hist.cc's (unused) main() references args::verbose etc., which
    # live here.  Pulled in only to satisfy the linker.
    os.path.join(N3_SRC, "SharpenHist", "args.cc"),
    os.path.join(N3_SRC, "MincProg", "print_version.c"),
]

SHIM_SOURCES = [
    os.path.join(HERE, "n3_shim.cc"),
    os.path.join(HERE, "n3_field_shim.cc"),
]

#: Which LAPACK/BLAS to link, and where to find it.  Overridable, because
#: which one it is measurably changes the answer -- see ``tests/convergence.py``
#: and the LAPACK section of ``README.md``.
#:
#:   N3_LAPACK_LIBS="mkl_rt"                   # names, space separated
#:   N3_LAPACK_LIB_DIRS="/opt/intel/oneapi/mkl/latest/lib"
#:
#: To link EBTKS's own bundled f2c'd LAPACK instead of a system one, point
#: these at the EBTKS archive: N3_LAPACK_LIBS="EBTKS".  It resolves after the
#: shim's own objects, so only the clapack members are taken from it.
#:
#: Left unset, the default is to link *no* LAPACK/BLAS at all and leave
#: n3_shim's calls to it (dgemm_, dsysv_, ...) as undefined symbols, resolved
#: at import time against whatever is already loaded in the process -- which
#: this package's own entry point guarantees is PyTorch's.  ``torch/__init__.py``
#: dlopens its own dependency library with ``RTLD_GLOBAL`` for exactly this
#: (see its "Note [Global dependencies]"), specifically so that other native
#: extensions can share its MKL/Accelerate/OpenBLAS instead of linking a
#: second copy.  That second copy is a real failure mode, not a hypothetical
#: one: on macOS, a conda or Homebrew environment's own liblapack.dylib is
#: usually a symlink to an OpenBLAS build with its own bundled libomp.dylib,
#: and PyTorch's wheel bundles a *different* libomp.dylib -- two copies of
#: LLVM's OpenMP runtime loaded into one process is what aborts (or, forced
#: past that with ``KMP_DUPLICATE_LIB_OK``, segfaults instead). Riding on
#: PyTorch's own BLAS sidesteps that entirely, on any platform, without this
#: file having to know what PyTorch chose.
#:
#: This relies on two things holding, both already true of every documented
#: entry point (``torch_n3.backends.legacy`` imports ``torch`` before the
#: shim, and so does every test): ``torch`` (or anything else providing the
#: same symbols, e.g. ``numpy``) must already be imported by the time a
#: LAPACK-touching call happens, not merely before this extension is built;
#: and the platform's default linker has to permit undefined symbols in a
#: shared object, which is the ELF default and is macOS's default for Python
#: extensions specifically (``-undefined dynamic_lookup``, already in
#: ``sysconfig``'s ``LDSHARED``) -- ``-Wl,--allow-shlib-undefined`` below is
#: a no-op where that already holds and a safety net where a hardened
#: toolchain's default ``LDFLAGS`` narrowed it.
_LAPACK_LIBS_ENV = os.environ.get("N3_LAPACK_LIBS")
if _LAPACK_LIBS_ENV is not None:
    LAPACK_LIBS = _LAPACK_LIBS_ENV.split()
    LAPACK_EXTRA_LINK_ARGS = []
elif sys.platform == "darwin":
    LAPACK_LIBS = []
    LAPACK_EXTRA_LINK_ARGS = []
else:
    LAPACK_LIBS = []
    LAPACK_EXTRA_LINK_ARGS = ["-Wl,--allow-shlib-undefined"]
LAPACK_LIB_DIRS = [d for d in os.environ.get("N3_LAPACK_LIB_DIRS", "").split(os.pathsep)
                   if d]


def build(verbose=True):
    # Everything the build generates -- headers, object files, the extension
    # itself -- goes under BUILD_DIR, which is gitignored.  Only the finished
    # .so is copied next to this script, where Python can import it.
    generated = os.path.join(BUILD_DIR, "generated")
    os.makedirs(generated, exist_ok=True)
    with open(os.path.join(generated, "config.h"), "w") as fp:
        fp.write(CONFIG_H)
    with open(os.path.join(generated, "version.h"), "w") as fp:
        fp.write(VERSION_H)

    with open(os.path.join(HERE, "n3_shim.h")) as fp:
        declarations = fp.read()

    ffibuilder = cffi.FFI()
    ffibuilder.cdef(declarations)
    ffibuilder.set_source(
        "torch_n3._legacy._n3legacy",
        # The shim is compiled as C++, so its declarations need C linkage to
        # match the definitions in n3_shim.cc.
        'extern "C" {\n#include "n3_shim.h"\n}\n',
        source_extension=".cc",
        sources=SHIM_SOURCES + LEGACY_SOURCES + EBTKS_SOURCES,
        include_dirs=[
            HERE,
            generated,
            # Ahead of everything else: these shadow <volume_io.h>,
            # <time_stamp.h> and <ParseArgv.h> so no MINC library is needed.
            os.path.join(HERE, "compat"),
            os.path.join(EBTKS, "include"),
            os.path.join(EBTKS, "templates"),
            os.path.join(N3_SRC, "Splines"),
            os.path.join(N3_SRC, "VolumeHist"),
            os.path.join(N3_SRC, "SharpenHist"),
            os.path.join(N3_SRC, "CorrectField"),
        ],
        library_dirs=LAPACK_LIB_DIRS,
        libraries=LAPACK_LIBS,
        extra_link_args=LAPACK_EXTRA_LINK_ARGS,
        define_macros=[
            ("HAVE_CONFIG_H", "1"),
            ("USE_COMPMAT", "1"),
            ("USE_DBLMAT", "1"),
            ("USE_FCOMPMAT", "1"),
            # sharpen_hist.cc is a program, and only its helper functions are
            # wanted here, so its entry point is renamed out of the way.
            ("main", "n3_sharpen_hist_unused_main"),
        ],
        extra_compile_args=["-O2", "-w"],
    )
    # cffi lays the extension out under tmpdir following the module's package
    # path, and scatters object files alongside it; keeping tmpdir inside
    # BUILD_DIR confines all of that to one throwaway directory.
    built = ffibuilder.compile(tmpdir=BUILD_DIR, verbose=verbose)
    installed = os.path.join(HERE, os.path.basename(built))
    shutil.copy2(built, installed)
    return installed


if __name__ == "__main__":
    build(verbose="-q" not in sys.argv)
