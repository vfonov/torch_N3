"""Build the CFFI extension that exposes the original N3 C++ code to Python.

Run directly to (re)build in place::

    python3 torch_n3/_legacy/build_legacy.py

The extension compiles legacy N3 sources straight out of ``legacy/N3/src`` --
they are never copied or modified -- and links against the EBTKS library and
LAPACK that ship with the installed MINC toolkit.
"""

import os
import shutil
import sys

import cffi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
N3_SRC = os.path.join(ROOT, "legacy", "N3", "src")
BUILD_DIR = os.path.join(HERE, "build")

# Where EBTKS headers/libs live.  MINC_TOOLKIT is set in this environment;
# fall back to the usual install location.
TOOLKIT = os.environ.get("MINC_TOOLKIT", "/opt/minc/1.9.18.13")

# N3 normally generates config.h with autoconf/cmake.  The legacy sources need
# only these two symbols, so we generate a minimal one rather than configuring
# the whole legacy build.
CONFIG_H = """\
#ifndef N3_SHIM_CONFIG_H
#define N3_SHIM_CONFIG_H
#define HAVE_ISFINITE 1
#define HAVE_FLOAT_H 1
#endif
"""

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
        sources=SHIM_SOURCES + LEGACY_SOURCES,
        include_dirs=[
            HERE,
            generated,
            os.path.join(N3_SRC, "Splines"),
            os.path.join(N3_SRC, "VolumeHist"),
            os.path.join(N3_SRC, "SharpenHist"),
            os.path.join(N3_SRC, "CorrectField"),
            os.path.join(TOOLKIT, "include"),
        ],
        library_dirs=[os.path.join(TOOLKIT, "lib")],
        libraries=["EBTKS", "minc2", "lapack", "blas"],
        define_macros=[
            ("HAVE_CONFIG_H", "1"),
            ("USE_COMPMAT", "1"),
            ("USE_DBLMAT", "1"),
            ("USE_FCOMPMAT", "1"),
            # sharpen_hist.cc is a program; we want its helper functions, not
            # its entry point.
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
