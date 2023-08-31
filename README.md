# About MLAS
MLAS is a compute library containing processor optimized GEMM kernels and platform specific threading code.

# version
copy from v1.15.1
hash: baeece44ba075009c6bfe95891a8c1b3d4571cb3

Fix linux compiler error:
1. missing memset/memcpy header file <string.h>
2. platform.cpp: syscall missing header file <unistd.h>


Extended functions:
1. add support openmp
2. add samples
