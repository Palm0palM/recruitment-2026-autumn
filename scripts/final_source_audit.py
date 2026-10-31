#!/usr/bin/env python3
"""Assert frozen numerical source identity, allowing only reviewed adapters."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def git_text(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def tokens(text):
    pattern = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[A-Za-z_]\w*|\d+|[^\s]'
    return [t for t in re.findall(pattern, text, flags=re.S)
            if not t.startswith(("//", "/*"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    subprocess.run(["git", "diff", "--exit-code", "af56878", "--",
                    "src/main.cpp", "validate.py", "include/melspec.h", "data"], check=True)
    result = {"protected_unchanged": True}

    cpu = Path("src/melspec_cpu.cpp").read_text()
    candidate = git_text("cbf214d", "src/melspec.cpp")
    stripped = cpu.replace('#include "melspec_backend.h"\n', "", 1)
    stripped = stripped.replace("compute_melspectrogram_cpu(", "compute_melspectrogram(", 1)
    assert tokens(stripped) == tokens(candidate), "CPU differs beyond private entry/header/comments"
    result["cpu"] = dict(source="cbf214d (on e448c989)", sha256=digest(cpu),
                         numerical_tokens_identical=True)

    cuda = Path("src/melspec_cuda.cu").read_text()
    frozen = git_text("3e111946", "src/melspec_cuda.cu")
    stripped = cuda.replace('#include "melspec_backend.h"\n', "", 1)
    begin = stripped.index("static void check_cuda_initialization(")
    end = stripped.index("#if MELSPEC_CUDA_OVERLAP_INIT\nstruct ContextInit", begin)
    stripped = stripped[:begin] + stripped[end:]
    stripped = stripped.replace("            check_cuda_initialization(status);", """            if (status != cudaSuccess)
                throw std::runtime_error(std::string("CUDA context initialization: ") +
                                         cudaGetErrorString(status));""", 1)
    stripped = stripped.replace("#else\n    check_cuda_initialization(cudaFree(nullptr));\n", "", 1)
    stripped = stripped[:stripped.index("\nbool cuda_backend_available()")].rstrip() + "\n"
    assert stripped == frozen, "CUDA differs beyond typed initialization / availability adapters"
    result["cuda"] = dict(source="3e111946", sha256=digest(cuda), frozen_sha256=digest(frozen),
                          frozen_pipeline_identical_after_removing_adapters=True)

    hip = Path("src/melspec_hip.cu").read_text()
    frozen = git_text("eff3ed1", "src/melspec_hip.cu")
    # Entire kernel/type/RAII/transfer-helper section, byte-exact.
    def kernels(s):
        a = s.index("namespace {\n")
        b = s.index("#if MELSPEC_HIP_PREWARM\n// The first HIP queue", a)
        return s[a:b]
    assert kernels(hip) == kernels(frozen), "HIP kernels or pipeline helpers changed"
    stripped = hip[hip.index("void compute_melspectrogram_hip("):]
    stripped = stripped[:stripped.index("\nbool hip_backend_available()")].rstrip()+"\n"
    a = stripped.index("#if MELSPEC_HIP_PREWARM && MELSPEC_HIP_EAGER_PREWARM\n")
    b = stripped.index("#endif\n", a) + len("#endif\n")
    stripped = stripped[:a] + stripped[b:]
    stripped = stripped.replace("#if MELSPEC_HIP_PREWARM\n    (void)hip_prewarm();\n#endif\n", "", 1)
    stripped = stripped.replace("#else\n    check_hip_initialization(hipInit(0));\n", "", 1)
    stripped = stripped.replace("hip_prewarm().finish();", "hip_prewarm_instance.finish();", 1)
    # Diagnostic-only variadic argument ordering was wrong in the frozen
    # profiler (async_h2d/staging_rows labels swapped). PROFILE=0 erases it.
    stripped = stripped.replace(
        "MELSPEC_HIP_RESIZE_THREAD, MELSPEC_HIP_ASYNC_H2D,\n                  prewarm_wait_ms, MELSPEC_HIP_STAGING_ROWS,",
        "MELSPEC_HIP_RESIZE_THREAD, MELSPEC_HIP_STAGING_ROWS,\n                  MELSPEC_HIP_ASYNC_H2D, prewarm_wait_ms,", 1)
    frozen_compute = frozen[frozen.index("void compute_melspectrogram_hip("):]
    assert stripped == frozen_compute, "HIP compute differs beyond startup policy"
    result["hip"] = dict(source="eff3ed1", sha256=digest(hip), frozen_sha256=digest(frozen),
                         kernels_sha256=digest(kernels(hip)), numerical_pipeline_identical=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
