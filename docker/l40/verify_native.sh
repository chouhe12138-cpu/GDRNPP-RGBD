#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/gdrnpp

python -c "import detectron2._C, core.csrc.fps._ext, core.csrc.flow.flow_cuda, core.csrc.ransac_voting.ransac_voting, core.csrc.torch_nndistance.torch_nndistance_aten, core.csrc.uncertainty_pnp._ext, lib.egl_renderer.CppEGLRenderer, bop_renderer; print('native imports PASS')"

while IFS= read -r -d '' so_file; do
    if ldd "${so_file}" | grep -q 'not found'; then
        echo "FAIL: unresolved dependency in ${so_file}" >&2
        ldd "${so_file}" >&2
        exit 1
    fi
done < <(find /workspace/gdrnpp /opt/bop_renderer/build -type f -name '*.so' -print0)

bop_so="$(python -c 'import bop_renderer; print(bop_renderer.__file__)')"
ldd "${bop_so}" | grep -Eq 'libOSMesa\.so\.8 => /'
find /workspace/gdrnpp/core/csrc -type f -name '*.so' -print0 | xargs -0 -r cuobjdump --list-elf 2>/dev/null | grep -Fq 'sm_89'
echo "native ldd=PASS libOSMesa.so.8=PASS sm_89=PASS"
