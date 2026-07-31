#!/usr/bin/env bash
set -Eeuo pipefail

vendor_dir="${1:-/tmp/vendor}"
project_root="/workspace/gdrnpp"
pnp_root="${project_root}/core/csrc/uncertainty_pnp"

export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS="${MAX_JOBS:-4}"

cd "${vendor_dir}"
sha256sum --check SHA256SUMS

rm -rf /tmp/ceres-src /tmp/ceres-build "${pnp_root}/lib"
mkdir -p /tmp/ceres-src /tmp/ceres-build
tar -xzf "${vendor_dir}/ceres-solver-1.14.0.tar.gz" --strip-components=1 -C /tmp/ceres-src

cmake -S /tmp/ceres-src -B /tmp/ceres-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${pnp_root}" -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DMINIGLOG=OFF -DGFLAGS=OFF -DSUITESPARSE=OFF -DCXSPARSE=OFF -DEIGENSPARSE=ON -DLAPACK=OFF
cmake --build /tmp/ceres-build --parallel "${MAX_JOBS}"
cmake --install /tmp/ceres-build
rm -rf /tmp/ceres-src /tmp/ceres-build

mkdir -p "${pnp_root}/include" "${pnp_root}/lib"
rm -rf "${pnp_root}/include/eigen3"
ln -s /usr/include/eigen3 "${pnp_root}/include/eigen3"
ln -s /usr/lib/x86_64-linux-gnu/libglog.so "${pnp_root}/lib/libglog.so"

build_inplace() {
    local directory="$1"
    shift
    pushd "${directory}" >/dev/null
    rm -rf build
    find . -maxdepth 2 -type f \( -name '*.so' -o -name '*.o' \) -delete
    python setup.py "$@"
    popd >/dev/null
}

build_inplace "${project_root}/core/csrc/fps"
build_inplace "${project_root}/core/csrc/flow" build_ext --inplace
build_inplace "${project_root}/core/csrc/ransac_voting" build_ext --inplace
build_inplace "${project_root}/core/csrc/torch_nndistance" build_ext --inplace
build_inplace "${pnp_root}"

egl_cmake="${project_root}/lib/egl_renderer/CMakeLists.txt"
grep -Fx 'target_link_libraries(CppEGLRenderer PRIVATE pybind11::module dl pthread GL)' "${egl_cmake}"
sed -i 's|^target_link_libraries(CppEGLRenderer PRIVATE pybind11::module dl pthread GL)$|target_link_libraries(CppEGLRenderer PRIVATE pybind11::module dl pthread GL EGL)|' "${egl_cmake}"
grep -Fx 'target_link_libraries(CppEGLRenderer PRIVATE pybind11::module dl pthread GL EGL)' "${egl_cmake}"
build_inplace "${project_root}/lib/egl_renderer" build_ext --inplace

rm -rf /opt/bop_renderer
mkdir -p /opt/bop_renderer
tar -xzf "${vendor_dir}/bop_renderer-8fd19a3463b331d74d3b21ca5c50668127f06041.tar.gz" --strip-components=1 -C /opt/bop_renderer
sed -i 's|^set(CMAKE_MODULE_PATH CMake \$CMAKE_MODULE_PATH)$|list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")|' /opt/bop_renderer/CMakeLists.txt
grep -Fx 'list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")' /opt/bop_renderer/CMakeLists.txt
cmake -S /opt/bop_renderer -B /opt/bop_renderer/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python)"
cmake --build /opt/bop_renderer/build --parallel "${MAX_JOBS}"

rm -rf "${project_root}/core/csrc/flow/build" "${project_root}/core/csrc/ransac_voting/build" "${project_root}/core/csrc/torch_nndistance/build" "${project_root}/lib/egl_renderer/build"
