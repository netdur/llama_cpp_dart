#!/bin/bash

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <llama_cpp_path> <dev_team>"
    exit 1
fi

llama_cpp_path="$1"
dev_team="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_base_dir="${script_dir}/.."

# Function to build for a specific platform
build_for_platform() {
    local platform="$1"
    local build_dir="build_${platform}"
    local output_dir="${output_base_dir}/bin/${platform}"
    local shared_libs="ON"
    local lib_extension="dylib"

    local postfix=""
    if [[ "$platform" == "SIMULATORARM64" ]]; then
        postfix="-iphonesimulator"
    fi
    if [[ "$platform" == "OS64" ]]; then
        postfix="-iphoneos"
    fi

    echo "Building for platform: ${platform}"
    
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"

    cmake -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
          -DBUILD_SHARED_LIBS=${shared_libs} \
          -DLLAMA_BUILD_TESTS=OFF \
          -DLLAMA_BUILD_EXAMPLES=OFF \
          -DLLAMA_BUILD_SERVER=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          -G Xcode \
          -DCMAKE_TOOLCHAIN_FILE="${script_dir}/ios-arm64.toolchain.cmake" \
          -DPLATFORM="${platform}" \
          -DDEPLOYMENT_TARGET=12 \
          -DENABLE_BITCODE=0 \
          -DENABLE_ARC=0 \
          -DENABLE_VISIBILITY=1 \
          -DENABLE_STRICT_TRY_COMPILE=1 \
          -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM="${dev_team}" \
          -DCMAKE_INSTALL_PREFIX="./install" \
          ..

    cmake --build . --config Release
    cmake --install . --config Release

    # Copy libraries
    mkdir -p "${output_dir}"
    local libs=(
        "install/lib/libllama.${lib_extension}"
        "install/lib/libggml.${lib_extension}"
        "install/lib/libggml-base.${lib_extension}"
        "install/lib/libggml-metal.${lib_extension}"
        "install/lib/libggml-cpu.${lib_extension}"
        "install/lib/libggml-blas.${lib_extension}"
    )

    for lib in "${libs[@]}"; do
        if [ -f "$lib" ]; then
            cp "$lib" "${output_dir}/"
        else
            echo "Warning: $lib not found"
        fi
    done

    cd ..
}

main() {
    cp "${script_dir}/ios-arm64.toolchain.cmake" "${llama_cpp_path}/"

    pushd "${llama_cpp_path}" > /dev/null

    build_for_platform "MAC_ARM64"
    # build_for_platform "OS64"
    # build_for_platform "SIMULATORARM64"

    # return to original directory
    popd > /dev/null

    rm -rf "${output_base_dir}/include"
    mkdir -p "${output_base_dir}/include"
    cp -r "${llama_cpp_path}/build_MAC_ARM64/install/include/"* "${output_base_dir}/include/"
}

main "$@"