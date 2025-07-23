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

build_for_platform() {
    local platform="$1"
    local build_dir="build_${platform}"
    local output_dir="${output_base_dir}/bin/${platform}"
    local shared_libs="ON"
    local lib_extension="dylib"
    local deployment_target=13.1

    echo "Building for platform: ${platform} with deployment target iOS ${deployment_target}"
    
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"

    cmake -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
          -DBUILD_SHARED_LIBS=${shared_libs} \
          -DLLAMA_CURL=OFF \
          -DLLAMA_BUILD_TESTS=OFF \
          -DLLAMA_BUILD_EXAMPLES=OFF \
          -DLLAMA_BUILD_SERVER=OFF \
          -DLLAMA_BUILD_TOOLS=ON  \
          -DCMAKE_PROJECT_INCLUDE="${script_dir}/no_bundle.cmake" \
          -DCMAKE_BUILD_TYPE=Release \
          -G Xcode \
          -DCMAKE_TOOLCHAIN_FILE="${script_dir}/ios-arm64.toolchain.cmake" \
          -DPLATFORM="${platform}" \
          -DDEPLOYMENT_TARGET=${deployment_target} \
          -DENABLE_BITCODE=0 \
          -DENABLE_ARC=0 \
          -DENABLE_VISIBILITY=1 \
          -DENABLE_STRICT_TRY_COMPILE=1 \
          -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM="${dev_team}" \
          -DCMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE=NO \
          -DCMAKE_INSTALL_RPATH="@loader_path/Frameworks" \
          -DCMAKE_INSTALL_PREFIX="./install" \
          ..

    cmake --build . --config Release --parallel
    cmake --install . --config Release

    # Copy libraries WITHOUT post-processing that strips iOS info
    mkdir -p "${output_dir}"
    local libs=(
        "install/lib/libllama.${lib_extension}"
        "install/lib/libggml.${lib_extension}"
        "install/lib/libggml-base.${lib_extension}"
        "install/lib/libggml-metal.${lib_extension}"
        "install/lib/libggml-cpu.${lib_extension}"
        "install/lib/libggml-blas.${lib_extension}"
        "install/lib/libmtmd.${lib_extension}"
    )

    for lib in "${libs[@]}"; do
        if [ -f "$lib" ]; then
            # Copy library directly without rpath modification
            cp "$lib" "${output_dir}/"
            local output_lib="${output_dir}/$(basename "$lib")"
            
            # Only set the install name, don't modify rpaths
            install_name_tool -id "@rpath/$(basename "$lib")" "$output_lib"
            
            # Minimal signing to preserve iOS load commands
            codesign --remove-signature "$output_lib" 2>/dev/null || true
            codesign --force --sign - "$output_lib" 2>/dev/null || true
            
            # Verify iOS version info is preserved
            if otool -l "$output_lib" | grep -q "LC_VERSION_MIN_IPHONEOS\|LC_BUILD_VERSION"; then
                echo "✅ $(basename "$lib") - iOS version info preserved"
            else
                echo "❌ $(basename "$lib") - iOS version info lost"
            fi
        else
            echo "❌ $lib not found"
        fi
    done

    cd ..
}

main() {
    local platform=${3:-"OS64"}
    
    cp "${script_dir}/ios-arm64.toolchain.cmake" "${llama_cpp_path}/"

    pushd "${llama_cpp_path}" > /dev/null

    build_for_platform "${platform}"

    popd > /dev/null
    
    echo "Build completed successfully for ${platform}."
}

main "$@"