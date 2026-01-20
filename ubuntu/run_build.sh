#!/usr/bin/env bash
# bash ubuntu/run_build.sh  --cmake-arg -DGGML_CUDA_FORCE_MMQ=ON

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ubuntu/run_build.sh [path-to-llama.cpp] [options] [-- <extra cmake args>]

Builds llama.cpp on Ubuntu with CUDA enabled and copies the shared libraries
into bin/<output> inside this repository.

Options:
  -t, --build-type <type>     CMake build type (default: Release)
  -b, --build-dir <name>      Build directory inside llama.cpp (default: build-linux-cuda)
  -o, --output <name>         Subdirectory inside bin/ for the .so files (default: linux-cuda)
  -a, --cuda-arch <list>      Value for CMAKE_CUDA_ARCHITECTURES (default: native)
  -g, --generator <name>      CMake generator (default: Ninja if available)
  -c, --clean                 Remove the build directory before configuring
      --cmake-arg <arg>       Extra option passed to CMake (repeatable)
  -h, --help                  Show this message

Examples:
  ubuntu/run_build.sh src/llama.cpp
  ubuntu/run_build.sh src/llama.cpp --cuda-arch "86;89" --cmake-arg -DGGML_CUDA_FORCE_MMQ=ON
EOF
}

require_value() {
    if [[ -z "${2:-}" ]]; then
        echo "Error: option '$1' requires a value" >&2
        usage
        exit 1
    fi
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_base_dir="${script_dir}/.."

build_type="Release"
build_dir_name="build-linux-cuda"
output_name="linux-cuda"
cuda_archs="native"
generator=""
clean_build=false
default_llama_cpp_path="${script_dir}/../src/llama.cpp"
llama_cpp_path=""
declare -a cmake_extra_args
cmake_extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--build-type)
            require_value "$1" "${2:-}"
            build_type="$2"
            shift 2
            ;;
        -b|--build-dir)
            require_value "$1" "${2:-}"
            build_dir_name="$2"
            shift 2
            ;;
        -o|--output)
            require_value "$1" "${2:-}"
            output_name="$2"
            shift 2
            ;;
        -a|--cuda-arch)
            require_value "$1" "${2:-}"
            cuda_archs="$2"
            shift 2
            ;;
        -g|--generator)
            require_value "$1" "${2:-}"
            generator="$2"
            shift 2
            ;;
        -c|--clean)
            clean_build=true
            shift
            ;;
        --cmake-arg)
            require_value "$1" "${2:-}"
            cmake_extra_args+=("$2")
            shift 2
            ;;
        --cmake-arg=*)
            cmake_extra_args+=("${1#*=}")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                cmake_extra_args+=("$1")
                shift
            done
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -z "$llama_cpp_path" ]]; then
                llama_cpp_path="$1"
            else
                cmake_extra_args+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "$llama_cpp_path" ]]; then
    llama_cpp_path="$default_llama_cpp_path"
fi

if [[ -z "$generator" ]] && command -v ninja >/dev/null 2>&1; then
    generator="Ninja"
fi

if [[ ! -d "$llama_cpp_path" ]]; then
    echo "Error: '${llama_cpp_path}' does not exist or is not a directory." >&2
    exit 1
fi

llama_cpp_path="$(cd "$llama_cpp_path" && pwd)"
build_dir="${llama_cpp_path}/${build_dir_name}"
install_dir="${build_dir}/install"
output_dir="${output_base_dir}/bin/${output_name}"

if $clean_build && [[ -d "$build_dir" ]]; then
    echo "Removing existing build directory: ${build_dir}"
    rm -rf "$build_dir"
fi

pushd "$llama_cpp_path" >/dev/null

cmake_args=(
    -S .
    -B "$build_dir_name"
    "-DCMAKE_BUILD_TYPE=${build_type}"
    "-DCMAKE_INSTALL_RPATH=\$ORIGIN"
    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
    "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON"
    "-DBUILD_SHARED_LIBS=ON"
    "-DLLAMA_CURL=OFF"
    "-DLLAMA_BUILD_TESTS=OFF"
    "-DLLAMA_BUILD_EXAMPLES=OFF"
    "-DLLAMA_BUILD_SERVER=ON"
    "-DLLAMA_BUILD_COMMON=ON"
    "-DLLAMA_BUILD_TOOLS=ON"
    "-DLLAMA_FATAL_WARNINGS=OFF"
    "-DGGML_CUDA=ON"
    "-DCMAKE_INSTALL_PREFIX=${install_dir}"
)

if [[ -n "$cuda_archs" ]]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${cuda_archs}")
fi

if [[ -n "$generator" ]]; then
    cmake_args+=(-G "$generator")
fi

if [[ ${#cmake_extra_args[@]} -gt 0 ]]; then
    cmake_args+=("${cmake_extra_args[@]}")
fi

echo "Configuring llama.cpp with CUDA support..."
cmake "${cmake_args[@]}"

echo "Building llama.cpp (${build_type})..."
cmake --build "$build_dir_name" --config "$build_type" --parallel

echo "Installing artifacts..."
cmake --install "$build_dir_name" --config "$build_type"

popd >/dev/null

lib_root="${install_dir}/lib"
if [[ ! -d "$lib_root" && -d "${install_dir}/lib64" ]]; then
    lib_root="${install_dir}/lib64"
fi

if [[ ! -d "$lib_root" ]]; then
    echo "Error: could not find installed libraries in ${install_dir}." >&2
    exit 1
fi

rm -rf "$output_dir"
mkdir -p "$output_dir"

required_libs=(
    "libllama.so"
    "libggml.so"
    "libggml-base.so"
    "libggml-cpu.so"
    "libggml-cuda.so"
    "libmtmd.so"
)

optional_libs=(
    "libggml-blas.so"
)

missing_required=0

copy_lib() {
    local lib_name="$1"
    local required="$2"
    local found=0
    shopt -s nullglob
    local matches=("${lib_root}/${lib_name}"*)
    shopt -u nullglob
    if [[ ${#matches[@]} -eq 0 ]]; then
        if [[ "$required" == "true" ]]; then
            echo "Error: required library '${lib_name}' not found in ${lib_root}" >&2
            missing_required=1
        else
            echo "Warning: optional library '${lib_name}' not found, skipping."
        fi
        return
    fi
    for src in "${matches[@]}"; do
        cp -P "$src" "$output_dir/"
        echo "Copied $(basename "$src")"
        found=1
    done
    if [[ $found -eq 0 && "$required" == "true" ]]; then
        echo "Error: required library '${lib_name}' not found in ${lib_root}" >&2
        missing_required=1
    fi
}

for lib in "${required_libs[@]}"; do
    copy_lib "$lib" "true"
done

for lib in "${optional_libs[@]}"; do
    copy_lib "$lib" "false"
done

if [[ $missing_required -ne 0 ]]; then
    echo "One or more required libraries were not copied. See errors above." >&2
    exit 1
fi

echo "✅ CUDA build complete. Libraries are in ${output_dir}"
