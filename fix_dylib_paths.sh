#!/bin/bash

# fix_dylib_paths.sh

# Fix libggml-base.dylib
install_name_tool -id "@loader_path/libggml-base.dylib" libggml-base.dylib

# Fix libggml-cpu.dylib
install_name_tool -id "@loader_path/libggml-cpu.dylib" libggml-cpu.dylib
install_name_tool -change "@rpath/libggml-base.dylib" "@loader_path/libggml-base.dylib" libggml-cpu.dylib

# Fix libggml-blas.dylib
install_name_tool -id "@loader_path/libggml-blas.dylib" libggml-blas.dylib
install_name_tool -change "@rpath/libggml-base.dylib" "@loader_path/libggml-base.dylib" libggml-blas.dylib

# Fix libggml-metal.dylib
install_name_tool -id "@loader_path/libggml-metal.dylib" libggml-metal.dylib
install_name_tool -change "@rpath/libggml-base.dylib" "@loader_path/libggml-base.dylib" libggml-metal.dylib

# Fix libggml.dylib
install_name_tool -id "@loader_path/libggml.dylib" libggml.dylib
install_name_tool -change "@rpath/libggml-cpu.dylib" "@loader_path/libggml-cpu.dylib" libggml.dylib
install_name_tool -change "@rpath/libggml-blas.dylib" "@loader_path/libggml-blas.dylib" libggml.dylib
install_name_tool -change "@rpath/libggml-metal.dylib" "@loader_path/libggml-metal.dylib" libggml.dylib
install_name_tool -change "@rpath/libggml-base.dylib" "@loader_path/libggml-base.dylib" libggml.dylib

# Fix libllama.dylib
install_name_tool -id "@loader_path/libllama.dylib" libllama.dylib
install_name_tool -change "@rpath/libggml.dylib" "@loader_path/libggml.dylib" libllama.dylib
install_name_tool -change "@rpath/libggml-cpu.dylib" "@loader_path/libggml-cpu.dylib" libllama.dylib
install_name_tool -change "@rpath/libggml-blas.dylib" "@loader_path/libggml-blas.dylib" libllama.dylib
install_name_tool -change "@rpath/libggml-metal.dylib" "@loader_path/libggml-metal.dylib" libllama.dylib
install_name_tool -change "@rpath/libggml-base.dylib" "@loader_path/libggml-base.dylib" libllama.dylib

# Verify changes
echo "Verifying changes..."
for lib in libllama.dylib libggml*.dylib; do
    echo "=== $lib ==="
    otool -L "$lib"
    echo
done