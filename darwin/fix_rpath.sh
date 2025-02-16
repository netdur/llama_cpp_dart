#!/bin/bash

# Set the directory where your dylib files are located.
LIB_DIR="../bin/MAC_ARM64"

fix_rpaths() {
  local lib_file="$1"

  if [ ! -f "$lib_file" ]; then
    echo "Error: Library file not found: $lib_file"
    exit 1
  fi

  echo "Fixing rpaths for: $lib_file"

  dependencies=$(otool -L "$lib_file" | grep @rpath | awk '{print $1}')

  for dep in $dependencies; do
      dep_name=$(echo "$dep" | awk -F/ '{print $NF}' | awk '{print $1}')
      install_name_tool -change "$dep" "@rpath/$dep_name" "$lib_file"
      if [ $? -ne 0 ]; then
          echo "Error: install_name_tool -change failed for $dep in $lib_file"
          exit 1
      fi
  done

  install_name_tool -add_rpath "$PWD/$LIB_DIR" "$lib_file"
    if [ $? -ne 0 ]; then
        echo "Error: install_name_tool -add_rpath failed for $lib_file"
        exit 1
    fi

  echo "Verifying changes for $lib_file:"
  otool -L "$lib_file"
  echo "-------------------------"
}

if [ ! -d "$LIB_DIR" ]; then
  echo "Error: Library directory '$LIB_DIR' not found.  Run this script from the project root."
  exit 1
fi

fix_rpaths "$LIB_DIR/libllama.dylib"

fix_rpaths "$LIB_DIR/libggml.dylib"

for lib in "$LIB_DIR/libggml-"*.dylib; do
    fix_rpaths "$lib"
done



echo "Re-signing libraries..."
codesign -s - -f "$LIB_DIR"/*.dylib
if [ $? -ne 0 ]; then
    echo "Error: codesign failed"
    exit 1
fi

chmod +x "$LIB_DIR"/*.dylib

echo "Rpath fixing and re-signing complete."

exit 0