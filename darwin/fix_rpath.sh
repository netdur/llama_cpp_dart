#!/bin/bash

# Set the directory where your dylib files are located.
if [ $# -lt 1 ]; then
    echo "Usage: $0 <platform> (MAC_ARM64, OS64, SIMULATOR64, or SIMULATORARM64)"
    exit 1
else
    PLATFORM="$1"
fi

LIB_DIR="../bin/${PLATFORM}"

fix_rpaths() {
  local lib_file="$1"

  if [ ! -f "$lib_file" ]; then
    echo "Error: Library file not found: $lib_file"
    exit 1
  fi

  echo "Fixing rpaths for: $lib_file"

  # Get the absolute path for better rpath handling
  ABSOLUTE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/${LIB_DIR}" && pwd)"
  
  # First clear existing rpaths to avoid duplicates
  existing_rpaths=$(otool -l "$lib_file" | grep -A2 LC_RPATH | grep path | awk '{print $2}')
  for rpath in $existing_rpaths; do
    echo "Removing existing rpath: $rpath"
    install_name_tool -delete_rpath "$rpath" "$lib_file" || true
  done

  # Add standard framework paths
  install_name_tool -add_rpath "@executable_path/Frameworks" "$lib_file"
  install_name_tool -add_rpath "@loader_path/Frameworks" "$lib_file"
  
  # Now add the absolute path
  install_name_tool -add_rpath "$ABSOLUTE_LIB_DIR" "$lib_file"
  if [ $? -ne 0 ]; then
      echo "Error: install_name_tool -add_rpath failed for $lib_file"
      exit 1
  fi

  # Fix dependencies
  dependencies=$(otool -L "$lib_file" | grep @rpath | awk '{print $1}')
  for dep in $dependencies; do
      dep_name=$(echo "$dep" | awk -F/ '{print $NF}' | awk '{print $1}')
      install_name_tool -change "$dep" "@rpath/$dep_name" "$lib_file"
      if [ $? -ne 0 ]; then
          echo "Error: install_name_tool -change failed for $dep in $lib_file"
          exit 1
      fi
  done

  echo "Verifying changes for $lib_file:"
  otool -L "$lib_file"
  echo "RPaths:"
  otool -l "$lib_file" | grep -A2 LC_RPATH
  echo "-------------------------"
}

if [ ! -d "$LIB_DIR" ]; then
  echo "Error: Library directory '$LIB_DIR' not found. Run this script from the project root."
  exit 1
fi

# Main libraries
fix_rpaths "$LIB_DIR/libllama.dylib"
fix_rpaths "$LIB_DIR/libggml.dylib"

# Component libraries
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

echo "Rpath fixing and re-signing complete for ${PLATFORM}."

exit 0