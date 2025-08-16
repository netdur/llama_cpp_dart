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

  # Check if this is an iOS library (has iOS version info)
  local is_ios_lib=$(otool -l "$lib_file" | grep -c "LC_VERSION_MIN_IPHONEOS\|LC_BUILD_VERSION" || true)
  
  if [ "$is_ios_lib" -gt 0 ]; then
    echo "📱 iOS library detected - using iOS-safe rpath fixing"
    
    # For iOS libraries, be more conservative with rpath changes
    # Only remove problematic rpaths, not all of them
    existing_rpaths=$(otool -l "$lib_file" | grep -A2 LC_RPATH | grep path | awk '{print $2}')
    for rpath in $existing_rpaths; do
      # Only remove absolute paths and problematic rpaths
      if [[ "$rpath" == /* ]] || [[ "$rpath" == "@executable_path/Frameworks" ]]; then
        echo "Removing problematic rpath: $rpath"
        install_name_tool -delete_rpath "$rpath" "$lib_file" 2>/dev/null || true
      fi
    done

    # Add only iOS-appropriate rpaths
    # install_name_tool -add_rpath "@loader_path/Frameworks" "$lib_file" 2>/dev/null || true
    install_name_tool -add_rpath "@loader_path" "$lib_file" 2>/dev/null || true
    
  else
    echo "🖥️ macOS library detected - using standard rpath fixing"
    
    # For macOS libraries, use the original logic
    existing_rpaths=$(otool -l "$lib_file" | grep -A2 LC_RPATH | grep path | awk '{print $2}')
    for rpath in $existing_rpaths; do
      echo "Removing existing rpath: $rpath"
      install_name_tool -delete_rpath "$rpath" "$lib_file" 2>/dev/null || true
    done

    # Add standard framework paths for macOS
    install_name_tool -add_rpath "@executable_path/Frameworks" "$lib_file" 2>/dev/null || true
    install_name_tool -add_rpath "@loader_path/Frameworks" "$lib_file" 2>/dev/null || true
    
    # Add absolute path only for macOS
    ABSOLUTE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/${LIB_DIR}" && pwd)"
    install_name_tool -add_rpath "$ABSOLUTE_LIB_DIR" "$lib_file" 2>/dev/null || true
  fi

  # Fix dependencies (same for both platforms)
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
  
  # Check if iOS version info is still there
  local ios_info_after=$(otool -l "$lib_file" | grep -c "LC_VERSION_MIN_IPHONEOS\|LC_BUILD_VERSION" || true)
  if [ "$is_ios_lib" -gt 0 ] && [ "$ios_info_after" -eq 0 ]; then
    echo "⚠️  WARNING: iOS version info was lost!"
  elif [ "$is_ios_lib" -gt 0 ] && [ "$ios_info_after" -gt 0 ]; then
    echo "✅ iOS version info preserved"
  fi
  echo "-------------------------"
}

safe_codesign() {
  local lib_file="$1"
  
  # Check if this is an iOS library
  local is_ios_lib=$(otool -l "$lib_file" | grep -c "LC_VERSION_MIN_IPHONEOS\|LC_BUILD_VERSION" || true)
  
  if [ "$is_ios_lib" -gt 0 ]; then
    echo "📱 iOS-safe signing: $(basename "$lib_file")"
    # Use minimal signing options for iOS libraries
    codesign --remove-signature "$lib_file" 2>/dev/null || true
    codesign --force --sign - --preserve-metadata=identifier,entitlements,flags "$lib_file" 2>/dev/null || true
  else
    echo "🖥️ Standard signing: $(basename "$lib_file")"
    # Use standard signing for macOS
    codesign -s - -f "$lib_file" 2>/dev/null || true
  fi
  
  # Verify iOS info is still there after signing
  local ios_info_after=$(otool -l "$lib_file" | grep -c "LC_VERSION_MIN_IPHONEOS\|LC_BUILD_VERSION" || true)
  if [ "$is_ios_lib" -gt 0 ] && [ "$ios_info_after" -eq 0 ]; then
    echo "❌ iOS version info lost during signing!"
  elif [ "$is_ios_lib" -gt 0 ] && [ "$ios_info_after" -gt 0 ]; then
    echo "✅ iOS version info preserved after signing"
  fi
}

if [ ! -d "$LIB_DIR" ]; then
  echo "Error: Library directory '$LIB_DIR' not found. Run this script from the project root."
  exit 1
fi

echo "Fixing rpaths for ${PLATFORM}..."

# Main libraries
fix_rpaths "$LIB_DIR/libllama.dylib"
fix_rpaths "$LIB_DIR/libggml.dylib"
fix_rpaths "$LIB_DIR/libmtmd.dylib"

# Component libraries
for lib in "$LIB_DIR/libggml-"*.dylib; do
    if [ -f "$lib" ]; then
        fix_rpaths "$lib"
    fi
done

echo "Re-signing libraries..."
for lib in "$LIB_DIR"/*.dylib; do
    if [ -f "$lib" ]; then
        safe_codesign "$lib"
    fi
done

chmod +x "$LIB_DIR"/*.dylib

echo "Rpath fixing and re-signing complete for ${PLATFORM}."

exit 0


# for f in bin/MAC_ARM64/*.dylib; do install_name_tool -delete_rpath @loader_path "$f" 2>/dev/null; install_name_tool -add_rpath @loader_path "$f"; done
# codesign --force --sign - bin/MAC_ARM64/*.dylib