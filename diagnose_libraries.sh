#!/bin/bash

# Script to diagnose differences between working and non-working libraries
# Usage: ./diagnose_libraries.sh

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Define directories
WORKING_DIR="bin/MAC_ARM64.old"
NEW_DIR="bin/MAC_ARM64"

if [ ! -d "$WORKING_DIR" ] || [ ! -d "$NEW_DIR" ]; then
    echo -e "${RED}Error: Both $WORKING_DIR and $NEW_DIR must exist${NC}"
    exit 1
fi

echo -e "${GREEN}Comparing library files between working ($WORKING_DIR) and new ($NEW_DIR) directories${NC}"
echo

# Function to examine a specific library
examine_library() {
    local lib_name=$1
    local working_lib="$WORKING_DIR/$lib_name"
    local new_lib="$NEW_DIR/$lib_name"
    
    if [ ! -f "$working_lib" ] || [ ! -f "$new_lib" ]; then
        echo -e "${RED}Error: $lib_name not found in both directories${NC}"
        return
    fi
    
    echo -e "${BLUE}=== Examining $lib_name ===${NC}"
    
    # Compare file sizes
    local working_size=$(stat -f "%z" "$working_lib")
    local new_size=$(stat -f "%z" "$new_lib")
    
    echo -e "${YELLOW}File sizes:${NC}"
    echo "Working: $working_size bytes"
    echo "New: $new_size bytes"
    echo
    
    # Compare otool -L output (dependencies)
    echo -e "${YELLOW}Dependencies (otool -L):${NC}"
    echo -e "${YELLOW}Working:${NC}"
    otool -L "$working_lib" | sed 's/^/    /'
    echo -e "${YELLOW}New:${NC}"
    otool -L "$new_lib" | sed 's/^/    /'
    echo
    
    # Compare rpaths
    echo -e "${YELLOW}RPaths (otool -l | grep LC_RPATH -A2):${NC}"
    echo -e "${YELLOW}Working:${NC}"
    otool -l "$working_lib" | grep -A2 LC_RPATH | sed 's/^/    /'
    echo -e "${YELLOW}New:${NC}"
    otool -l "$new_lib" | grep -A2 LC_RPATH | sed 's/^/    /'
    echo
    
    # Check codesign
    echo -e "${YELLOW}Code Signature:${NC}"
    echo -e "${YELLOW}Working:${NC}"
    codesign -vv "$working_lib" 2>&1 | sed 's/^/    /'
    echo -e "${YELLOW}New:${NC}"
    codesign -vv "$new_lib" 2>&1 | sed 's/^/    /'
    echo
    
    # Check install names
    echo -e "${YELLOW}Install Name (otool -D):${NC}"
    echo -e "${YELLOW}Working:${NC}"
    otool -D "$working_lib" | sed 's/^/    /'
    echo -e "${YELLOW}New:${NC}"
    otool -D "$new_lib" | sed 's/^/    /'
    echo
    
    # Check compatibility version
    echo -e "${YELLOW}Compatibility Version:${NC}"
    echo -e "${YELLOW}Working:${NC}"
    otool -l "$working_lib" | grep -A4 LC_ID_DYLIB | grep compat | sed 's/^/    /'
    echo -e "${YELLOW}New:${NC}"
    otool -l "$new_lib" | grep -A4 LC_ID_DYLIB | grep compat | sed 's/^/    /'
    echo
    
    # Basic information about the binary
    echo -e "${YELLOW}File Info:${NC}"
    echo -e "${YELLOW}Working:${NC}"
    file "$working_lib" | sed 's/^/    /'
    echo -e "${YELLOW}New:${NC}"
    file "$new_lib" | sed 's/^/    /'
    echo
    
    echo -e "${BLUE}=== End of $lib_name examination ===${NC}"
    echo "-------------------------------------------------------------"
    echo
}

# Examine main libraries
examine_library "libllama.dylib"
examine_library "libggml.dylib"
examine_library "libggml-metal.dylib"

echo -e "${GREEN}Diagnosis complete.${NC}"
echo
echo "If you suspect a specific issue with the fix_rpath.sh script, try running:"
echo "otool -l bin/MAC_ARM64/libllama.dylib | grep -A2 LC_RPATH"
echo
echo "Also check if your dart example needs additional environment variables:"
echo "DYLD_PRINT_LIBRARIES=1 DYLD_PRINT_RPATHS=1 dart example/simple.dart"
echo
echo "This will show library loading details and might reveal the issue."