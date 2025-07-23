#!/bin/bash

echo "🔍 Detailed library analysis..."

for lib in bin/OS64/*.dylib; do
    if [ -f "$lib" ]; then
        echo "=== $(basename "$lib") ==="
        
        # Method 1: Count matches (same as rpath script)
        ios_count=$(otool -l "$lib" | grep -c "LC_VERSION_MIN_IPHONEOS\|LC_BUILD_VERSION" 2>/dev/null || echo "0")
        echo "iOS load command count: $ios_count"
        
        # Method 2: Look for actual content
        ios_content=$(otool -l "$lib" | grep -A 5 "LC_VERSION_MIN_IPHONEOS\|LC_BUILD_VERSION" 2>/dev/null || echo "")
        if [ -n "$ios_content" ]; then
            echo "iOS version content found:"
            echo "$ios_content"
        else
            echo "No iOS version content found"
        fi
        
        # Method 3: Check what load commands actually exist
        echo "Load commands present:"
        otool -l "$lib" | grep "cmd " | head -5
        
        echo "---"
    fi
done