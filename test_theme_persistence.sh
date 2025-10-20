#!/bin/bash

echo "=== Theme Persistence Test ==="

echo "1. Current theme settings:"
grep -A 5 "\[theme\]" "/home/deepak/.config/DupFinder Team/DupFinder.conf" 2>/dev/null || echo "No theme settings found"

echo ""
echo "2. Starting application to check theme loading..."
timeout 8s ./build/dupfinder 2>&1 | grep -E "(Loading theme preference|Theme preference loaded|ThemeManager initialized)" | head -3

echo ""
echo "3. Theme settings after application run:"
grep -A 5 "\[theme\]" "/home/deepak/.config/DupFinder Team/DupFinder.conf" 2>/dev/null || echo "No theme settings found"

echo ""
echo "=== Test Complete ==="