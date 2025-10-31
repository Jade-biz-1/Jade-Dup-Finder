#!/bin/bash
# Convenience script to check theme compliance
# Usage: ./check_theme_compliance.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Theme Compliance Checker${NC}"
echo "========================"
echo ""

# Check if tool is built
if [ ! -f "build/tools/theme_compliance_checker" ]; then
    echo -e "${RED}✗ Tool not built yet${NC}"
    echo "Building theme_compliance_checker..."
    
    cd build
    cmake ..
    make theme_compliance_checker
    cd ..
    
    if [ ! -f "build/tools/theme_compliance_checker" ]; then
        echo -e "${RED}✗ Build failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Tool built successfully${NC}"
    echo ""
fi

# Run the checker
echo "Scanning source code..."
echo ""

if ./build/tools/theme_compliance_checker ./src "$@"; then
    echo ""
    echo -e "${GREEN}✓ No theme compliance violations found!${NC}"
    exit 0
else
    echo ""
    echo -e "${YELLOW}⚠ Theme compliance violations detected${NC}"
    echo ""
    echo "To fix violations:"
    echo "  1. Review the violations listed above"
    echo "  2. Use ThemeManager methods instead of hardcoded styles"
    echo "  3. Run this script again to verify fixes"
    echo ""
    echo "For more info, see: tools/README.md"
    exit 1
fi
