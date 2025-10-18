#!/bin/bash

# Test Generation Tool for DupFinder Testing Suite
# 
# This script generates test files from templates, automatically replacing
# placeholders with the specified component name and test type.
#
# Usage: ./generate_test.sh <test_type> <component_name> [output_directory]
#
# Examples:
#   ./generate_test.sh unit FileScanner
#   ./generate_test.sh integration DatabaseManager tests/integration/
#   ./generate_test.sh ui MainWindow tests/ui/

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="$(dirname "$SCRIPT_DIR")/templates"
DEFAULT_OUTPUT_DIR="."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage information
show_usage() {
    cat << EOF
Test Generation Tool for DupFinder Testing Suite

USAGE:
    $0 <test_type> <component_name> [output_directory]

ARGUMENTS:
    test_type        Type of test to generate (unit, integration, ui, performance, etc.)
    component_name   Name of the component to test (e.g., FileScanner, MainWindow)
    output_directory Optional output directory (default: current directory)

AVAILABLE TEST TYPES:
    unit            Basic unit test template
    mock            Mock-based unit test template
    parameterized   Data-driven unit test template
    exception       Exception testing template
    integration     Component integration test template
    database        Database integration test template
    filesystem      File system integration test template
    ui              UI widget interaction test template
    dialog          Dialog testing template
    visual          Visual regression test template
    benchmark       Performance benchmark test template
    load            Load testing template
    memory          Memory testing template
    custom          Custom framework extension template

EXAMPLES:
    $0 unit FileScanner
    $0 integration DatabaseManager tests/integration/
    $0 ui MainWindow tests/ui/
    $0 performance HashCalculator tests/performance/

OPTIONS:
    -h, --help      Show this help message
    -l, --list      List available templates
    -v, --verbose   Enable verbose output
    --dry-run       Show what would be generated without creating files

EOF
}

# List available templates
list_templates() {
    print_info "Available test templates:"
    echo
    
    if [ -d "$TEMPLATES_DIR" ]; then
        for template in "$TEMPLATES_DIR"/*.cpp; do
            if [ -f "$template" ]; then
                basename=$(basename "$template" .cpp)
                template_type=$(echo "$basename" | sed 's/-template$//' | sed 's/-test$//')
                description=$(grep -m1 "@brief" "$template" | sed 's/.*@brief //' || echo "No description available")
                printf "  %-15s %s\n" "$template_type" "$description"
            fi
        done
    else
        print_error "Templates directory not found: $TEMPLATES_DIR"
        exit 1
    fi
    echo
}

# Validate inputs
validate_inputs() {
    local test_type="$1"
    local component_name="$2"
    
    # Check if test type is provided
    if [ -z "$test_type" ]; then
        print_error "Test type is required"
        show_usage
        exit 1
    fi
    
    # Check if component name is provided
    if [ -z "$component_name" ]; then
        print_error "Component name is required"
        show_usage
        exit 1
    fi
    
    # Validate component name format
    if ! [[ "$component_name" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]]; then
        print_error "Invalid component name: $component_name"
        print_error "Component name must start with a letter and contain only letters, numbers, and underscores"
        exit 1
    fi
    
    # Check if template exists
    local template_file="$TEMPLATES_DIR/${test_type}-test-template.cpp"
    if [ ! -f "$template_file" ]; then
        # Try alternative naming patterns
        template_file="$TEMPLATES_DIR/${test_type}-template.cpp"
        if [ ! -f "$template_file" ]; then
            template_file="$TEMPLATES_DIR/unit-test-template.cpp"  # Fallback to basic template
            if [ ! -f "$template_file" ]; then
                print_error "Template not found for test type: $test_type"
                print_info "Available templates:"
                list_templates
                exit 1
            else
                print_warning "Specific template not found, using basic unit test template"
            fi
        fi
    fi
    
    echo "$template_file"
}

# Generate test file from template
generate_test_file() {
    local template_file="$1"
    local component_name="$2"
    local output_dir="$3"
    local test_type="$4"
    
    # Create output directory if it doesn't exist
    if [ ! -d "$output_dir" ]; then
        if [ "$DRY_RUN" = "true" ]; then
            print_info "Would create directory: $output_dir"
        else
            mkdir -p "$output_dir"
            print_info "Created directory: $output_dir"
        fi
    fi
    
    # Generate output filename
    local output_file="$output_dir/test_${component_name,,}.cpp"  # Convert to lowercase
    
    # Check if output file already exists
    if [ -f "$output_file" ] && [ "$DRY_RUN" != "true" ]; then
        print_warning "Output file already exists: $output_file"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Aborted"
            exit 0
        fi
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        print_info "Would generate: $output_file"
        print_info "Template: $template_file"
        print_info "Component: $component_name"
        return 0
    fi
    
    # Perform template substitutions
    print_info "Generating test file from template..."
    
    # Create temporary file for processing
    local temp_file=$(mktemp)
    
    # Perform substitutions
    sed -e "s/COMPONENT_NAME/$component_name/g" \
        -e "s/component-name/${component_name,,}/g" \
        -e "s/COMPONENT_TYPE/$test_type/g" \
        "$template_file" > "$temp_file"
    
    # Move to final location
    mv "$temp_file" "$output_file"
    
    print_success "Generated test file: $output_file"
    
    # Generate additional files if needed
    generate_additional_files "$component_name" "$output_dir" "$test_type"
    
    # Show next steps
    show_next_steps "$output_file" "$component_name"
}

# Generate additional files (CMakeLists.txt, etc.)
generate_additional_files() {
    local component_name="$1"
    local output_dir="$2"
    local test_type="$3"
    
    # Generate CMakeLists.txt if it doesn't exist
    local cmake_file="$output_dir/CMakeLists.txt"
    if [ ! -f "$cmake_file" ]; then
        print_info "Generating CMakeLists.txt..."
        
        cat > "$cmake_file" << EOF
# CMakeLists.txt for ${component_name} tests
# Generated by DupFinder test generation tool

cmake_minimum_required(VERSION 3.16)

# Find required Qt components
find_package(Qt6 REQUIRED COMPONENTS Core Test)

# Enable automatic MOC, UIC, and RCC processing
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Test executable
add_executable(test_${component_name,,}
    test_${component_name,,}.cpp
)

# Link libraries
target_link_libraries(test_${component_name,,}
    Qt6::Core
    Qt6::Test
    # Add your component library here
    # ${component_name}
)

# Include directories
target_include_directories(test_${component_name,,} PRIVATE
    \${CMAKE_SOURCE_DIR}/include
    \${CMAKE_SOURCE_DIR}/tests/framework
)

# Add test to CTest
add_test(NAME ${component_name}Test COMMAND test_${component_name,,})

# Set test properties
set_tests_properties(${component_name}Test PROPERTIES
    LABELS "${test_type}"
    TIMEOUT 300
)
EOF
        
        print_success "Generated CMakeLists.txt: $cmake_file"
    fi
}

# Show next steps to the user
show_next_steps() {
    local output_file="$1"
    local component_name="$2"
    
    echo
    print_success "Test generation completed!"
    echo
    print_info "Next steps:"
    echo "  1. Edit the generated file: $output_file"
    echo "  2. Replace TODO comments with actual test implementation"
    echo "  3. Update #include statements for your component"
    echo "  4. Compile and run the test:"
    echo
    echo "     # Compile"
    echo "     g++ -I/path/to/qt/include -I/path/to/qt/include/QtTest \\"
    echo "         -I/path/to/your/headers \\"
    echo "         $output_file -o test_${component_name,,} \\"
    echo "         -lQt6Test -lQt6Core"
    echo
    echo "     # Run"
    echo "     ./test_${component_name,,}"
    echo
    print_info "For more information, see the testing documentation:"
    echo "  - Framework Overview: docs/testing/framework-overview.md"
    echo "  - Test Writing Guidelines: docs/testing/test-writing-guidelines.md"
    echo "  - Examples: docs/testing/examples/"
    echo
}

# Parse command line arguments
VERBOSE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--list)
            list_templates
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Get arguments
TEST_TYPE="$1"
COMPONENT_NAME="$2"
OUTPUT_DIR="${3:-$DEFAULT_OUTPUT_DIR}"

# Validate inputs and get template file
TEMPLATE_FILE=$(validate_inputs "$TEST_TYPE" "$COMPONENT_NAME")

# Enable verbose output if requested
if [ "$VERBOSE" = "true" ]; then
    set -x
fi

# Generate the test file
print_info "Generating $TEST_TYPE test for $COMPONENT_NAME..."
generate_test_file "$TEMPLATE_FILE" "$COMPONENT_NAME" "$OUTPUT_DIR" "$TEST_TYPE"

exit 0