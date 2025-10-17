cmake_minimum_required(VERSION 3.20)

# Project configuration
project(DupFinder
    VERSION 1.0.0
    DESCRIPTION "Cross-platform duplicate file finder"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build optimization settings
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)  # Enable LTO
endif()

# Enable parallel builds
if(CMAKE_GENERATOR MATCHES "Makefiles")
    include(ProcessorCount)
    ProcessorCount(N)
    if(NOT N EQUAL 0)
        set(CMAKE_MAKE_PROGRAM "${CMAKE_MAKE_PROGRAM} -j${N}")
    endif()
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif()

# Find required Qt6 components
find_package(Qt6 REQUIRED COMPONENTS
    Core
    Widgets
    Concurrent
    Network
)

# Platform detection
if(WIN32)
    set(PLATFORM_NAME "windows")
elseif(APPLE)
    set(PLATFORM_NAME "macos")
elseif(UNIX)
    set(PLATFORM_NAME "linux")
endif()

# Compiler-specific flags
if(MSVC)
    # Enable all warnings and treat warnings as errors for MSVC
    add_compile_options(/W4 /WX)
    # Enable parallel compilation
    add_compile_options(/MP)
    # Security features
    add_compile_options(/guard:cf /DYNAMICBASE /NXCOMPAT)
else()
    # Enable comprehensive warnings for GCC/Clang
    add_compile_options(-Wall -Wextra -pedantic)
    # Additional useful warnings
    add_compile_options(-Wconversion -Wsign-conversion -Wunused -Wcast-align)
    # Security features
    add_compile_options(-fstack-protector-strong -D_FORTIFY_SOURCE=2)
    # Debug info in release builds for profiling
    if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_compile_options(-g -O2)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Source files
set(CORE_SOURCES
    src/core/duplicate_detector.cpp
    src/core/file_scanner.cpp
    src/core/hash_calculator.cpp
    src/core/file_manager.cpp
    src/core/safety_manager.cpp
    src/core/app_config.cpp
    src/core/logger.cpp
    src/core/scan_history_manager.cpp
    src/core/selection_history_manager.cpp
    src/core/file_operation_queue.cpp
)

set(GUI_SOURCES
    src/gui/main_window.cpp
    src/gui/main_window_widgets.cpp
    src/gui/scan_dialog.cpp
    src/gui/results_widget.cpp
    src/gui/results_window.cpp
    src/gui/settings_dialog.cpp
    src/gui/scan_history_dialog.cpp
    src/gui/restore_dialog.cpp
    src/gui/confirmation_dialog.cpp
    src/gui/thumbnail_cache.cpp
    src/gui/thumbnail_delegate.cpp
    src/gui/exclude_pattern_widget.cpp
    src/gui/preset_manager_dialog.cpp
    src/gui/scan_scope_preview_widget.cpp
    src/gui/scan_progress_dialog.cpp
    src/gui/scan_error_dialog.cpp
    src/gui/advanced_filter_dialog.cpp
    src/gui/file_operation_progress_dialog.cpp
    src/gui/grouping_options_dialog.cpp
    src/gui/safety_features_dialog.cpp
    src/gui/duplicate_relationship_widget.cpp
    src/gui/smart_selection_dialog.cpp
)

set(PLATFORM_SOURCES
    src/platform/${PLATFORM_NAME}/platform_file_ops.cpp
    src/platform/${PLATFORM_NAME}/trash_manager.cpp
    src/platform/${PLATFORM_NAME}/system_integration.cpp
)

set(HEADER_FILES
    include/duplicate_detector.h
    include/file_scanner.h
    include/hash_calculator.h
    include/file_manager.h
    include/safety_manager.h
    include/app_config.h
    src/core/logger.h
    include/scan_history_manager.h
    include/selection_history_manager.h
    include/file_operation_queue.h
    include/main_window.h
    include/scan_dialog.h
    include/results_widget.h
    src/gui/results_window.h
    include/settings_dialog.h
    include/scan_history_dialog.h
    include/restore_dialog.h
    include/confirmation_dialog.h
    include/thumbnail_cache.h
    include/thumbnail_delegate.h
    include/exclude_pattern_widget.h
    include/preset_manager_dialog.h
    include/scan_scope_preview_widget.h
    include/scan_progress_dialog.h
    include/scan_error_dialog.h
    include/advanced_filter_dialog.h
    include/file_operation_progress_dialog.h
    include/duplicate_relationship_widget.h
    include/smart_selection_dialog.h
    include/grouping_options_dialog.h
    include/safety_features_dialog.h
    include/platform_file_ops.h
    include/trash_manager.h
    include/system_integration.h
)

# Main executable
add_executable(dupfinder
    src/main.cpp
    ${CORE_SOURCES}
    ${GUI_SOURCES}
    ${PLATFORM_SOURCES}
    ${HEADER_FILES}
)

# Link Qt6 libraries
target_link_libraries(dupfinder
    Qt6::Core
    Qt6::Widgets
    Qt6::Concurrent
    Qt6::Network
)

# Platform-specific linking
if(WIN32)
    # Windows specific libraries
    target_link_libraries(dupfinder shell32 ole32)
    # Set Windows application properties
    set_target_properties(dupfinder PROPERTIES
        WIN32_EXECUTABLE TRUE
    )
elseif(APPLE)
    # macOS specific libraries
    find_library(FOUNDATION_LIB Foundation)
    find_library(APPKIT_LIB AppKit)
    target_link_libraries(dupfinder ${FOUNDATION_LIB} ${APPKIT_LIB})
    # Set macOS bundle properties
    set_target_properties(dupfinder PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/resources/Info.plist
    )
endif()

# Enable Qt's automatic MOC, UIC, and RCC processing
set_target_properties(dupfinder PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
    # Optimization properties
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    # Version information
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Resources
set(RESOURCE_FILES
    resources/dupfinder.qrc
)

# Add resources to target
target_sources(dupfinder PRIVATE ${RESOURCE_FILES})


# Install configuration
install(TARGETS dupfinder
    BUNDLE DESTINATION .
    RUNTIME DESTINATION bin
)

# Platform-specific install configuration
if(WIN32)
    # Windows installer configuration
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "DupFinder")
    set(CPACK_NSIS_PACKAGE_NAME "DupFinder")
elseif(APPLE)
    # macOS bundle configuration
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "DupFinder")
elseif(UNIX)
    # Linux package configuration
    set(CPACK_GENERATOR "DEB;RPM;TGZ")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "DupFinder Team")
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
endif()

# CPack configuration
set(CPACK_PACKAGE_NAME "DupFinder")
set(CPACK_PACKAGE_VENDOR "DupFinder Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Cross-platform duplicate file finder")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "DupFinder")

include(CPack)

# Testing configuration
enable_testing()

# Add subdirectories for tests
add_subdirectory(tests)

# Note: Test files have been moved to tests/ directory and are handled by tests/CMakeLists.txt

# Custom target for running all tests
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMENT "Running all tests"
    USES_TERMINAL
)

# Development tools
find_program(CLANG_FORMAT_EXECUTABLE clang-format)
if(CLANG_FORMAT_EXECUTABLE)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file 
            ${CORE_SOURCES} ${GUI_SOURCES} ${PLATFORM_SOURCES} ${HEADER_FILES}
        COMMENT "Formatting source code"
        VERBATIM
    )
endif()

# Code analysis tools
find_program(CPPCHECK_EXECUTABLE cppcheck)
if(CPPCHECK_EXECUTABLE)
    add_custom_target(cppcheck
        COMMAND ${CPPCHECK_EXECUTABLE}
            --enable=all
            --std=c++17
            --verbose
            --error-exitcode=1
            --suppress=missingIncludeSystem
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        COMMENT "Running static analysis with cppcheck"
        VERBATIM
    )
endif()

# Memory check target (valgrind on Linux)
if(UNIX AND NOT APPLE)
    find_program(VALGRIND_EXECUTABLE valgrind)
    if(VALGRIND_EXECUTABLE)
        add_custom_target(memcheck
            COMMAND ${VALGRIND_EXECUTABLE}
                --tool=memcheck
                --leak-check=full
                --show-reachable=yes
                --track-origins=yes
                --error-exitcode=1
                $<TARGET_FILE:dupfinder>
            DEPENDS dupfinder
            COMMENT "Running memory check with valgrind"
            VERBATIM
        )
    endif()
endif()

# Coverage analysis (only in Debug mode)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    option(ENABLE_COVERAGE "Enable coverage analysis" OFF)
    if(ENABLE_COVERAGE)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            add_compile_options(--coverage)
            add_link_options(--coverage)
            
            find_program(GCOV_EXECUTABLE gcov)
            find_program(LCOV_EXECUTABLE lcov)
            find_program(GENHTML_EXECUTABLE genhtml)
            
            if(GCOV_EXECUTABLE AND LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
                add_custom_target(coverage
                    COMMAND ${CMAKE_COMMAND} -E make_directory coverage
                    COMMAND ${LCOV_EXECUTABLE} --capture --directory . --output-file coverage/coverage.info
                    COMMAND ${LCOV_EXECUTABLE} --remove coverage/coverage.info '/usr/*' --output-file coverage/coverage.info
                    COMMAND ${LCOV_EXECUTABLE} --remove coverage/coverage.info '*/tests/*' --output-file coverage/coverage.info
                    COMMAND ${GENHTML_EXECUTABLE} coverage/coverage.info --output-directory coverage/html
                    COMMENT "Generating coverage report"
                    VERBATIM
                )
            endif()
        endif()
    endif()
endif()

# Documentation build (optional)
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    if(EXISTS ${DOXYGEN_IN})
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        add_custom_target(docs ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
    else()
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} -g
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
            COMMENT "Generate Doxygen configuration file"
            VERBATIM
        )
    endif()
endif()

# Build summary target
add_custom_target(summary
    COMMAND ${CMAKE_COMMAND} -E echo "Build Summary:"
    COMMAND ${CMAKE_COMMAND} -E echo "=============="
    COMMAND ${CMAKE_COMMAND} -E echo "Available targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  dupfinder      - Main application"
    COMMAND ${CMAKE_COMMAND} -E echo "  unit_tests     - Unit tests"
    COMMAND ${CMAKE_COMMAND} -E echo "  integration_tests - Integration tests"
    COMMAND ${CMAKE_COMMAND} -E echo "  check          - Run all tests"
    COMMAND ${CMAKE_COMMAND} -E echo "  package        - Create distribution package"
    COMMAND ${CMAKE_COMMAND} -E echo "Development targets (if available):"
    COMMAND ${CMAKE_COMMAND} -E echo "  format         - Format source code"
    COMMAND ${CMAKE_COMMAND} -E echo "  cppcheck       - Static analysis"
    COMMAND ${CMAKE_COMMAND} -E echo "  memcheck       - Memory analysis (Linux)"
    COMMAND ${CMAKE_COMMAND} -E echo "  coverage       - Coverage report (Debug)"
    COMMAND ${CMAKE_COMMAND} -E echo "  docs           - API documentation"
    VERBATIM
)

# Print build configuration summary
message(STATUS "")
message(STATUS "DupFinder Build Configuration Summary")
message(STATUS "=====================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${PLATFORM_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Qt6 Version: ${Qt6_VERSION}")
message(STATUS "Available Build Targets:")
message(STATUS "  - dupfinder (main application)")
message(STATUS "  - check (run all tests)")
message(STATUS "  - package (create installer)")
if(CLANG_FORMAT_EXECUTABLE)
    message(STATUS "  - format (code formatting)")
endif()
if(CPPCHECK_EXECUTABLE)
    message(STATUS "  - cppcheck (static analysis)")
endif()
if(VALGRIND_EXECUTABLE)
    message(STATUS "  - memcheck (memory analysis)")
endif()
if(DOXYGEN_FOUND)
    message(STATUS "  - docs (API documentation)")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND ENABLE_COVERAGE)
    message(STATUS "  - coverage (test coverage)")
endif()
message(STATUS "")
message(STATUS "Run 'make summary' for a complete target list")
message(STATUS "")
