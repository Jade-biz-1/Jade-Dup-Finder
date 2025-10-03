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
else()
    # Enable all warnings and treat warnings as errors for GCC/Clang
    add_compile_options(-Wall -Wextra -Werror -pedantic)
    # Enable additional security features
    add_compile_options(-fstack-protector-strong)
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
)

set(GUI_SOURCES
    src/gui/main_window.cpp
    src/gui/scan_dialog.cpp
    src/gui/results_widget.cpp
    src/gui/settings_dialog.cpp
    src/gui/confirmation_dialog.cpp
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
    include/main_window.h
    include/scan_dialog.h
    include/results_widget.h
    include/settings_dialog.h
    include/confirmation_dialog.h
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

# Print build configuration summary
message(STATUS "")
message(STATUS "DupFinder Build Configuration Summary")
message(STATUS "=====================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${PLATFORM_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Qt6 Version: ${Qt6_VERSION}")
message(STATUS "")