cmake_minimum_required(VERSION 3.20)

# Project configuration
project(CloneClean
    VERSION 1.0.0
    DESCRIPTION "One File. One Place. - Intelligent duplicate file finder and cleaner"
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

# Find required Qt6 components (with Qt5 fallback for development)
find_package(Qt6 QUIET COMPONENTS Core Widgets Concurrent Test Network)
if(Qt6_FOUND)
    message(STATUS "Found Qt6: ${Qt6_VERSION}")
    set(QT_VERSION 6)
else()
    message(WARNING "Qt6 not found, trying Qt5...")
    find_package(Qt5 REQUIRED COMPONENTS Core Widgets Concurrent Test Network)
    if(Qt5_FOUND)
        message(STATUS "Found Qt5: ${Qt5_VERSION}")
        set(QT_VERSION 5)
    else()
        message(FATAL_ERROR "Neither Qt6 nor Qt5 found. Please install Qt from https://www.qt.io/download")
    endif()
endif()

# GPU acceleration support
option(ENABLE_GPU_ACCELERATION "Enable GPU acceleration for hash calculations" ON)

set(CLONECLEAN_BUILD_VARIANT "auto" CACHE STRING "Select build variant: auto, cpu, or gpu")
set_property(CACHE CLONECLEAN_BUILD_VARIANT PROPERTY STRINGS auto cpu gpu)

if(CLONECLEAN_BUILD_VARIANT STREQUAL "cpu")
    set(ENABLE_GPU_ACCELERATION OFF CACHE BOOL "Enable GPU acceleration for hash calculations" FORCE)
elseif(CLONECLEAN_BUILD_VARIANT STREQUAL "gpu")
    set(ENABLE_GPU_ACCELERATION ON CACHE BOOL "Enable GPU acceleration for hash calculations" FORCE)
endif()

if(ENABLE_GPU_ACCELERATION)
    message(STATUS "Checking for GPU acceleration libraries...")

    # Try CUDA first (preferred for NVIDIA GPUs)
    # Use modern CMake CUDA language support instead of deprecated FindCUDA
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        # Find CUDA Toolkit to get imported targets like CUDA::cudart
        find_package(CUDAToolkit REQUIRED)
        set(CUDA_FOUND TRUE)
        set(CUDA_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        message(STATUS "CUDA toolkit root: ${CUDAToolkit_INCLUDE_DIRS}")
        # Modern CMake uses CUDA language support - no need for manual library/include setup
        set(GPU_BACKEND "CUDA")
        set(USE_CUDA ON)
        set(HAS_CUDA ON)
    else()
        message(STATUS "CUDA not found, trying OpenCL...")

        # Fallback to OpenCL (works with AMD/Intel GPUs and MinGW)
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            message(STATUS "OpenCL found: Enabling OpenCL acceleration")
            set(GPU_BACKEND "OpenCL")
            set(USE_OPENCL ON)
            set(HAS_OPENCL ON)
        else()
            message(WARNING "No GPU acceleration libraries found. Building CPU-only version.")
            message(WARNING "To enable GPU acceleration:")
            message(WARNING "  - For NVIDIA: Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
            message(WARNING "    Then use: cmake .. -G \"Visual Studio 16 2019\" (or appropriate VS version)")
            message(WARNING "  - For AMD/Intel: Install OpenCL development packages")
            set(ENABLE_GPU_ACCELERATION OFF)
        endif()
    endif()

    if(ENABLE_GPU_ACCELERATION)
        message(STATUS "GPU acceleration enabled with backend: ${GPU_BACKEND}")
    endif()
endif()

# Platform detection
if(WIN32)
    set(PLATFORM_NAME "windows")
elseif(APPLE)
    set(PLATFORM_NAME "macos")
elseif(UNIX)
    set(PLATFORM_NAME "linux")
endif()

# Compiler-specific flags
option(CLONECLEAN_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(MSVC)
    # Enable all warnings for MSVC (C++ only - CUDA uses its own warning flags)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
    if(CLONECLEAN_WARNINGS_AS_ERRORS)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
    # Enable parallel compilation for C++ only (not CUDA - nvcc doesn't support /MP)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/MP>)
    # Security features for C++ only (CUDA passes different flags)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/guard:cf>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/DYNAMICBASE>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/NXCOMPAT>)
else()
    # Enable comprehensive warnings for GCC/Clang
    add_compile_options(-Wall -Wextra -pedantic)
    # Additional useful warnings
    add_compile_options(-Wconversion -Wsign-conversion -Wunused -Wcast-align)
    if(CLONECLEAN_WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()
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

# Add GPU include directories if enabled
if(ENABLE_GPU_ACCELERATION)
    if(USE_CUDA)
        # Modern CMake CUDA language support handles include directories automatically
        # No manual include_directories needed
    elseif(USE_OPENCL)
        include_directories(${OpenCL_INCLUDE_DIRS})
    endif()
endif()

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
    src/core/theme_manager.cpp
    src/core/theme_error_handler.cpp
    src/core/theme_performance_optimizer.cpp
    src/core/final_theme_validator.cpp
    src/core/component_registry.cpp
    src/core/style_validator.cpp
    src/core/theme_persistence.cpp
    src/core/window_state_manager.cpp
    # Advanced Detection Algorithms (Phase 2)
    src/core/detection_algorithm.cpp
    src/core/detection_algorithm_factory.cpp
    src/core/exact_hash_algorithm.cpp
    src/core/quick_scan_algorithm.cpp
    src/core/perceptual_hash_algorithm.cpp
    src/core/document_similarity_algorithm.cpp
    # File Type Enhancements (Phase 2)
    src/core/archive_handler.cpp
    src/core/document_handler.cpp
    src/core/media_handler.cpp
    src/core/file_type_manager.cpp
)

# Add GPU sources if enabled
if(ENABLE_GPU_ACCELERATION)
    list(APPEND CORE_SOURCES
        # GPU Acceleration (Phase 2)
        src/gpu/gpu_detector.cpp
        src/gpu/gpu_hash_calculator.cpp
        src/gpu/gpu_context.cpp
    )
endif()

set(GUI_SOURCES
    src/gui/main_window.cpp
    src/gui/main_window_widgets.cpp
    src/gui/scan_dialog.cpp
    # src/gui/results_widget.cpp  # Obsolete stub file - removed
    src/gui/results_window.cpp
    src/gui/loading_overlay.cpp
    src/gui/settings_dialog.cpp
    src/gui/scan_history_dialog.cpp
    src/gui/restore_dialog.cpp
    # src/gui/confirmation_dialog.cpp  # Obsolete stub file - removed
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
    src/gui/about_dialog.cpp
    src/gui/theme_editor.cpp
    src/gui/theme_recovery_dialog.cpp
    src/gui/theme_notification_widget.cpp
    # Temporarily disabled for Settings dialog testing
    # src/gui/duplicate_relationship_widget.cpp
    # src/gui/smart_selection_dialog.cpp
)

set(PLATFORM_SOURCES
    src/platform/${PLATFORM_NAME}/platform_file_ops.cpp
    src/platform/${PLATFORM_NAME}/system_integration.cpp
)

# Platform-specific trash manager (macOS uses Objective-C++)
if(APPLE)
    list(APPEND PLATFORM_SOURCES src/platform/${PLATFORM_NAME}/trash_manager.mm)
else()
    list(APPEND PLATFORM_SOURCES src/platform/${PLATFORM_NAME}/trash_manager.cpp)
endif()

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
    include/theme_manager.h
    include/theme_error_handler.h
    include/theme_performance_optimizer.h
    include/final_theme_validator.h
    include/component_registry.h
    include/style_validator.h
    include/theme_persistence.h
    include/theme_editor.h
    src/core/window_state_manager.h
    # Advanced Detection Algorithms (Phase 2)
    src/core/detection_algorithm.h
    src/core/detection_algorithm_factory.h
    src/core/exact_hash_algorithm.h
    src/core/quick_scan_algorithm.h
    src/core/perceptual_hash_algorithm.h
    src/core/document_similarity_algorithm.h
    # File Type Enhancements (Phase 2)
    src/core/archive_handler.h
    src/core/document_handler.h
    src/core/media_handler.h
    src/core/file_type_manager.h
    # GUI headers required for Qt's MOC
    include/main_window.h
    include/scan_dialog.h
    include/results_window.h
    include/loading_overlay.h
    include/settings_dialog.h
    include/scan_history_dialog.h
    include/restore_dialog.h
    include/thumbnail_cache.h
    include/thumbnail_delegate.h
    include/exclude_pattern_widget.h
    include/preset_manager_dialog.h
    include/scan_scope_preview_widget.h
    include/scan_progress_dialog.h
    include/scan_error_dialog.h
    include/advanced_filter_dialog.h
    include/file_operation_progress_dialog.h
    include/grouping_options_dialog.h
    include/safety_features_dialog.h
    include/about_dialog.h
    include/theme_notification_widget.h
    include/theme_recovery_dialog.h
)

# Add GPU headers if enabled
if(ENABLE_GPU_ACCELERATION)
    list(APPEND HEADER_FILES
        # GPU Acceleration (Phase 2)
        src/gpu/gpu_detector.h
        src/gpu/gpu_config.h
        src/gpu/gpu_hash_calculator.h
        src/gpu/gpu_context.h
        src/gpu/cuda_hash_calculator.h
        src/gpu/opencl_hash_calculator.h
    )
endif()

# Main executable
add_executable(cloneclean
    src/main.cpp
    ${CORE_SOURCES}
    ${GUI_SOURCES}
    ${PLATFORM_SOURCES}
    ${HEADER_FILES}
)

# Note: With modern CMake and AUTOMOC enabled, MOC files are handled automatically
# Visual Studio multi-config generators will compile the autogen MOC files without manual intervention

# Add preprocessor definitions for GPU backends
if(ENABLE_GPU_ACCELERATION)
    if(USE_CUDA)
        target_compile_definitions(cloneclean PRIVATE HAS_CUDA)
    elseif(USE_OPENCL)
        target_compile_definitions(cloneclean PRIVATE HAS_OPENCL)
    endif()
endif()

# GPU-specific compilation
if(ENABLE_GPU_ACCELERATION)
    if(USE_CUDA)
        # CUDA requires MSVC on Windows - check if we're using the right compiler
        if(WIN32 AND NOT MSVC)
            message(FATAL_ERROR "CUDA acceleration requires MSVC compiler on Windows.\n"
                              "Current compiler: ${CMAKE_CXX_COMPILER_ID}\n"
                              "To enable CUDA acceleration:\n"
                              "1. Open 'Visual Studio Developer Command Prompt' (x64 Native Tools)\n"
                              "2. Run: cmake .. -G \"Visual Studio 16 2019\" (or appropriate VS version)\n"
                              "3. Build with: cmake --build . --config Release\n"
                              "Alternatively, you can disable GPU acceleration: cmake .. -DENABLE_GPU_ACCELERATION=OFF")
        else()
            message(STATUS "CUDA compilation enabled with MSVC")
            # Modern CMake CUDA language support
            add_library(gpu_acceleration STATIC
                src/gpu/cuda_hash_calculator.cu
            )
            # Set CUDA properties for the library
            set_target_properties(gpu_acceleration PROPERTIES
                CUDA_STANDARD 17
                CUDA_ARCHITECTURES "60;70;75;80;86"  # Support common NVIDIA GPU architectures
                # Disable problematic MSVC flags for CUDA compilation
                COMPILE_OPTIONS ""
            )
            # Remove /MP flag for CUDA files (incompatible with nvcc)
            if(MSVC)
                target_compile_options(gpu_acceleration PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>  # Suppress charset warnings
                )
            endif()
            # Link CUDA runtime to the GPU library
            target_link_libraries(gpu_acceleration CUDA::cudart)
        endif()
    elseif(USE_OPENCL)
        message(STATUS "OpenCL compilation enabled")
        add_library(gpu_acceleration STATIC
            src/gpu/opencl_hash_calculator.cpp
        )
    endif()
endif()

# Link Qt libraries
if(QT_VERSION EQUAL 6)
    target_link_libraries(cloneclean
        Qt6::Core
        Qt6::Widgets
        Qt6::Concurrent
        Qt6::Network
    )
else()
    target_link_libraries(cloneclean
        Qt5::Core
        Qt5::Widgets
        Qt5::Concurrent
        Qt5::Network
    )
endif()

# Link GPU libraries
if(ENABLE_GPU_ACCELERATION)
    target_link_libraries(cloneclean gpu_acceleration)
    if(USE_CUDA)
        # Modern CMake CUDA language support - link with CUDA runtime
        target_link_libraries(cloneclean CUDA::cudart)
    elseif(USE_OPENCL)
        target_link_libraries(cloneclean OpenCL::OpenCL)
    endif()
endif()

# Platform-specific linking
if(WIN32)
    # Windows specific libraries
    target_link_libraries(cloneclean shell32 ole32)
    # Set Windows application properties
    set_target_properties(cloneclean PROPERTIES
        WIN32_EXECUTABLE TRUE
    )
elseif(APPLE)
    # macOS specific libraries
    find_library(FOUNDATION_LIB Foundation)
    find_library(APPKIT_LIB AppKit)
    target_link_libraries(cloneclean ${FOUNDATION_LIB} ${APPKIT_LIB})

    # Set macOS bundle properties
    set_target_properties(cloneclean PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/resources/Info.plist
        OUTPUT_NAME "cloneclean"
    )
endif()

# Enable Qt's automatic MOC, UIC, and RCC processing
set_target_properties(cloneclean PROPERTIES
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
    resources/cloneclean.qrc
)

# Add resources to target
target_sources(cloneclean PRIVATE ${RESOURCE_FILES})


# Install configuration
install(TARGETS cloneclean
    BUNDLE DESTINATION .
    RUNTIME DESTINATION bin
)

# Qt deployment for Windows
if(WIN32)
    if(QT_VERSION EQUAL 6)
        find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${Qt6_DIR}/../../../bin" "${Qt6_DIR}/../../bin")
    else()
        find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${Qt5_DIR}/../../../bin" "${Qt5_DIR}/../../bin")
    endif()

    if(WINDEPLOYQT_EXECUTABLE)
        message(STATUS "windeployqt found: ${WINDEPLOYQT_EXECUTABLE}")
        
        # Deploy Qt libraries after build
        add_custom_command(TARGET cloneclean POST_BUILD
            COMMAND ${WINDEPLOYQT_EXECUTABLE}
                --no-translations
                --no-compiler-runtime
                --no-system-d3d-compiler
                --dir $<TARGET_FILE_DIR:cloneclean>
                $<TARGET_FILE:cloneclean>
            COMMENT "Deploying Qt libraries for CloneClean"
            VERBATIM
        )

        # Install Qt DLLs deployed by windeployqt
        # Handle both Debug (d suffix) and Release builds
        if(QT_VERSION EQUAL 6)
            # Try both debug and release variants
            install(FILES
                $<TARGET_FILE_DIR:cloneclean>/Qt6Core.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt6Cored.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt6Gui.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt6Guid.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt6Widgets.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt6Widgetsd.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt6Network.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt6Networkd.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt6Svg.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt6Svgd.dll
                $<TARGET_FILE_DIR:cloneclean>/opengl32sw.dll
                DESTINATION bin
                OPTIONAL
            )
        else()
            # Try both debug and release variants
            install(FILES
                $<TARGET_FILE_DIR:cloneclean>/Qt5Core.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt5Cored.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt5Gui.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt5Guid.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt5Widgets.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt5Widgetsd.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt5Network.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt5Networkd.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt5Svg.dll
                $<TARGET_FILE_DIR:cloneclean>/Qt5Svgd.dll
                $<TARGET_FILE_DIR:cloneclean>/opengl32sw.dll
                DESTINATION bin
                OPTIONAL
            )
        endif()

        # Install Qt plugin directories deployed by windeployqt
        install(DIRECTORY $<TARGET_FILE_DIR:cloneclean>/platforms
            DESTINATION bin
            OPTIONAL
        )
        install(DIRECTORY $<TARGET_FILE_DIR:cloneclean>/generic
            DESTINATION bin
            OPTIONAL
        )
        install(DIRECTORY $<TARGET_FILE_DIR:cloneclean>/iconengines
            DESTINATION bin
            OPTIONAL
        )
        install(DIRECTORY $<TARGET_FILE_DIR:cloneclean>/imageformats
            DESTINATION bin
            OPTIONAL
        )
        install(DIRECTORY $<TARGET_FILE_DIR:cloneclean>/styles
            DESTINATION bin
            OPTIONAL
        )
        install(DIRECTORY $<TARGET_FILE_DIR:cloneclean>/networkinformation
            DESTINATION bin
            OPTIONAL
        )
        install(DIRECTORY $<TARGET_FILE_DIR:cloneclean>/tls
            DESTINATION bin
            OPTIONAL
        )
    else()
        message(WARNING "windeployqt not found. Qt libraries will not be automatically deployed. Install Qt Creator or add windeployqt to PATH.")
    endif()
elseif(APPLE)
    # Qt deployment for macOS
    # Find macdeployqt tool
    if(QT_VERSION EQUAL 6)
        find_program(MACDEPLOYQT_EXECUTABLE macdeployqt HINTS "${Qt6_DIR}/../../../libexec" "${Qt6_DIR}/../../libexec" "${Qt6_DIR}/../../../bin" "${Qt6_DIR}/../../bin")
    else()
        find_program(MACDEPLOYQT_EXECUTABLE macdeployqt HINTS "${Qt5_DIR}/../../../libexec" "${Qt5_DIR}/../../libexec" "${Qt5_DIR}/../../../bin" "${Qt5_DIR}/../../bin")
    endif()

    if(MACDEPLOYQT_EXECUTABLE)
        message(STATUS "macdeployqt found: ${MACDEPLOYQT_EXECUTABLE}")

        # Deploy Qt frameworks into the app bundle after build
        # macdeployqt automatically sets the rpath, so we don't need to do it manually
        add_custom_command(TARGET cloneclean POST_BUILD
            COMMAND ${MACDEPLOYQT_EXECUTABLE}
                $<TARGET_BUNDLE_DIR:cloneclean>
                -always-overwrite
            COMMENT "Deploying Qt frameworks for macOS CloneClean"
            VERBATIM
        )
    else()
        message(WARNING "macdeployqt not found. Qt frameworks will not be automatically deployed.")
        message(WARNING "To enable macOS deployment:")
        message(WARNING "  - Install Qt via Homebrew: brew install qt@6")
        message(WARNING "  - Or install Qt from qt.io with the macOS deployment tools")
    endif()
endif()

# Platform-specific install configuration
if(WIN32)
    # Find NSIS executable
    # Note: Use ProgramFiles(x86) via cmake variable, not ENV due to parentheses
    file(TO_CMAKE_PATH "$ENV{ProgramFiles}" PROGRAM_FILES_PATH)
    if(DEFINED ENV{ProgramW6432})
        file(TO_CMAKE_PATH "$ENV{ProgramW6432}" PROGRAM_FILES_64_PATH)
    else()
        set(PROGRAM_FILES_64_PATH "${PROGRAM_FILES_PATH}")
    endif()
    # ProgramFiles(x86) is only set on 64-bit Windows
    if(DEFINED ENV{ProgramFiles\(x86\)})
        file(TO_CMAKE_PATH "$ENV{ProgramFiles\(x86\)}" PROGRAM_FILES_X86_PATH)
    else()
        set(PROGRAM_FILES_X86_PATH "${PROGRAM_FILES_PATH}")
    endif()

    find_program(NSIS_EXECUTABLE makensis
        PATHS
            "C:/Program Files (x86)/NSIS"
            "C:/Program Files/NSIS"
            "${PROGRAM_FILES_X86_PATH}/NSIS"
            "${PROGRAM_FILES_64_PATH}/NSIS"
            "${PROGRAM_FILES_PATH}/NSIS"
    )
    
    if(NSIS_EXECUTABLE)
        # Windows installer configuration
        set(CPACK_GENERATOR "NSIS")
        set(CPACK_NSIS_DISPLAY_NAME "CloneClean")
        set(CPACK_NSIS_PACKAGE_NAME "CloneClean")
        set(CPACK_NSIS_EXECUTABLE "${NSIS_EXECUTABLE}")

        # Create Start menu shortcuts
        set(CPACK_NSIS_MENU_LINKS
            "bin/cloneclean.exe" "CloneClean"
            "https://github.com/Jade-biz-1/Jade-Dup-Finder" "CloneClean on GitHub"
        )

        # Create desktop shortcut and uninstall shortcut in Start Menu
        set(CPACK_NSIS_CREATE_ICONS_EXTRA
            "CreateShortCut \\\"$DESKTOP\\\\CloneClean.lnk\\\" \\\"$INSTDIR\\\\bin\\\\cloneclean.exe\\\"
             CreateShortCut \\\"$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\Uninstall.lnk\\\" \\\"$INSTDIR\\\\Uninstall.exe\\\""
        )
        set(CPACK_NSIS_DELETE_ICONS_EXTRA
            "Delete \\\"$DESKTOP\\\\CloneClean.lnk\\\"
             Delete \\\"$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\Uninstall.lnk\\\""
        )

        # Installer branding
        set(CPACK_NSIS_CONTACT "CloneClean Team")
        set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/Jade-biz-1/Jade-Dup-Finder")
        set(CPACK_NSIS_HELP_LINK "https://github.com/Jade-biz-1/Jade-Dup-Finder")
        
        message(STATUS "NSIS found: ${NSIS_EXECUTABLE}")
    else()
        message(WARNING "NSIS not found. Windows installer will not be available. Install NSIS from https://nsis.sourceforge.io/")
        # Fallback to ZIP packaging
        set(CPACK_GENERATOR "ZIP")
    endif()
elseif(APPLE)
    # macOS bundle configuration
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "CloneClean")
    set(CPACK_DMG_FORMAT "UDZO")  # Compressed DMG
    set(CPACK_BUNDLE_NAME "CloneClean")
    set(CPACK_BUNDLE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.icns")
    set(CPACK_BUNDLE_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/resources/Info.plist")

    message(STATUS "macOS packaging configured: DMG installer")
elseif(UNIX)
    # Linux package configuration
    set(CPACK_GENERATOR "DEB;RPM;TGZ")

    # Debian package configuration
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "CloneClean Team")
    set(CPACK_DEBIAN_PACKAGE_SECTION "utils")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libqt6core6, libqt6gui6, libqt6widgets6")

    # RPM package configuration
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_GROUP "Applications/File")
    set(CPACK_RPM_PACKAGE_REQUIRES "qt6-qtbase >= 6.0")

    message(STATUS "Linux packaging configured: DEB, RPM, TGZ installers")
endif()

# CPack configuration
set(CPACK_PACKAGE_NAME "CloneClean")
set(CPACK_PACKAGE_VENDOR "CloneClean Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Cross-platform duplicate file finder")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "CloneClean")

set(CLONECLEAN_PACKAGE_SUFFIX "" CACHE STRING "Optional suffix appended to generated package filenames")
set(_cloneclean_package_suffix "${CLONECLEAN_PACKAGE_SUFFIX}")
if(_cloneclean_package_suffix)
    string(REGEX REPLACE "[^A-Za-z0-9_\-]" "-" _cloneclean_package_suffix "${_cloneclean_package_suffix}")
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-${PLATFORM_NAME}-${_cloneclean_package_suffix}")
else()
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-${PLATFORM_NAME}")
endif()

include(CPack)

# Custom target to copy installers to dist folder
if(WIN32)
    # Windows: Copy NSIS installer or ZIP to dist folder
    # Use Win64 for Windows to match build system structure
    set(_dist_dir "${CMAKE_SOURCE_DIR}/dist/Win64/${CMAKE_BUILD_TYPE}")
    if(NSIS_EXECUTABLE)
        set(_package_ext "exe")
    else()
        set(_package_ext "zip")
    endif()
    add_custom_target(copy_installer
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_dist_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/${CPACK_PACKAGE_FILE_NAME}.${_package_ext}"
            "${_dist_dir}/${CPACK_PACKAGE_FILE_NAME}.${_package_ext}"
        DEPENDS package
        COMMENT "Copying Windows installer to dist/Win64/${CMAKE_BUILD_TYPE}/"
        VERBATIM
    )
    message(STATUS "Installer will be copied to: ${_dist_dir}")
elseif(APPLE)
    # macOS: Copy DMG to dist folder
    # Detect architecture for proper dist folder structure
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        set(_macos_arch_dir "ARM")
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
        set(_macos_arch_dir "X64")
    else()
        # Fallback to system processor
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(_macos_arch_dir "ARM")
        else()
            set(_macos_arch_dir "X64")
        endif()
    endif()
    set(_dist_dir "${CMAKE_SOURCE_DIR}/dist/MacOS/${_macos_arch_dir}/${CMAKE_BUILD_TYPE}")
    add_custom_target(copy_installer
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_dist_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/${CPACK_PACKAGE_FILE_NAME}.dmg"
            "${_dist_dir}/${CPACK_PACKAGE_FILE_NAME}.dmg"
        DEPENDS package
        COMMENT "Copying macOS DMG to dist/MacOS/${_macos_arch_dir}/${CMAKE_BUILD_TYPE}/"
        VERBATIM
    )
    message(STATUS "DMG installer will be copied to: ${_dist_dir}")
elseif(UNIX)
    # Linux: Copy DEB, RPM, and TGZ to dist folder
    set(_dist_dir "${CMAKE_SOURCE_DIR}/dist/Linux/${CMAKE_BUILD_TYPE}")
    add_custom_target(copy_installer
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_dist_dir}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/${CPACK_PACKAGE_FILE_NAME}.deb"
            "${_dist_dir}/${CPACK_PACKAGE_FILE_NAME}.deb"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/${CPACK_PACKAGE_FILE_NAME}.rpm"
            "${_dist_dir}/${CPACK_PACKAGE_FILE_NAME}.rpm"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/${CPACK_PACKAGE_FILE_NAME}.tar.gz"
            "${_dist_dir}/${CPACK_PACKAGE_FILE_NAME}.tar.gz"
        DEPENDS package
        COMMENT "Copying Linux packages to dist/Linux/${CMAKE_BUILD_TYPE}/"
        VERBATIM
    )
    message(STATUS "Linux packages will be copied to: ${_dist_dir}")
endif()

# Testing configuration
# enable_testing()

# Add subdirectories for tests
# add_subdirectory(tests)

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
                $<TARGET_FILE:cloneclean>
            DEPENDS cloneclean
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
    COMMAND ${CMAKE_COMMAND} -E echo "  cloneclean      - Main application"
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
message(STATUS "CloneClean Build Configuration Summary")
message(STATUS "=====================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${PLATFORM_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Qt${QT_VERSION} Version: ${Qt${QT_VERSION}_VERSION}")
message(STATUS "Available Build Targets:")
message(STATUS "  - cloneclean (main application)")
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
