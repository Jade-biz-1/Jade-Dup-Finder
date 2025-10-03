# DupFinder Development Workflow Guide

**Version:** 1.0  
**Created:** 2025-10-03  
**Team:** DupFinder Development Team  

---

## Overview

This document establishes the development workflow, coding standards, review processes, and best practices for the DupFinder project. Following these guidelines ensures code consistency, quality, and maintainability across the entire development lifecycle.

### Workflow Principles
- **Quality First:** Code quality and safety over speed
- **Collaborative Development:** Peer review and knowledge sharing
- **Continuous Integration:** Automated testing and validation
- **Documentation:** Clear and comprehensive documentation
- **Incremental Delivery:** Small, focused commits and regular releases

---

## Git Workflow

### 1. Branch Strategy

We use a **Git Flow** based branching model with the following branches:

#### Main Branches
- **`main`**: Production-ready code, always stable
- **`develop`**: Integration branch for ongoing development

#### Feature Branches
- **`feature/feature-name`**: New features and enhancements
- **`bugfix/bug-description`**: Bug fixes for develop branch
- **`hotfix/critical-issue`**: Emergency fixes for production
- **`release/version-number`**: Release preparation and final testing

#### Examples
```bash
# Feature branch
feature/file-scanner-implementation
feature/gui-results-widget
feature/windows-platform-support

# Bug fix branch
bugfix/memory-leak-hash-calculator
bugfix/ui-responsive-during-scan

# Release branch
release/1.0.0-beta
release/1.1.0

# Hotfix branch
hotfix/critical-crash-large-files
```

### 2. Commit Standards

#### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code formatting (no logic changes)
- **refactor**: Code refactoring (no functional changes)
- **test**: Adding or modifying tests
- **chore**: Build system, dependencies, tooling

#### Examples
```bash
feat(core): implement basic file scanner with progress reporting

- Add recursive directory traversal using QDirIterator
- Implement configurable file filtering by size and patterns
- Add progress signals for UI integration
- Include comprehensive error handling for permissions

Closes #15

fix(gui): resolve memory leak in results widget

The duplicate group display was not properly releasing memory
when clearing results. Added explicit cleanup in destructor
and clear methods.

Fixes #42

docs(api): add comprehensive API documentation for hash calculator

- Document all public methods and signals
- Add usage examples and best practices
- Include performance considerations
- Update architecture diagrams

test(core): add unit tests for duplicate detection algorithms

- Test size-based pre-filtering
- Test hash-based duplicate grouping
- Test recommendation algorithm
- Achieve 95% code coverage

Refs #28
```

### 3. Pull Request Process

#### Before Creating PR
1. **Ensure tests pass**: `make check` or `ninja check`
2. **Format code**: `make format` or `ninja format`
3. **Update documentation**: Update relevant docs if needed
4. **Rebase on latest develop**: `git rebase develop`

#### PR Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Screenshots (if applicable)
Include screenshots for UI changes.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is documented
- [ ] Tests added/updated
- [ ] No breaking changes (or properly documented)

## Related Issues
Closes #issue_number
```

#### Review Requirements
- **Minimum 1 reviewer** for feature branches
- **Minimum 2 reviewers** for critical/complex changes
- **All tests must pass** before merge
- **No merge conflicts** with target branch

---

## Coding Standards

### 1. C++ Style Guide

#### General Principles
- Follow **C++17** best practices
- Use **RAII** (Resource Acquisition Is Initialization)
- Prefer **standard library** over custom implementations
- Use **smart pointers** for memory management
- Follow **const-correctness**

#### Naming Conventions

```cpp
// Classes: PascalCase
class FileScanner {
public:
    // Public methods: camelCase
    void startScan(const ScanOptions& options);
    bool isScanning() const;
    
    // Public member variables: camelCase (rare)
    int publicVariable;
    
private:
    // Private methods: camelCase
    void processFile(const QString& filePath);
    
    // Private member variables: m_ prefix + camelCase
    bool m_isScanning;
    QList<FileInfo> m_scannedFiles;
    std::unique_ptr<ScanWorker> m_worker;
};

// Enums: PascalCase for enum, PascalCase for values
enum class DetectionMode {
    Quick,
    Deep,
    Smart
};

// Constants: ALL_CAPS with underscores
constexpr int MAX_THREAD_COUNT = 16;
constexpr const char* DEFAULT_CONFIG_FILE = "dupfinder.conf";

// Functions: camelCase
bool validateFilePath(const QString& path);
QStringList getSystemPaths();

// Namespaces: lowercase
namespace dupfinder {
namespace utils {
    // ...
}
}
```

#### File Organization

```cpp
#pragma once  // Use #pragma once instead of include guards

// System includes first
#include <memory>
#include <chrono>
#include <vector>

// Qt includes second  
#include <QObject>
#include <QString>
#include <QFileInfo>

// Local includes last
#include "base_scanner.h"
#include "file_info.h"

/**
 * @brief FileScanner handles recursive directory scanning
 * 
 * This class provides functionality for scanning directories
 * and identifying files based on configurable criteria.
 * 
 * @since 1.0.0
 */
class FileScanner : public QObject {
    Q_OBJECT
    
public:
    /**
     * @brief Construct a new FileScanner
     * @param parent Parent QObject for memory management
     */
    explicit FileScanner(QObject* parent = nullptr);
    
    /**
     * @brief Destroy the FileScanner and cleanup resources
     */
    ~FileScanner() override;
    
    // ... rest of class
    
private:
    class FileScannerPrivate;
    std::unique_ptr<FileScannerPrivate> d;  // Pimpl idiom for ABI stability
};
```

#### Qt-Specific Guidelines

```cpp
class MyWidget : public QWidget {
    Q_OBJECT
    
public:
    explicit MyWidget(QWidget* parent = nullptr);
    
    // Use Qt types for Qt-related functionality
    void setFilePath(const QString& path);  // Not std::string
    QSize minimumSizeHint() const override;
    
public slots:
    void onFileSelected();
    
signals:
    void fileChanged(const QString& newPath);
    
private slots:
    void handleTimeout();
    
private:
    // Use Qt containers for Qt objects
    QList<QLabel*> m_labels;
    QHash<QString, QVariant> m_settings;
    
    // Use standard containers for non-Qt data
    std::vector<int> m_indices;
    std::unordered_map<std::string, double> m_metrics;
};
```

### 2. Code Quality Standards

#### Documentation
```cpp
/**
 * @brief Calculate SHA-256 hash for a file
 * 
 * This method calculates the SHA-256 hash of a file in chunks
 * to handle large files efficiently. Progress is reported through
 * the fileProgress signal for files larger than 100MB.
 * 
 * @param filePath Absolute path to the file to hash
 * @return HashResult containing hash value and metadata
 * @throws HashCalculationException if file cannot be read
 * 
 * @note This method is thread-safe
 * @see HashResult, fileProgress()
 * @since 1.0.0
 */
HashResult calculateHash(const QString& filePath);
```

#### Error Handling
```cpp
// Use exceptions for exceptional conditions
class FileScannerException : public std::exception {
public:
    explicit FileScannerException(const QString& message) 
        : m_message(message.toStdString()) {}
    
    const char* what() const noexcept override {
        return m_message.c_str();
    }
    
private:
    std::string m_message;
};

// Use Qt signals for expected error conditions
void FileScanner::processFile(const QString& filePath) {
    try {
        // File processing logic
        if (!QFile::exists(filePath)) {
            emit scanError(QString("File not found: %1").arg(filePath));
            return;
        }
        
        // ... processing
    } catch (const std::exception& e) {
        emit scanError(QString("Unexpected error: %1").arg(e.what()));
    }
}
```

#### Resource Management
```cpp
// Use RAII and smart pointers
class FileProcessor {
public:
    FileProcessor(const QString& filePath) 
        : m_file(std::make_unique<QFile>(filePath)) {
        if (!m_file->open(QIODevice::ReadOnly)) {
            throw FileAccessException("Cannot open file");
        }
    }
    
    // No need for explicit destructor - smart pointer handles cleanup
    
private:
    std::unique_ptr<QFile> m_file;
    // QFile destructor will be called automatically
};

// For Qt objects with parent-child relationships
void MainWindow::createWidgets() {
    // Parent will handle cleanup
    auto layout = new QVBoxLayout(this);
    auto button = new QPushButton("Scan", this);
    
    layout->addWidget(button);
}
```

### 3. Testing Standards

#### Unit Test Structure
```cpp
#include <QtTest>
#include "file_scanner.h"
#include "test_helpers.h"

class FileScannerTest : public QObject {
    Q_OBJECT
    
private slots:
    void initTestCase();      // Run once before all tests
    void init();             // Run before each test
    void cleanup();          // Run after each test
    void cleanupTestCase();  // Run once after all tests
    
    // Test methods
    void testBasicScanning();
    void testFileFiltering();
    void testErrorHandling();
    void testProgressReporting();
    
private:
    QString m_testDataPath;
    QTemporaryDir m_tempDir;
    std::unique_ptr<FileScanner> m_scanner;
};

void FileScannerTest::initTestCase() {
    m_testDataPath = m_tempDir.path();
    QVERIFY(m_tempDir.isValid());
    
    // Create test data structure
    createTestFiles();
}

void FileScannerTest::init() {
    m_scanner = std::make_unique<FileScanner>();
}

void FileScannerTest::testBasicScanning() {
    // Given
    FileScanner::ScanOptions options;
    options.targetPaths << m_testDataPath;
    
    QSignalSpy scanCompletedSpy(m_scanner.get(), &FileScanner::scanCompleted);
    QSignalSpy fileFoundSpy(m_scanner.get(), &FileScanner::fileFound);
    
    // When
    m_scanner->startScan(options);
    
    // Then
    QVERIFY(scanCompletedSpy.wait(5000));  // Wait up to 5 seconds
    QCOMPARE(scanCompletedSpy.count(), 1);
    QVERIFY(fileFoundSpy.count() > 0);
    
    auto files = m_scanner->getScannedFiles();
    QVERIFY(!files.isEmpty());
}

// Test runner
QTEST_MAIN(FileScannerTest)
#include "file_scanner_test.moc"
```

#### Test Organization
```
tests/
├── unit/
│   ├── core/
│   │   ├── test_file_scanner.cpp
│   │   ├── test_hash_calculator.cpp
│   │   └── test_duplicate_detector.cpp
│   ├── gui/
│   │   ├── test_main_window.cpp
│   │   └── test_results_widget.cpp
│   └── platform/
│       ├── test_linux_file_ops.cpp
│       └── test_platform_factory.cpp
├── integration/
│   ├── test_full_workflow.cpp
│   └── test_cross_component.cpp
├── performance/
│   ├── test_large_datasets.cpp
│   └── benchmark_algorithms.cpp
└── helpers/
    ├── test_helpers.h
    ├── mock_file_system.h
    └── test_data_generator.h
```

---

## Development Environment

### 1. Required Tools

#### Essential Tools
- **C++ Compiler**: GCC 9+ or Clang 10+ or MSVC 2019+
- **Build System**: CMake 3.20+
- **Qt Framework**: Qt 6.4 LTS or later
- **Version Control**: Git 2.25+
- **IDE/Editor**: Qt Creator, Visual Studio, CLion, or VS Code

#### Code Quality Tools
- **clang-format**: Code formatting
- **clang-tidy**: Static analysis
- **cppcheck**: Additional static analysis
- **valgrind**: Memory leak detection (Linux/macOS)
- **AddressSanitizer**: Memory error detection

#### Installation Commands
```bash
# Ubuntu/Debian
sudo apt install build-essential cmake git qt6-base-dev qt6-tools-dev
sudo apt install clang-format clang-tidy cppcheck valgrind

# macOS (Homebrew)
brew install cmake git qt6
brew install clang-format

# Windows (vcpkg)
vcpkg install qt6[core,widgets,concurrent,network]:x64-windows
```

### 2. IDE Configuration

#### Qt Creator Configuration
1. **Code Style**: Import `.clang-format` settings
2. **Build Settings**: Configure Debug and Release builds
3. **Run Settings**: Set up application and test runners
4. **Plugins**: Enable Git, CMake, and testing plugins

#### Visual Studio Code Configuration
```json
// .vscode/settings.json
{
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "C_Cpp.default.cppStandard": "c++17",
    "cmake.configureArgs": [
        "-DCMAKE_BUILD_TYPE=Debug"
    ],
    "files.associations": {
        "*.qrc": "xml",
        "*.ui": "xml"
    },
    "[cpp]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-vscode.cpptools"
    }
}

// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build",
            "type": "cmake",
            "command": "build",
            "group": "build",
            "problemMatcher": "$gcc"
        },
        {
            "label": "Test",
            "type": "shell",
            "command": "ctest",
            "args": ["--output-on-failure"],
            "group": "test",
            "dependsOn": "Build"
        }
    ]
}
```

### 3. Pre-commit Hooks

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash

# Format code
echo "Running clang-format..."
make format
if [ $? -ne 0 ]; then
    echo "Code formatting failed!"
    exit 1
fi

# Run static analysis
echo "Running clang-tidy..."
find src -name "*.cpp" -exec clang-tidy {} \;
if [ $? -ne 0 ]; then
    echo "Static analysis found issues!"
    exit 1
fi

# Run tests
echo "Running unit tests..."
make check
if [ $? -ne 0 ]; then
    echo "Tests failed!"
    exit 1
fi

echo "Pre-commit checks passed!"
```

---

## Continuous Integration

### 1. GitHub Actions Workflow

#### Main CI Pipeline
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  build-and-test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        build_type: [Debug, Release]
    
    runs-on: ${{ matrix.os }}
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Qt
      uses: jurplel/install-qt-action@v3
      with:
        version: '6.4.0'
        modules: 'qtcharts qtdatavisualization'
    
    - name: Configure CMake
      run: cmake -B build -DCMAKE_BUILD_TYPE=${{ matrix.build_type }}
    
    - name: Build
      run: cmake --build build --config ${{ matrix.build_type }}
    
    - name: Test
      working-directory: build
      run: ctest --output-on-failure -C ${{ matrix.build_type }}
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: failure()
      with:
        name: test-results-${{ matrix.os }}-${{ matrix.build_type }}
        path: build/Testing/
```

#### Code Quality Pipeline
```yaml
# .github/workflows/code-quality.yml
name: Code Quality

on:
  pull_request:
    branches: [ main, develop ]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install tools
      run: |
        sudo apt update
        sudo apt install clang-format clang-tidy cppcheck
    
    - name: Check formatting
      run: |
        find src include -name "*.cpp" -o -name "*.h" | \
        xargs clang-format --dry-run --Werror
    
    - name: Static analysis
      run: |
        find src -name "*.cpp" | xargs clang-tidy --warnings-as-errors='*'
    
    - name: Security scan
      run: |
        cppcheck --enable=all --error-exitcode=1 src/
```

### 2. Build Validation

#### Automated Checks
- **Compilation**: All platforms and configurations
- **Unit Tests**: 85%+ code coverage required
- **Integration Tests**: Core workflows validated
- **Static Analysis**: Zero warnings policy
- **Memory Leaks**: Valgrind/AddressSanitizer clean
- **Performance**: No regression beyond 5%

#### Quality Gates
```cmake
# CMakeLists.txt - Quality gates
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Enable all warnings
    target_compile_options(dupfinder PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Werror -pedantic>
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Werror -pedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
    )
    
    # Enable sanitizers
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(dupfinder PRIVATE -fsanitize=address)
        target_link_options(dupfinder PRIVATE -fsanitize=address)
    endif()
endif()
```

---

## Release Management

### 1. Version Strategy

#### Semantic Versioning
- **MAJOR.MINOR.PATCH** format (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

#### Version Tags
```bash
# Development versions
1.0.0-dev
1.0.0-alpha.1
1.0.0-beta.2
1.0.0-rc.1

# Release versions
1.0.0
1.0.1
1.1.0
2.0.0
```

### 2. Release Process

#### Release Checklist
```markdown
## Pre-Release (Release Branch)
- [ ] Create release branch: `release/x.y.z`
- [ ] Update version numbers in CMakeLists.txt
- [ ] Update CHANGELOG.md
- [ ] Run full test suite on all platforms
- [ ] Performance regression testing
- [ ] Security review
- [ ] Documentation update
- [ ] Translation update

## Release (Main Branch)
- [ ] Merge release branch to main
- [ ] Create release tag: `git tag -a vx.y.z`
- [ ] Build and sign release packages
- [ ] Upload to distribution platforms
- [ ] Update download links
- [ ] Publish release notes
- [ ] Announce release

## Post-Release
- [ ] Merge main back to develop
- [ ] Close resolved issues
- [ ] Archive release branch
- [ ] Monitor for critical issues
- [ ] Plan next release
```

#### Automated Release
```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Create Release
      uses: actions/create-release@v1
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        body_path: CHANGELOG.md
        draft: false
        prerelease: false
```

---

## Code Review Process

### 1. Review Guidelines

#### What to Review
- **Functionality**: Does it work as intended?
- **Design**: Is the solution well-architected?
- **Code Quality**: Readable, maintainable, follows standards?
- **Performance**: Any performance implications?
- **Security**: Any security vulnerabilities?
- **Tests**: Adequate test coverage?
- **Documentation**: Clear and complete?

#### Review Checklist
```markdown
## Functionality
- [ ] Code implements requirements correctly
- [ ] Edge cases are handled
- [ ] Error conditions are managed properly

## Code Quality
- [ ] Code follows project style guide
- [ ] Variable and function names are clear
- [ ] No code duplication
- [ ] Complex logic is documented

## Architecture
- [ ] Changes fit within existing architecture
- [ ] Dependencies are appropriate
- [ ] Interfaces are well-defined
- [ ] SOLID principles followed

## Performance
- [ ] No obvious performance issues
- [ ] Memory usage is reasonable
- [ ] Database queries are optimized (if applicable)

## Security
- [ ] Input validation is present
- [ ] No sensitive data in logs
- [ ] Authentication/authorization correct

## Tests
- [ ] Unit tests are present and meaningful
- [ ] Test coverage is adequate (85%+)
- [ ] Tests are maintainable

## Documentation
- [ ] Public APIs are documented
- [ ] Complex algorithms explained
- [ ] README updated if needed
```

### 2. Review Etiquette

#### For Authors
- **Small PRs**: Keep changes focused and small
- **Self-Review**: Review your own code first
- **Context**: Provide clear PR description
- **Responsive**: Address feedback promptly
- **Professional**: Accept criticism gracefully

#### For Reviewers
- **Timely**: Review within 24-48 hours
- **Constructive**: Provide helpful feedback
- **Specific**: Point out exact issues
- **Educational**: Explain the "why" behind suggestions
- **Positive**: Acknowledge good practices

#### Comment Examples
```markdown
# Good feedback
"Consider using std::make_unique here instead of new for exception safety."

"This method is doing too many things. Could we split it into smaller functions?"

"Great use of RAII here! This makes the code much safer."

# Poor feedback
"This is wrong."
"Bad code."
"Change this."
```

---

## Performance Guidelines

### 1. Performance Best Practices

#### Algorithm Complexity
```cpp
// Good - O(n) complexity
void FileScanner::groupFilesBySize(const QList<FileInfo>& files) {
    QHash<qint64, QList<FileInfo>> sizeGroups;
    for (const auto& file : files) {
        sizeGroups[file.fileSize].append(file);
    }
}

// Bad - O(n²) complexity
void FileScanner::findDuplicatesNaive(const QList<FileInfo>& files) {
    for (int i = 0; i < files.size(); ++i) {
        for (int j = i + 1; j < files.size(); ++j) {
            if (files[i].fileSize == files[j].fileSize) {
                // Process duplicate
            }
        }
    }
}
```

#### Memory Management
```cpp
// Good - Stream processing for large datasets
void HashCalculator::processLargeFile(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) return;
    
    QCryptographicHash hash(QCryptographicHash::Sha256);
    
    constexpr qint64 CHUNK_SIZE = 64 * 1024;  // 64KB chunks
    while (!file.atEnd()) {
        QByteArray chunk = file.read(CHUNK_SIZE);
        hash.addData(chunk);
        
        // Allow UI updates
        QCoreApplication::processEvents();
    }
    
    emit hashCalculated(hash.result());
}

// Bad - Loading entire file into memory
void HashCalculator::processLargeFileBad(const QString& filePath) {
    QFile file(filePath);
    QByteArray data = file.readAll();  // Could be GBs of data!
    
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(data);
    
    emit hashCalculated(hash.result());
}
```

#### Qt-Specific Optimizations
```cpp
// Good - Reserve capacity when known
void FileScanner::processDirectory(const QString& dirPath) {
    QDir dir(dirPath);
    QFileInfoList entries = dir.entryInfoList();
    
    QList<FileInfo> results;
    results.reserve(entries.size());  // Avoid reallocations
    
    for (const auto& entry : entries) {
        if (entry.isFile()) {
            results.append(convertToFileInfo(entry));
        }
    }
}

// Good - Use const references to avoid copies
void processFiles(const QList<FileInfo>& files) {  // const reference
    for (const auto& file : files) {  // const reference
        processFile(file);
    }
}
```

### 2. Performance Monitoring

#### Benchmarking
```cpp
#include <QtTest>
#include <chrono>

class PerformanceTest : public QObject {
    Q_OBJECT
    
private slots:
    void benchmarkHashCalculation();
    void benchmarkDuplicateDetection();
};

void PerformanceTest::benchmarkHashCalculation() {
    HashCalculator calculator;
    QString testFile = createLargeTestFile(100 * 1024 * 1024);  // 100MB
    
    auto start = std::chrono::high_resolution_clock::now();
    
    QBENCHMARK {
        calculator.calculateHashSync(testFile);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    qDebug() << "Hash calculation took:" << duration.count() << "ms";
    
    // Performance regression test
    QVERIFY(duration.count() < 5000);  // Should complete within 5 seconds
}
```

---

## Security Guidelines

### 1. Secure Coding Practices

#### Input Validation
```cpp
bool FileScanner::isValidPath(const QString& path) {
    // Validate path format
    if (path.isEmpty() || path.length() > MAX_PATH_LENGTH) {
        return false;
    }
    
    // Check for directory traversal attempts
    if (path.contains("..") || path.contains("~")) {
        return false;
    }
    
    // Platform-specific validation
#ifdef Q_OS_WIN
    // Windows path validation
    if (path.contains(QRegularExpression("[<>:\"|?*]"))) {
        return false;
    }
#endif
    
    return QDir(path).exists();
}

// Always validate user input
void FileScanner::startScan(const ScanOptions& options) {
    // Validate all paths before processing
    for (const auto& path : options.targetPaths) {
        if (!isValidPath(path)) {
            emit scanError(tr("Invalid path: %1").arg(path));
            return;
        }
    }
    
    // Proceed with validated input
    performScan(options);
}
```

#### Safe File Operations
```cpp
bool SafetyManager::moveToTrash(const QString& filePath) {
    // Validate file exists and we have permissions
    QFileInfo info(filePath);
    if (!info.exists() || !info.isWritable()) {
        return false;
    }
    
    // Never delete system files
    if (isSystemFile(filePath)) {
        emit protectedFileAccess(filePath, OperationType::MoveToTrash);
        return false;
    }
    
    // Use platform-specific trash mechanism
    try {
        auto platformOps = PlatformFactory::createFileOperations();
        return platformOps->moveToTrash(filePath);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to move file to trash: %s", e.what());
        return false;
    }
}
```

#### Memory Safety
```cpp
// Good - Use smart pointers
class HashCalculator {
private:
    std::unique_ptr<HashWorker> m_worker;
    std::shared_ptr<CacheManager> m_cache;
    
public:
    void setCache(std::shared_ptr<CacheManager> cache) {
        m_cache = std::move(cache);  // Move semantics
    }
};

// Good - Bounds checking
void ResultsWidget::selectFile(int index) {
    if (index < 0 || index >= m_files.size()) {
        LOG_WARNING("Invalid file index: %d", index);
        return;
    }
    
    // Safe to access
    processFile(m_files[index]);
}
```

### 2. Data Privacy

#### Local Processing
```cpp
// Good - All processing stays local
class DuplicateDetector {
public:
    void detectDuplicates(const QList<FileInfo>& files) {
        // All analysis performed locally
        // No data sent to external servers
        
        processFilesLocally(files);
        
        // Results stay on local machine
        emit duplicatesFound(m_results);
    }
    
private:
    void processFilesLocally(const QList<FileInfo>& files) {
        // Implementation that never transmits file data
    }
};
```

#### Secure Temporary Files
```cpp
// Good - Secure temporary file handling
class HashCalculator {
private:
    void createSecureTempFile() {
        QTemporaryFile tempFile;
        tempFile.setAutoRemove(true);  // Automatic cleanup
        
        // Set restrictive permissions (owner only)
        tempFile.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
        
        if (tempFile.open()) {
            // Use temporary file
            processTempFile(tempFile.fileName());
        }
        
        // File automatically removed when object destroyed
    }
};
```

---

## Documentation Standards

### 1. Code Documentation

#### Header Documentation
```cpp
/**
 * @file duplicate_detector.h
 * @brief Core duplicate detection algorithms and data structures
 * @author DupFinder Team
 * @date 2025-10-03
 * @version 1.0.0
 * 
 * This file contains the main duplicate detection engine that analyzes
 * files to identify duplicates using various algorithms including size
 * comparison, hash-based detection, and metadata analysis.
 */

#pragma once

#include <QObject>
#include <QString>

/**
 * @namespace DuplicateUtils
 * @brief Utility functions for duplicate detection operations
 * 
 * This namespace contains helper functions and algorithms used
 * throughout the duplicate detection process.
 */
namespace DuplicateUtils {
    /**
     * @brief Compare two files by their hash values
     * @param file1 First file to compare
     * @param file2 Second file to compare
     * @param hashCache Cache of previously calculated hashes
     * @return true if files have identical hash values
     * 
     * @note This function assumes hash values are already calculated
     * @see HashCalculator::calculateHash()
     * @since 1.0.0
     */
    bool compareByHash(const FileInfo& file1, 
                      const FileInfo& file2,
                      const QHash<QString, QByteArray>& hashCache);
}
```

#### README Structure
```markdown
# DupFinder

Brief description of what the project does.

## Features

- Core functionality list
- Key differentiators
- Platform support

## Quick Start

### Prerequisites
- System requirements
- Dependencies

### Installation
- Step-by-step installation guide
- Platform-specific instructions

### Usage
- Basic usage examples
- Common workflows

## Documentation

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Architecture Design](docs/ARCHITECTURE_DESIGN.md)
- [API Documentation](docs/API_DESIGN.md)
- [Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)

## Contributing

- How to contribute
- Development setup
- Coding standards

## License

License information
```

### 2. Change Documentation

#### CHANGELOG Format
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2025-12-01

### Added
- Advanced duplicate detection algorithms
- Smart file recommendation system
- Batch file operations with progress tracking

### Changed
- Improved scanning performance by 40%
- Updated UI with modern design
- Enhanced error handling and reporting

### Fixed
- Memory leak in hash calculator component
- UI freezing during large file scans
- Incorrect duplicate grouping for edge cases

### Security
- Added input validation for all file paths
- Implemented secure temporary file handling

## [1.0.0] - 2025-10-15

### Added
- Initial release with core functionality
- Cross-platform support (Linux, Windows, macOS)
- Basic duplicate detection using SHA-256 hashes
```

---

This development workflow guide establishes the foundation for consistent, high-quality development of the DupFinder application. Following these guidelines will ensure code maintainability, team collaboration, and successful project delivery.