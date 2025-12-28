# Hands-on Lab 1: Your First Test

**Duration**: 45-60 minutes | **Level**: Beginner | **Module**: Testing Fundamentals

## Lab Overview

In this hands-on lab, you'll write your very first test for a CloneClean component. You'll learn the basic structure of a test, how to set up test data, and how to verify expected behavior.

## Learning Objectives

By completing this lab, you will:
- Write a complete test class using Qt Test Framework
- Understand test setup and teardown
- Create and manage test data
- Use basic Qt Test assertions
- Run and interpret test results

## Prerequisites

- Completed Module 1: Testing Fundamentals
- Basic C++ knowledge
- Qt development environment set up
- CloneClean project accessible

## Lab Setup

### Step 1: Create Lab Directory
```bash
cd cloneclean/tests
mkdir lab01
cd lab01
```

### Step 2: Verify Environment
```bash
# Ensure Qt Test is available
qmake -query QT_VERSION
# Should show Qt 6.x.x

# Ensure you can compile Qt Test programs
echo '#include <QtTest>' > test_compile.cpp
echo 'int main() { return 0; }' >> test_compile.cpp
g++ -I$(qmake -query QT_INSTALL_HEADERS) -c test_compile.cpp
rm test_compile.cpp test_compile.o
```

## Lab Exercises

### Exercise 1: Simple Calculator Test (20 minutes)

We'll start by testing a simple calculator class to learn the basics.

#### Step 1.1: Create the Calculator Class

Create `simple_calculator.h`:
```cpp
#ifndef SIMPLE_CALCULATOR_H
#define SIMPLE_CALCULATOR_H

class SimpleCalculator {
public:
    SimpleCalculator() = default;
    
    int add(int a, int b) const;
    int subtract(int a, int b) const;
    int multiply(int a, int b) const;
    double divide(int a, int b) const;
    
    bool isLastOperationValid() const { return m_lastOperationValid; }
    
private:
    mutable bool m_lastOperationValid = true;
};

#endif // SIMPLE_CALCULATOR_H
```

Create `simple_calculator.cpp`:
```cpp
#include "simple_calculator.h"
#include <stdexcept>

int SimpleCalculator::add(int a, int b) const {
    m_lastOperationValid = true;
    return a + b;
}

int SimpleCalculator::subtract(int a, int b) const {
    m_lastOperationValid = true;
    return a - b;
}

int SimpleCalculator::multiply(int a, int b) const {
    m_lastOperationValid = true;
    return a * b;
}

double SimpleCalculator::divide(int a, int b) const {
    if (b == 0) {
        m_lastOperationValid = false;
        return 0.0;
    }
    m_lastOperationValid = true;
    return static_cast<double>(a) / b;
}
```

#### Step 1.2: Write Your First Test

Create `test_simple_calculator.cpp`:
```cpp
#include <QtTest>
#include "simple_calculator.h"

class TestSimpleCalculator : public QObject {
    Q_OBJECT

private slots:
    // Test lifecycle methods
    void initTestCase();
    void init();
    void cleanup();
    void cleanupTestCase();
    
    // Test methods
    void testAdd_WhenAddingTwoPositiveNumbers_ReturnsSum();
    void testSubtract_WhenSubtractingSmaller_ReturnsPositive();
    void testMultiply_WhenMultiplyingByZero_ReturnsZero();
    void testDivide_WhenDividingByNonZero_ReturnsQuotient();
    void testDivide_WhenDividingByZero_SetsInvalidFlag();

private:
    SimpleCalculator* m_calculator;
};

void TestSimpleCalculator::initTestCase() {
    // This runs once before all tests
    qDebug() << "Starting SimpleCalculator test suite";
}

void TestSimpleCalculator::init() {
    // This runs before each test method
    m_calculator = new SimpleCalculator();
}

void TestSimpleCalculator::cleanup() {
    // This runs after each test method
    delete m_calculator;
    m_calculator = nullptr;
}

void TestSimpleCalculator::cleanupTestCase() {
    // This runs once after all tests
    qDebug() << "Completed SimpleCalculator test suite";
}

void TestSimpleCalculator::testAdd_WhenAddingTwoPositiveNumbers_ReturnsSum() {
    // Arrange
    int a = 5;
    int b = 3;
    int expected = 8;
    
    // Act
    int result = m_calculator->add(a, b);
    
    // Assert
    QCOMPARE(result, expected);
    QVERIFY(m_calculator->isLastOperationValid());
}

void TestSimpleCalculator::testSubtract_WhenSubtractingSmaller_ReturnsPositive() {
    // Arrange
    int a = 10;
    int b = 4;
    int expected = 6;
    
    // Act
    int result = m_calculator->subtract(a, b);
    
    // Assert
    QCOMPARE(result, expected);
    QVERIFY(m_calculator->isLastOperationValid());
}

void TestSimpleCalculator::testMultiply_WhenMultiplyingByZero_ReturnsZero() {
    // Arrange
    int a = 42;
    int b = 0;
    int expected = 0;
    
    // Act
    int result = m_calculator->multiply(a, b);
    
    // Assert
    QCOMPARE(result, expected);
    QVERIFY(m_calculator->isLastOperationValid());
}

void TestSimpleCalculator::testDivide_WhenDividingByNonZero_ReturnsQuotient() {
    // Arrange
    int a = 15;
    int b = 3;
    double expected = 5.0;
    
    // Act
    double result = m_calculator->divide(a, b);
    
    // Assert
    QCOMPARE(result, expected);
    QVERIFY(m_calculator->isLastOperationValid());
}

void TestSimpleCalculator::testDivide_WhenDividingByZero_SetsInvalidFlag() {
    // Arrange
    int a = 10;
    int b = 0;
    
    // Act
    double result = m_calculator->divide(a, b);
    
    // Assert
    QCOMPARE(result, 0.0);
    QVERIFY(!m_calculator->isLastOperationValid());
}

QTEST_MAIN(TestSimpleCalculator)
#include "test_simple_calculator.moc"
```

#### Step 1.3: Compile and Run

Create `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.16)
project(Lab01Tests)

find_package(Qt6 REQUIRED COMPONENTS Core Test)

set(CMAKE_AUTOMOC ON)

add_executable(test_simple_calculator
    simple_calculator.cpp
    test_simple_calculator.cpp
)

target_link_libraries(test_simple_calculator
    Qt6::Core
    Qt6::Test
)

add_test(NAME SimpleCalculatorTest COMMAND test_simple_calculator)
```

Compile and run:
```bash
mkdir build && cd build
cmake ..
make
./test_simple_calculator
```

**Expected Output**:
```
********* Start testing of TestSimpleCalculator *********
Config: Using QtTest library 6.x.x
PASS   : TestSimpleCalculator::initTestCase()
PASS   : TestSimpleCalculator::testAdd_WhenAddingTwoPositiveNumbers_ReturnsSum()
PASS   : TestSimpleCalculator::testSubtract_WhenSubtractingSmaller_ReturnsPositive()
PASS   : TestSimpleCalculator::testMultiply_WhenMultiplyingByZero_ReturnsZero()
PASS   : TestSimpleCalculator::testDivide_WhenDividingByNonZero_ReturnsQuotient()
PASS   : TestSimpleCalculator::testDivide_WhenDividingByZero_SetsInvalidFlag()
PASS   : TestSimpleCalculator::cleanupTestCase()
Totals: 6 passed, 0 failed, 0 skipped, 0 blacklisted, Xms
********* Finished testing of TestSimpleCalculator *********
```

### Exercise 2: File Utility Test (25 minutes)

Now let's test something more relevant to CloneClean - a file utility class.

#### Step 2.1: Create File Utility Class

Create `file_utility.h`:
```cpp
#ifndef FILE_UTILITY_H
#define FILE_UTILITY_H

#include <QString>
#include <QStringList>

class FileUtility {
public:
    FileUtility() = default;
    
    // File operations
    bool fileExists(const QString& path) const;
    qint64 getFileSize(const QString& path) const;
    QString getFileExtension(const QString& path) const;
    
    // Directory operations
    QStringList listFiles(const QString& directory, const QStringList& filters = {}) const;
    int countFiles(const QString& directory, bool recursive = false) const;
    
    // Utility functions
    QString formatFileSize(qint64 bytes) const;
    bool isImageFile(const QString& path) const;
    
private:
    QStringList m_imageExtensions = {"jpg", "jpeg", "png", "gif", "bmp", "tiff"};
};

#endif // FILE_UTILITY_H
```

Create `file_utility.cpp`:
```cpp
#include "file_utility.h"
#include <QFile>
#include <QFileInfo>
#include <QDir>

bool FileUtility::fileExists(const QString& path) const {
    return QFile::exists(path);
}

qint64 FileUtility::getFileSize(const QString& path) const {
    QFileInfo info(path);
    return info.exists() ? info.size() : -1;
}

QString FileUtility::getFileExtension(const QString& path) const {
    QFileInfo info(path);
    return info.suffix().toLower();
}

QStringList FileUtility::listFiles(const QString& directory, const QStringList& filters) const {
    QDir dir(directory);
    if (!dir.exists()) {
        return {};
    }
    
    QDir::Filters dirFilters = QDir::Files | QDir::NoDotAndDotDot;
    return dir.entryList(filters.isEmpty() ? QStringList("*") : filters, dirFilters);
}

int FileUtility::countFiles(const QString& directory, bool recursive) const {
    QDir dir(directory);
    if (!dir.exists()) {
        return -1;
    }
    
    QDir::Filters filters = QDir::Files | QDir::NoDotAndDotDot;
    if (recursive) {
        filters |= QDir::AllDirs;
    }
    
    int count = 0;
    QStringList entries = dir.entryList(filters);
    
    for (const QString& entry : entries) {
        QString fullPath = dir.absoluteFilePath(entry);
        QFileInfo info(fullPath);
        
        if (info.isFile()) {
            count++;
        } else if (recursive && info.isDir()) {
            count += countFiles(fullPath, true);
        }
    }
    
    return count;
}

QString FileUtility::formatFileSize(qint64 bytes) const {
    if (bytes < 0) {
        return "Unknown";
    }
    
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    
    if (bytes >= GB) {
        return QString::number(bytes / GB) + " GB";
    } else if (bytes >= MB) {
        return QString::number(bytes / MB) + " MB";
    } else if (bytes >= KB) {
        return QString::number(bytes / KB) + " KB";
    } else {
        return QString::number(bytes) + " bytes";
    }
}

bool FileUtility::isImageFile(const QString& path) const {
    QString extension = getFileExtension(path);
    return m_imageExtensions.contains(extension);
}
```

#### Step 2.2: Write Comprehensive Tests

Create `test_file_utility.cpp`:
```cpp
#include <QtTest>
#include <QTemporaryDir>
#include <QFile>
#include <QTextStream>
#include "file_utility.h"

class TestFileUtility : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void init();
    void cleanup();
    void cleanupTestCase();
    
    // File existence tests
    void testFileExists_WhenFileExists_ReturnsTrue();
    void testFileExists_WhenFileDoesNotExist_ReturnsFalse();
    
    // File size tests
    void testGetFileSize_WhenFileExists_ReturnsCorrectSize();
    void testGetFileSize_WhenFileDoesNotExist_ReturnsMinusOne();
    
    // Extension tests
    void testGetFileExtension_WhenFileHasExtension_ReturnsLowerCase();
    void testGetFileExtension_WhenFileHasNoExtension_ReturnsEmpty();
    
    // Directory listing tests
    void testListFiles_WhenDirectoryExists_ReturnsFileList();
    void testListFiles_WhenDirectoryDoesNotExist_ReturnsEmpty();
    void testListFiles_WhenUsingFilters_ReturnsFilteredList();
    
    // File counting tests
    void testCountFiles_WhenDirectoryExists_ReturnsCorrectCount();
    void testCountFiles_WhenRecursive_CountsSubdirectories();
    
    // Utility function tests
    void testFormatFileSize_WithVariousSizes_ReturnsFormattedString();
    void testIsImageFile_WithImageExtensions_ReturnsTrue();
    void testIsImageFile_WithNonImageExtensions_ReturnsFalse();

private:
    void createTestFile(const QString& filename, const QString& content = "test content");
    void createTestDirectory(const QString& dirname);
    
    FileUtility* m_utility;
    QTemporaryDir* m_tempDir;
    QString m_testDirPath;
};

void TestFileUtility::initTestCase() {
    qDebug() << "Starting FileUtility test suite";
}

void TestFileUtility::init() {
    m_utility = new FileUtility();
    m_tempDir = new QTemporaryDir();
    QVERIFY(m_tempDir->isValid());
    m_testDirPath = m_tempDir->path();
}

void TestFileUtility::cleanup() {
    delete m_utility;
    m_utility = nullptr;
    delete m_tempDir;
    m_tempDir = nullptr;
}

void TestFileUtility::cleanupTestCase() {
    qDebug() << "Completed FileUtility test suite";
}

void TestFileUtility::testFileExists_WhenFileExists_ReturnsTrue() {
    // Arrange
    QString testFile = m_testDirPath + "/existing_file.txt";
    createTestFile(testFile);
    
    // Act
    bool exists = m_utility->fileExists(testFile);
    
    // Assert
    QVERIFY(exists);
}

void TestFileUtility::testFileExists_WhenFileDoesNotExist_ReturnsFalse() {
    // Arrange
    QString nonExistentFile = m_testDirPath + "/non_existent.txt";
    
    // Act
    bool exists = m_utility->fileExists(nonExistentFile);
    
    // Assert
    QVERIFY(!exists);
}

void TestFileUtility::testGetFileSize_WhenFileExists_ReturnsCorrectSize() {
    // Arrange
    QString testFile = m_testDirPath + "/size_test.txt";
    QString content = "Hello, World!"; // 13 bytes
    createTestFile(testFile, content);
    
    // Act
    qint64 size = m_utility->getFileSize(testFile);
    
    // Assert
    QCOMPARE(size, static_cast<qint64>(content.length()));
}

void TestFileUtility::testGetFileSize_WhenFileDoesNotExist_ReturnsMinusOne() {
    // Arrange
    QString nonExistentFile = m_testDirPath + "/non_existent.txt";
    
    // Act
    qint64 size = m_utility->getFileSize(nonExistentFile);
    
    // Assert
    QCOMPARE(size, static_cast<qint64>(-1));
}

void TestFileUtility::testGetFileExtension_WhenFileHasExtension_ReturnsLowerCase() {
    // Arrange & Act & Assert
    QCOMPARE(m_utility->getFileExtension("test.TXT"), QString("txt"));
    QCOMPARE(m_utility->getFileExtension("image.JPEG"), QString("jpeg"));
    QCOMPARE(m_utility->getFileExtension("document.PDF"), QString("pdf"));
}

void TestFileUtility::testGetFileExtension_WhenFileHasNoExtension_ReturnsEmpty() {
    // Arrange & Act & Assert
    QCOMPARE(m_utility->getFileExtension("filename"), QString(""));
    QCOMPARE(m_utility->getFileExtension("path/to/file"), QString(""));
}

void TestFileUtility::testListFiles_WhenDirectoryExists_ReturnsFileList() {
    // Arrange
    createTestFile(m_testDirPath + "/file1.txt");
    createTestFile(m_testDirPath + "/file2.txt");
    createTestFile(m_testDirPath + "/file3.doc");
    
    // Act
    QStringList files = m_utility->listFiles(m_testDirPath);
    
    // Assert
    QCOMPARE(files.size(), 3);
    QVERIFY(files.contains("file1.txt"));
    QVERIFY(files.contains("file2.txt"));
    QVERIFY(files.contains("file3.doc"));
}

void TestFileUtility::testListFiles_WhenDirectoryDoesNotExist_ReturnsEmpty() {
    // Arrange
    QString nonExistentDir = m_testDirPath + "/non_existent";
    
    // Act
    QStringList files = m_utility->listFiles(nonExistentDir);
    
    // Assert
    QVERIFY(files.isEmpty());
}

void TestFileUtility::testListFiles_WhenUsingFilters_ReturnsFilteredList() {
    // Arrange
    createTestFile(m_testDirPath + "/file1.txt");
    createTestFile(m_testDirPath + "/file2.txt");
    createTestFile(m_testDirPath + "/file3.doc");
    
    // Act
    QStringList txtFiles = m_utility->listFiles(m_testDirPath, {"*.txt"});
    
    // Assert
    QCOMPARE(txtFiles.size(), 2);
    QVERIFY(txtFiles.contains("file1.txt"));
    QVERIFY(txtFiles.contains("file2.txt"));
    QVERIFY(!txtFiles.contains("file3.doc"));
}

void TestFileUtility::testCountFiles_WhenDirectoryExists_ReturnsCorrectCount() {
    // Arrange
    createTestFile(m_testDirPath + "/file1.txt");
    createTestFile(m_testDirPath + "/file2.txt");
    createTestFile(m_testDirPath + "/file3.doc");
    
    // Act
    int count = m_utility->countFiles(m_testDirPath);
    
    // Assert
    QCOMPARE(count, 3);
}

void TestFileUtility::testCountFiles_WhenRecursive_CountsSubdirectories() {
    // Arrange
    createTestFile(m_testDirPath + "/file1.txt");
    createTestDirectory(m_testDirPath + "/subdir");
    createTestFile(m_testDirPath + "/subdir/file2.txt");
    createTestFile(m_testDirPath + "/subdir/file3.txt");
    
    // Act
    int count = m_utility->countFiles(m_testDirPath, true);
    
    // Assert
    QCOMPARE(count, 3); // 1 in root + 2 in subdir
}

void TestFileUtility::testFormatFileSize_WithVariousSizes_ReturnsFormattedString() {
    // Test various file sizes
    QCOMPARE(m_utility->formatFileSize(512), QString("512 bytes"));
    QCOMPARE(m_utility->formatFileSize(1024), QString("1 KB"));
    QCOMPARE(m_utility->formatFileSize(1024 * 1024), QString("1 MB"));
    QCOMPARE(m_utility->formatFileSize(1024LL * 1024 * 1024), QString("1 GB"));
    QCOMPARE(m_utility->formatFileSize(-1), QString("Unknown"));
}

void TestFileUtility::testIsImageFile_WithImageExtensions_ReturnsTrue() {
    // Test various image extensions
    QVERIFY(m_utility->isImageFile("photo.jpg"));
    QVERIFY(m_utility->isImageFile("image.PNG"));
    QVERIFY(m_utility->isImageFile("graphic.gif"));
    QVERIFY(m_utility->isImageFile("bitmap.BMP"));
}

void TestFileUtility::testIsImageFile_WithNonImageExtensions_ReturnsFalse() {
    // Test non-image extensions
    QVERIFY(!m_utility->isImageFile("document.txt"));
    QVERIFY(!m_utility->isImageFile("video.mp4"));
    QVERIFY(!m_utility->isImageFile("archive.zip"));
    QVERIFY(!m_utility->isImageFile("noextension"));
}

// Helper method implementations
void TestFileUtility::createTestFile(const QString& filename, const QString& content) {
    QFile file(filename);
    QVERIFY(file.open(QIODevice::WriteOnly));
    QTextStream stream(&file);
    stream << content;
    file.close();
    QVERIFY(file.exists());
}

void TestFileUtility::createTestDirectory(const QString& dirname) {
    QDir dir;
    QVERIFY(dir.mkpath(dirname));
}

QTEST_MAIN(TestFileUtility)
#include "test_file_utility.moc"
```

#### Step 2.3: Update CMakeLists.txt and Run

Update `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.16)
project(Lab01Tests)

find_package(Qt6 REQUIRED COMPONENTS Core Test)

set(CMAKE_AUTOMOC ON)

# Simple Calculator Test
add_executable(test_simple_calculator
    simple_calculator.cpp
    test_simple_calculator.cpp
)

target_link_libraries(test_simple_calculator
    Qt6::Core
    Qt6::Test
)

# File Utility Test
add_executable(test_file_utility
    file_utility.cpp
    test_file_utility.cpp
)

target_link_libraries(test_file_utility
    Qt6::Core
    Qt6::Test
)

# Add tests to CTest
add_test(NAME SimpleCalculatorTest COMMAND test_simple_calculator)
add_test(NAME FileUtilityTest COMMAND test_file_utility)
```

Compile and run:
```bash
cd build
make
./test_file_utility
```

## Lab Reflection

### Questions to Consider

1. **Test Structure**: How did the test lifecycle methods (init, cleanup) help with test isolation?

2. **Assertions**: Which Qt Test assertions did you find most useful? When would you use `QVERIFY` vs `QCOMPARE`?

3. **Test Data**: How did creating temporary files and directories make the tests more reliable?

4. **Test Naming**: How do the descriptive test method names help with understanding test failures?

5. **Edge Cases**: What additional edge cases could you test for the FileUtility class?

### Common Issues and Solutions

#### Issue: Tests fail due to file permissions
**Solution**: Ensure temporary directories have proper permissions, use QTemporaryDir

#### Issue: Tests are flaky (sometimes pass, sometimes fail)
**Solution**: Ensure proper cleanup, avoid hardcoded paths, use proper synchronization

#### Issue: Test output is hard to understand
**Solution**: Use descriptive test names, add debug output, use QVERIFY2 with custom messages

## Lab Summary

Congratulations! You've successfully:
- ✅ Written your first Qt Test Framework test
- ✅ Used test lifecycle methods for setup and cleanup
- ✅ Created and managed test data safely
- ✅ Applied various Qt Test assertions
- ✅ Tested both simple and complex functionality
- ✅ Handled file system operations in tests

### Key Learnings

1. **Test Structure**: Proper setup and cleanup ensure test isolation
2. **Assertions**: Choose the right assertion for clear failure messages
3. **Test Data**: Use temporary resources to avoid side effects
4. **Naming**: Descriptive names make tests self-documenting
5. **Coverage**: Test both happy path and edge cases

## Next Steps

1. **Practice**: Try writing tests for other simple classes
2. **Explore**: Look at existing CloneClean tests for more examples
3. **Continue**: Proceed to the next module or lab
4. **Share**: Discuss your experience with other learners

## Additional Challenges

If you want to practice more:

1. **Add More Tests**: Add tests for error conditions in FileUtility
2. **Parameterized Tests**: Research Qt Test data-driven testing
3. **Performance Tests**: Add timing measurements to your tests
4. **Mock Objects**: Research how to create mock objects for testing

---

**Lab 1 Complete!** 🎉

You've written your first tests and learned the fundamentals of the Qt Test Framework. You're now ready to tackle more complex testing scenarios.

*Time to complete: 45-60 minutes*
*Next: [Hands-on Lab 2: Integration Testing](hands-on-lab-02.md)*