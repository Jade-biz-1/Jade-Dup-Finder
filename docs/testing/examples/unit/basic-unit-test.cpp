/**
 * @file basic-unit-test.cpp
 * @brief Demonstrates basic unit testing patterns and best practices
 * 
 * This example shows how to:
 * - Structure a basic unit test class
 * - Use Qt Test Framework macros effectively
 * - Implement proper test setup and teardown
 * - Test both success and failure scenarios
 * - Use appropriate assertions for different data types
 * 
 * Key learning points:
 * - Follow the Arrange-Act-Assert pattern
 * - Use descriptive test method names
 * - Test edge cases and error conditions
 * - Provide meaningful failure messages
 */

#include <QtTest>
#include <QTemporaryDir>
#include <QFile>
#include <QTextStream>

// Mock class to demonstrate testing - represents a simple file processor
class FileProcessor {
public:
    explicit FileProcessor(const QString& workingDirectory = QString())
        : m_workingDir(workingDirectory) {}
    
    bool processFile(const QString& filename) {
        if (filename.isEmpty()) {
            return false;
        }
        
        QString fullPath = m_workingDir.isEmpty() ? filename : 
                          m_workingDir + "/" + filename;
        
        QFile file(fullPath);
        if (!file.exists()) {
            return false;
        }
        
        if (!file.open(QIODevice::ReadOnly)) {
            return false;
        }
        
        QTextStream stream(&file);
        QString content = stream.readAll();
        
        // Simple processing: count lines
        m_lastLineCount = content.split('\n', Qt::SkipEmptyParts).size();
        return true;
    }
    
    int getLastLineCount() const { return m_lastLineCount; }
    QString getWorkingDirectory() const { return m_workingDir; }
    void setWorkingDirectory(const QString& dir) { m_workingDir = dir; }
    
private:
    QString m_workingDir;
    int m_lastLineCount = 0;
};

/**
 * Basic unit test class demonstrating fundamental testing patterns
 */
class BasicUnitTest : public QObject {
    Q_OBJECT

private slots:
    // Test lifecycle methods
    void initTestCase();    // Run once before all tests
    void init();           // Run before each test method
    void cleanup();        // Run after each test method
    void cleanupTestCase(); // Run once after all tests
    
    // Test methods - each tests a specific behavior
    void testConstructor_WhenCreatedWithoutDirectory_HasEmptyWorkingDirectory();
    void testConstructor_WhenCreatedWithDirectory_SetsWorkingDirectory();
    void testProcessFile_WhenFileExists_ReturnsTrue();
    void testProcessFile_WhenFileDoesNotExist_ReturnsFalse();
    void testProcessFile_WhenFilenameIsEmpty_ReturnsFalse();
    void testProcessFile_WhenFileHasMultipleLines_CountsCorrectly();
    void testSetWorkingDirectory_WhenCalled_UpdatesDirectory();

private:
    // Helper methods
    void createTestFile(const QString& filename, const QString& content);
    QString getTestFilePath(const QString& filename) const;
    
    // Test data
    QTemporaryDir* m_tempDir;
    FileProcessor* m_processor;
};

void BasicUnitTest::initTestCase() {
    // This runs once before all test methods
    qDebug() << "Starting BasicUnitTest suite";
    
    // Any global setup can go here
    // For example, initializing logging, loading configuration, etc.
}

void BasicUnitTest::init() {
    // This runs before each test method
    // Create fresh test environment for each test
    m_tempDir = new QTemporaryDir();
    QVERIFY2(m_tempDir->isValid(), "Failed to create temporary directory");
    
    m_processor = new FileProcessor(m_tempDir->path());
}

void BasicUnitTest::cleanup() {
    // This runs after each test method
    // Clean up resources to ensure test independence
    delete m_processor;
    m_processor = nullptr;
    
    delete m_tempDir;
    m_tempDir = nullptr;
}

void BasicUnitTest::cleanupTestCase() {
    // This runs once after all test methods
    qDebug() << "Completed BasicUnitTest suite";
}

void BasicUnitTest::testConstructor_WhenCreatedWithoutDirectory_HasEmptyWorkingDirectory() {
    // Arrange - Create object without directory
    FileProcessor processor;
    
    // Act - Get the working directory
    QString workingDir = processor.getWorkingDirectory();
    
    // Assert - Should be empty
    QVERIFY2(workingDir.isEmpty(), "Working directory should be empty when not specified");
}

void BasicUnitTest::testConstructor_WhenCreatedWithDirectory_SetsWorkingDirectory() {
    // Arrange
    QString expectedDir = "/test/directory";
    
    // Act
    FileProcessor processor(expectedDir);
    
    // Assert
    QCOMPARE(processor.getWorkingDirectory(), expectedDir);
}

void BasicUnitTest::testProcessFile_WhenFileExists_ReturnsTrue() {
    // Arrange
    QString filename = "test_file.txt";
    QString content = "Line 1\nLine 2\nLine 3";
    createTestFile(filename, content);
    
    // Act
    bool result = m_processor->processFile(filename);
    
    // Assert
    QVERIFY2(result, "Processing should succeed for existing file");
    QCOMPARE(m_processor->getLastLineCount(), 3);
}

void BasicUnitTest::testProcessFile_WhenFileDoesNotExist_ReturnsFalse() {
    // Arrange
    QString nonExistentFile = "does_not_exist.txt";
    
    // Act
    bool result = m_processor->processFile(nonExistentFile);
    
    // Assert
    QVERIFY2(!result, "Processing should fail for non-existent file");
    QCOMPARE(m_processor->getLastLineCount(), 0);
}

void BasicUnitTest::testProcessFile_WhenFilenameIsEmpty_ReturnsFalse() {
    // Arrange
    QString emptyFilename = "";
    
    // Act
    bool result = m_processor->processFile(emptyFilename);
    
    // Assert
    QVERIFY2(!result, "Processing should fail for empty filename");
}

void BasicUnitTest::testProcessFile_WhenFileHasMultipleLines_CountsCorrectly() {
    // Arrange - Test various line count scenarios
    struct TestCase {
        QString filename;
        QString content;
        int expectedLines;
    };
    
    QList<TestCase> testCases = {
        {"empty.txt", "", 0},
        {"single_line.txt", "Single line", 1},
        {"two_lines.txt", "Line 1\nLine 2", 2},
        {"with_empty_lines.txt", "Line 1\n\nLine 3\n", 2}, // Empty lines skipped
        {"trailing_newline.txt", "Line 1\nLine 2\n", 2}
    };
    
    for (const auto& testCase : testCases) {
        // Arrange
        createTestFile(testCase.filename, testCase.content);
        
        // Act
        bool result = m_processor->processFile(testCase.filename);
        
        // Assert
        QVERIFY2(result, QString("Processing should succeed for %1").arg(testCase.filename).toLocal8Bit());
        QCOMPARE(m_processor->getLastLineCount(), testCase.expectedLines);
    }
}

void BasicUnitTest::testSetWorkingDirectory_WhenCalled_UpdatesDirectory() {
    // Arrange
    QString newDirectory = "/new/test/directory";
    QString originalDirectory = m_processor->getWorkingDirectory();
    
    // Act
    m_processor->setWorkingDirectory(newDirectory);
    
    // Assert
    QCOMPARE(m_processor->getWorkingDirectory(), newDirectory);
    QVERIFY(m_processor->getWorkingDirectory() != originalDirectory);
}

// Helper method implementations
void BasicUnitTest::createTestFile(const QString& filename, const QString& content) {
    QString fullPath = getTestFilePath(filename);
    
    // Create subdirectories if needed
    QFileInfo fileInfo(fullPath);
    QDir dir = fileInfo.dir();
    if (!dir.exists()) {
        QVERIFY2(dir.mkpath("."), QString("Failed to create directory: %1").arg(dir.path()).toLocal8Bit());
    }
    
    QFile file(fullPath);
    QVERIFY2(file.open(QIODevice::WriteOnly), QString("Failed to create test file: %1").arg(fullPath).toLocal8Bit());
    
    QTextStream stream(&file);
    stream << content;
    file.close();
    
    QVERIFY2(file.exists(), QString("Test file was not created: %1").arg(fullPath).toLocal8Bit());
}

QString BasicUnitTest::getTestFilePath(const QString& filename) const {
    return m_tempDir->path() + "/" + filename;
}

// Qt Test Framework boilerplate
QTEST_MAIN(BasicUnitTest)
#include "basic-unit-test.moc"

/*
 * Compilation and execution:
 * 
 * g++ -I/path/to/qt/include -I/path/to/qt/include/QtTest -I/path/to/qt/include/QtCore \
 *     basic-unit-test.cpp -o basic-unit-test \
 *     -lQt6Test -lQt6Core -fPIC
 * 
 * ./basic-unit-test
 * 
 * Expected output:
 * ********* Start testing of BasicUnitTest *********
 * Config: Using QtTest library 6.x.x
 * PASS   : BasicUnitTest::initTestCase()
 * PASS   : BasicUnitTest::testConstructor_WhenCreatedWithoutDirectory_HasEmptyWorkingDirectory()
 * PASS   : BasicUnitTest::testConstructor_WhenCreatedWithDirectory_SetsWorkingDirectory()
 * PASS   : BasicUnitTest::testProcessFile_WhenFileExists_ReturnsTrue()
 * PASS   : BasicUnitTest::testProcessFile_WhenFileDoesNotExist_ReturnsFalse()
 * PASS   : BasicUnitTest::testProcessFile_WhenFilenameIsEmpty_ReturnsFalse()
 * PASS   : BasicUnitTest::testProcessFile_WhenFileHasMultipleLines_CountsCorrectly()
 * PASS   : BasicUnitTest::testSetWorkingDirectory_WhenCalled_UpdatesDirectory()
 * PASS   : BasicUnitTest::cleanupTestCase()
 * Totals: 8 passed, 0 failed, 0 skipped, 0 blacklisted, Xms
 * ********* Finished testing of BasicUnitTest *********
 */