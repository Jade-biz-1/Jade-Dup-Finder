/**
 * End-to-End Workflow Test Suite
 * 
 * This test suite validates the complete scan → hash → detect → results workflow
 * with various real-world scenarios including:
 * - Real directory structures
 * - Edge cases (empty directories, symlinks, etc.)
 * - Cross-platform compatibility
 * - Various file system scenarios
 * 
 * Requirements: 4.3, 4.4
 */

#include <QCoreApplication>
#include <QDebug>
#include <QTimer>
#include <QEventLoop>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QCryptographicHash>

#include "file_scanner.h"
#include "hash_calculator.h"
#include "duplicate_detector.h"

// Test result tracking
struct TestResult {
    QString testName;
    bool passed;
    QString message;
    qint64 durationMs;
};

class EndToEndTestSuite {
public:
    EndToEndTestSuite() : m_totalTests(0), m_passedTests(0) {}
    
    void runAllTests() {
        qDebug() << "========================================";
        qDebug() << "End-to-End Workflow Test Suite";
        qDebug() << "========================================";
        qDebug();
        
        // Run all test scenarios
        testBasicWorkflow();
        testEmptyDirectories();
        testSymlinks();
        testLargeFileSet();
        testMixedFileSizes();
        testNestedDirectories();
        testSpecialCharactersInPaths();
        testConcurrentScans();
        testErrorRecovery();
        testPatternFiltering();
        
        // Print summary
        printSummary();
    }
    
    int getExitCode() const {
        return (m_passedTests == m_totalTests) ? 0 : 1;
    }
    
private:
    QList<TestResult> m_results;
    int m_totalTests;
    int m_passedTests;
    
    void recordTest(const QString& name, bool passed, const QString& message, qint64 durationMs) {
        TestResult result{name, passed, message, durationMs};
        m_results.append(result);
        m_totalTests++;
        if (passed) m_passedTests++;
        
        qDebug() << (passed ? "[PASS]" : "[FAIL]") << name;
        if (!message.isEmpty()) {
            qDebug() << "      " << message;
        }
        qDebug() << "       Duration:" << durationMs << "ms";
        qDebug();
    }
    
    void printSummary() {
        qDebug() << "========================================";
        qDebug() << "Test Summary";
        qDebug() << "========================================";
        qDebug() << "Total tests:" << m_totalTests;
        qDebug() << "Passed:" << m_passedTests;
        qDebug() << "Failed:" << (m_totalTests - m_passedTests);
        qDebug() << "Success rate:" << QString::number((double)m_passedTests / m_totalTests * 100, 'f', 1) + "%";
        qDebug();
        
        if (m_passedTests < m_totalTests) {
            qDebug() << "Failed tests:";
            for (const TestResult& result : m_results) {
                if (!result.passed) {
                    qDebug() << "  -" << result.testName << ":" << result.message;
                }
            }
        }
        
        qDebug() << "========================================";
    }
    
    // Test 1: Basic workflow with known duplicates
    void testBasicWorkflow() {
        QElapsedTimer timer;
        timer.start();
        
        QString testName = "Basic Workflow (scan → hash → detect)";
        qDebug() << "Running:" << testName;
        
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            recordTest(testName, false, "Failed to create temp directory", timer.elapsed());
            return;
        }
        
        // Create test structure
        QString basePath = tempDir.path();
        QDir().mkpath(basePath + "/folder1");
        QDir().mkpath(basePath + "/folder2");
        
        // Create duplicate files
        QByteArray content = "This is duplicate content for testing.";
        createFile(basePath + "/folder1/file1.txt", content);
        createFile(basePath + "/folder2/file1_copy.txt", content);
        createFile(basePath + "/folder2/file1_backup.txt", content);
        
        // Create unique file
        createFile(basePath + "/folder1/unique.txt", "Unique content here.");
        
        // Run workflow
        auto scannedFiles = runScan(basePath, 0);
        if (scannedFiles.isEmpty()) {
            recordTest(testName, false, "Scan returned no files", timer.elapsed());
            return;
        }
        
        auto duplicates = runFullWorkflow(scannedFiles);
        
        // Verify results
        bool passed = true;
        QString message;
        
        if (scannedFiles.size() != 4) {
            passed = false;
            message = QString("Expected 4 files, got %1").arg(scannedFiles.size());
        } else if (duplicates.size() != 1) {
            passed = false;
            message = QString("Expected 1 duplicate group, got %1").arg(duplicates.size());
        } else if (duplicates[0].fileCount != 3) {
            passed = false;
            message = QString("Expected 3 files in duplicate group, got %1").arg(duplicates[0].fileCount);
        } else {
            message = QString("Found %1 files, %2 duplicate groups").arg(scannedFiles.size()).arg(duplicates.size());
        }
        
        recordTest(testName, passed, message, timer.elapsed());
    }
    
    // Test 2: Empty directories
    void testEmptyDirectories() {
        QElapsedTimer timer;
        timer.start();
        
        QString testName = "Empty Directories Handling";
        qDebug() << "Running:" << testName;
        
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            recordTest(testName, false, "Failed to create temp directory", timer.elapsed());
            return;
        }
        
        QString basePath = tempDir.path();
        QDir().mkpath(basePath + "/empty1");
        QDir().mkpath(basePath + "/empty2/nested_empty");
        QDir().mkpath(basePath + "/with_file");
        
        createFile(basePath + "/with_file/test.txt", "Content");
        
        auto scannedFiles = runScan(basePath, 0);
        
        bool passed = (scannedFiles.size() == 1);
        QString message = passed ? 
            "Correctly handled empty directories" :
            QString("Expected 1 file, got %1").arg(scannedFiles.size());
        
        recordTest(testName, passed, message, timer.elapsed());
    }
    
    // Test 3: Symlinks handling
    void testSymlinks() {
        QElapsedTimer timer;
        timer.start();
        
        QString testName = "Symlink Handling";
        qDebug() << "Running:" << testName;
        
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            recordTest(testName, false, "Failed to create temp directory", timer.elapsed());
            return;
        }
        
        QString basePath = tempDir.path();
        QDir().mkpath(basePath + "/real");
        
        createFile(basePath + "/real/file.txt", "Real file content");
        
        // Create symlink (platform-specific)
#ifdef Q_OS_UNIX
        QFile::link(basePath + "/real/file.txt", basePath + "/symlink.txt");
        QFile::link(basePath + "/real", basePath + "/symlink_dir");
#endif
        
        auto scannedFiles = runScan(basePath, 0);
        
        // On Unix, we should handle symlinks gracefully
        // The exact behavior depends on implementation, but it shouldn't crash
        bool passed = !scannedFiles.isEmpty();
        QString message = QString("Scanned %1 files with symlinks present").arg(scannedFiles.size());
        
        recordTest(testName, passed, message, timer.elapsed());
    }
    
    // Test 4: Large file set (stress test)
    void testLargeFileSet() {
        QElapsedTimer timer;
        timer.start();
        
        QString testName = "Large File Set (1000+ files)";
        qDebug() << "Running:" << testName;
        
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            recordTest(testName, false, "Failed to create temp directory", timer.elapsed());
            return;
        }
        
        QString basePath = tempDir.path();
        
        // Create 1000 files with some duplicates
        int totalFiles = 1000;
        int duplicateGroups = 10;
        int filesPerGroup = 5;
        
        for (int group = 0; group < duplicateGroups; group++) {
            QByteArray content = QString("Duplicate group %1 content").arg(group).toUtf8();
            for (int i = 0; i < filesPerGroup; i++) {
                QString filePath = QString("%1/group%2_file%3.txt").arg(basePath).arg(group).arg(i);
                createFile(filePath, content);
            }
        }
        
        // Create unique files
        int uniqueFiles = totalFiles - (duplicateGroups * filesPerGroup);
        for (int i = 0; i < uniqueFiles; i++) {
            QString filePath = QString("%1/unique_%2.txt").arg(basePath).arg(i);
            createFile(filePath, QString("Unique content %1").arg(i).toUtf8());
        }
        
        auto scannedFiles = runScan(basePath, 0);
        auto duplicates = runFullWorkflow(scannedFiles);
        
        bool passed = true;
        QString message;
        
        if (scannedFiles.size() != totalFiles) {
            passed = false;
            message = QString("Expected %1 files, got %2").arg(totalFiles).arg(scannedFiles.size());
        } else if (duplicates.size() != duplicateGroups) {
            passed = false;
            message = QString("Expected %1 duplicate groups, got %2").arg(duplicateGroups).arg(duplicates.size());
        } else {
            message = QString("Successfully processed %1 files, found %2 duplicate groups")
                .arg(scannedFiles.size()).arg(duplicates.size());
        }
        
        recordTest(testName, passed, message, timer.elapsed());
    }
    
    // Test 5: Mixed file sizes
    void testMixedFileSizes() {
        QElapsedTimer timer;
        timer.start();
        
        QString testName = "Mixed File Sizes";
        qDebug() << "Running:" << testName;
        
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            recordTest(testName, false, "Failed to create temp directory", timer.elapsed());
            return;
        }
        
        QString basePath = tempDir.path();
        
        // Create files of various sizes
        createFile(basePath + "/empty.txt", "");
        createFile(basePath + "/tiny.txt", "x");
        createFile(basePath + "/small.txt", QByteArray(100, 'a'));
        createFile(basePath + "/medium.txt", QByteArray(10000, 'b'));
        createFile(basePath + "/large.txt", QByteArray(1000000, 'c'));
        
        // Create duplicates of different sizes
        createFile(basePath + "/small_dup.txt", QByteArray(100, 'a'));
        createFile(basePath + "/medium_dup.txt", QByteArray(10000, 'b'));
        
        auto scannedFiles = runScan(basePath, 0);
        auto duplicates = runFullWorkflow(scannedFiles);
        
        bool passed = (scannedFiles.size() == 7 && duplicates.size() == 2);
        QString message = passed ?
            QString("Correctly handled mixed sizes: %1 files, %2 duplicate groups")
                .arg(scannedFiles.size()).arg(duplicates.size()) :
            QString("Expected 7 files and 2 groups, got %1 files and %2 groups")
                .arg(scannedFiles.size()).arg(duplicates.size());
        
        recordTest(testName, passed, message, timer.elapsed());
    }
    
    // Test 6: Nested directories
    void testNestedDirectories() {
        QElapsedTimer timer;
        timer.start();
        
        QString testName = "Deeply Nested Directories";
        qDebug() << "Running:" << testName;
        
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            recordTest(testName, false, "Failed to create temp directory", timer.elapsed());
            return;
        }
        
        QString basePath = tempDir.path();
        
        // Create deeply nested structure
        QString nestedPath = basePath;
        for (int i = 0; i < 10; i++) {
            nestedPath += QString("/level%1").arg(i);
            QDir().mkpath(nestedPath);
            createFile(nestedPath + "/file.txt", QString("Content at level %1").arg(i).toUtf8());
        }
        
        auto scannedFiles = runScan(basePath, 0);
        
        bool passed = (scannedFiles.size() == 10);
        QString message = passed ?
            "Successfully scanned deeply nested directories" :
            QString("Expected 10 files, got %1").arg(scannedFiles.size());
        
        recordTest(testName, passed, message, timer.elapsed());
    }
    
    // Test 7: Special characters in paths
    void testSpecialCharactersInPaths() {
        QElapsedTimer timer;
        timer.start();
        
        QString testName = "Special Characters in Paths";
        qDebug() << "Running:" << testName;
        
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            recordTest(testName, false, "Failed to create temp directory", timer.elapsed());
            return;
        }
        
        QString basePath = tempDir.path();
        
        // Create files with special characters (platform-safe)
        QStringList specialNames = {
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.multiple.dots.txt",
            "file(with)parens.txt"
        };
        
        for (const QString& name : specialNames) {
            createFile(basePath + "/" + name, "Content");
        }
        
        auto scannedFiles = runScan(basePath, 0);
        
        bool passed = (scannedFiles.size() == specialNames.size());
        QString message = passed ?
            QString("Successfully handled %1 files with special characters").arg(scannedFiles.size()) :
            QString("Expected %1 files, got %2").arg(specialNames.size()).arg(scannedFiles.size());
        
        recordTest(testName, passed, message, timer.elapsed());
    }
    
    // Test 8: Concurrent scans
    void testConcurrentScans() {
        QElapsedTimer timer;
        timer.start();
        
        QString testName = "Concurrent Scan Operations";
        qDebug() << "Running:" << testName;
        
        // Create two separate temp directories
        QTemporaryDir tempDir1, tempDir2;
        if (!tempDir1.isValid() || !tempDir2.isValid()) {
            recordTest(testName, false, "Failed to create temp directories", timer.elapsed());
            return;
        }
        
        // Populate both directories
        for (int i = 0; i < 10; i++) {
            createFile(tempDir1.path() + QString("/file%1.txt").arg(i), QString("Content %1").arg(i).toUtf8());
            createFile(tempDir2.path() + QString("/file%1.txt").arg(i), QString("Content %1").arg(i).toUtf8());
        }
        
        // Run two scans concurrently
        FileScanner scanner1, scanner2;
        FileScanner::ScanOptions options;
        options.minimumFileSize = 0;
        
        options.targetPaths.clear();
        options.targetPaths << tempDir1.path();
        
        QEventLoop loop1;
        QList<FileScanner::FileInfo> files1;
        QObject::connect(&scanner1, &FileScanner::scanCompleted, [&]() {
            files1 = scanner1.getScannedFiles();
            loop1.quit();
        });
        
        scanner1.startScan(options);
        
        options.targetPaths.clear();
        options.targetPaths << tempDir2.path();
        
        QEventLoop loop2;
        QList<FileScanner::FileInfo> files2;
        QObject::connect(&scanner2, &FileScanner::scanCompleted, [&]() {
            files2 = scanner2.getScannedFiles();
            loop2.quit();
        });
        
        scanner2.startScan(options);
        
        // Wait for both
        QTimer::singleShot(5000, &loop1, &QEventLoop::quit);
        QTimer::singleShot(5000, &loop2, &QEventLoop::quit);
        loop1.exec();
        loop2.exec();
        
        bool passed = (files1.size() == 10 && files2.size() == 10);
        QString message = passed ?
            "Successfully ran concurrent scans" :
            QString("Expected 10 files each, got %1 and %2").arg(files1.size()).arg(files2.size());
        
        recordTest(testName, passed, message, timer.elapsed());
    }
    
    // Test 9: Error recovery
    void testErrorRecovery() {
        QElapsedTimer timer;
        timer.start();
        
        QString testName = "Error Recovery (invalid paths)";
        qDebug() << "Running:" << testName;
        
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            recordTest(testName, false, "Failed to create temp directory", timer.elapsed());
            return;
        }
        
        QString basePath = tempDir.path();
        createFile(basePath + "/valid.txt", "Valid content");
        
        // Try to scan with mix of valid and invalid paths
        FileScanner scanner;
        FileScanner::ScanOptions options;
        options.targetPaths << basePath;
        options.targetPaths << "/nonexistent/path/that/does/not/exist";
        options.minimumFileSize = 0;
        
        QEventLoop loop;
        QList<FileScanner::FileInfo> files;
        bool scanCompleted = false;
        
        QObject::connect(&scanner, &FileScanner::scanCompleted, [&]() {
            files = scanner.getScannedFiles();
            scanCompleted = true;
            loop.quit();
        });
        
        scanner.startScan(options);
        
        QTimer::singleShot(5000, &loop, &QEventLoop::quit);
        loop.exec();
        
        // Should complete and find at least the valid file
        bool passed = (scanCompleted && files.size() >= 1);
        QString message = passed ?
            QString("Recovered from errors, found %1 valid files").arg(files.size()) :
            "Failed to recover from invalid paths";
        
        recordTest(testName, passed, message, timer.elapsed());
    }
    
    // Test 10: Pattern filtering in workflow
    void testPatternFiltering() {
        QElapsedTimer timer;
        timer.start();
        
        QString testName = "Pattern Filtering in Workflow";
        qDebug() << "Running:" << testName;
        
        QTemporaryDir tempDir;
        if (!tempDir.isValid()) {
            recordTest(testName, false, "Failed to create temp directory", timer.elapsed());
            return;
        }
        
        QString basePath = tempDir.path();
        
        // Create files with different extensions
        createFile(basePath + "/doc1.txt", "Text content");
        createFile(basePath + "/doc2.txt", "Text content");
        createFile(basePath + "/image1.jpg", "Image data");
        createFile(basePath + "/image2.jpg", "Image data");
        createFile(basePath + "/data.csv", "CSV data");
        
        // Scan with pattern filter (only .txt files)
        FileScanner scanner;
        FileScanner::ScanOptions options;
        options.targetPaths << basePath;
        options.minimumFileSize = 0;
        options.includePatterns << "*.txt";
        
        QEventLoop loop;
        QList<FileScanner::FileInfo> files;
        
        QObject::connect(&scanner, &FileScanner::scanCompleted, [&]() {
            files = scanner.getScannedFiles();
            loop.quit();
        });
        
        scanner.startScan(options);
        
        QTimer::singleShot(5000, &loop, &QEventLoop::quit);
        loop.exec();
        
        bool passed = (files.size() == 2);
        QString message = passed ?
            "Pattern filtering worked correctly" :
            QString("Expected 2 .txt files, got %1").arg(files.size());
        
        recordTest(testName, passed, message, timer.elapsed());
    }
    
    // Helper methods
    void createFile(const QString& path, const QByteArray& content) {
        QFile file(path);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(content);
            file.close();
        }
    }
    
    QList<FileScanner::FileInfo> runScan(const QString& path, qint64 minSize) {
        FileScanner scanner;
        FileScanner::ScanOptions options;
        options.targetPaths << path;
        options.minimumFileSize = minSize;
        options.includeHiddenFiles = false;
        
        QEventLoop loop;
        QList<FileScanner::FileInfo> files;
        
        QObject::connect(&scanner, &FileScanner::scanCompleted, [&]() {
            files = scanner.getScannedFiles();
            loop.quit();
        });
        
        scanner.startScan(options);
        
        QTimer::singleShot(10000, &loop, &QEventLoop::quit);
        loop.exec();
        
        return files;
    }
    
    QList<DuplicateDetector::DuplicateGroup> runFullWorkflow(const QList<FileScanner::FileInfo>& files) {
        DuplicateDetector detector;
        DuplicateDetector::DetectionOptions options;
        options.level = DuplicateDetector::DetectionLevel::Standard;
        options.minimumFileSize = 0;
        options.skipEmptyFiles = false;
        
        detector.setOptions(options);
        
        QEventLoop loop;
        QList<DuplicateDetector::DuplicateGroup> groups;
        
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted, [&](int) {
            groups = detector.getDuplicateGroups();
            loop.quit();
        });
        
        detector.findDuplicates(files);
        
        QTimer::singleShot(15000, &loop, &QEventLoop::quit);
        loop.exec();
        
        return groups;
    }
};

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    
    EndToEndTestSuite suite;
    suite.runAllTests();
    
    return suite.getExitCode();
}
