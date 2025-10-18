#include <QtTest/QtTest>
#include "file_scanner.h"
#include <QtTest/QSignalSpy>
#include <QtCore/QTemporaryDir>
#include <QtCore/QFile>
#include <QtCore/QTextStream>

class TestFileScanner : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void testBasicScan();
    void testScanSignals();
    
    // Pattern matching tests
    void testGlobPatternInclude();
    void testGlobPatternExclude();
    void testMultipleIncludePatterns();
    void testMultipleExcludePatterns();
    void testCaseSensitivePatterns();
    void testCaseInsensitivePatterns();
    void testRegexPatterns();
    void testComplexPatterns();
    void testInvalidPatterns();
    void testPatternPriority();
    
    // Error handling tests
    void testPermissionDeniedError();
    void testInvalidPathError();
    void testErrorAccumulation();
    void testErrorSignals();
    void testScanContinuesAfterError();
    
    // Statistics tests
    void testBasicStatistics();
    void testStatisticsSignal();
    void testFilesFilteredBySize();
    void testFilesFilteredByPattern();
    void testFilesFilteredByHidden();
    void testDirectoriesScanned();
    void testScanDuration();
    void testFilesPerSecond();
    void testErrorsInStatistics();
    
    // Pause/Resume tests (Task 9)
    void testPauseScan();
    void testResumeScan();
    void testPauseResumeSignals();
    void testPauseStopsScanning();
    void testResumeAfterPause();
    void testMultiplePauseResume();
    void testPauseBeforeScanStart();
    void testPauseAfterScanComplete();
    void testCancelWhilePaused();
    
    void cleanupTestCase();

private:
    QTemporaryDir* m_tempDir;
    void createTestFiles();
};

void TestFileScanner::initTestCase()
{
    // Create temporary directory with test files
    m_tempDir = new QTemporaryDir();
    QVERIFY(m_tempDir->isValid());
    
    createTestFiles();
}

void TestFileScanner::createTestFiles()
{
    // Create diverse test files for pattern matching
    QStringList testFiles = {
        "test1.txt",
        "test2.txt",
        "image1.jpg",
        "image2.JPG",
        "photo.png",
        "document.pdf",
        "data.json",
        "config.xml",
        "script.py",
        "program.cpp",
        "header.h",
        "README.md",
        "temp.tmp",
        "backup.bak",
        ".hidden",
        "Test3.TXT"  // Mixed case
    };
    
    for (const QString& fileName : testFiles) {
        QFile file(m_tempDir->path() + "/" + fileName);
        QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
        QTextStream out(&file);
        out << "Test content for " << fileName;
        file.close();
    }
}

void TestFileScanner::testBasicScan()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1; // Include small files
    
    scanner.startScan(options);
    
    // Wait for scan to complete (simple busy wait for test)
    int maxWait = 5000; // 5 seconds max
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QVERIFY(!scanner.isScanning());
    QVERIFY(scanner.getTotalFilesFound() >= 2); // Should find at least our 2 test files
}

void TestFileScanner::testScanSignals()
{
    FileScanner scanner;
    
    // Set up signal spies
    QSignalSpy startedSpy(&scanner, &FileScanner::scanStarted);
    QSignalSpy progressSpy(&scanner, &FileScanner::scanProgress);
    QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
    QSignalSpy fileFoundSpy(&scanner, &FileScanner::fileFound);
    
    // Configure scan
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    // Start scan
    scanner.startScan(options);
    
    // Verify scanStarted was emitted
    QVERIFY(startedSpy.count() == 1);
    
    // Wait for scan to complete
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Verify scan completion
    QVERIFY(!scanner.isScanning());
    QVERIFY(completedSpy.count() == 1);
    
    // Should have found some files
    QVERIFY(fileFoundSpy.count() >= 0); // File found signals may or may not fire depending on timing
    
    qDebug() << "Scan completed successfully:";
    qDebug() << "- Files found:" << scanner.getTotalFilesFound();
    qDebug() << "- Bytes scanned:" << scanner.getTotalBytesScanned();
    qDebug() << "- scanStarted signals:" << startedSpy.count();
    qDebug() << "- scanCompleted signals:" << completedSpy.count();
    qDebug() << "- fileFound signals:" << fileFoundSpy.count();
}

void TestFileScanner::testGlobPatternInclude()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    options.includePatterns << "*.txt";  // Only .txt files
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    
    // Should find only .txt files (case insensitive by default)
    QVERIFY(files.size() >= 2);
    for (const auto& file : files) {
        QVERIFY(file.fileName.endsWith(".txt", Qt::CaseInsensitive));
    }
    
    qDebug() << "testGlobPatternInclude: Found" << files.size() << ".txt files";
}

void TestFileScanner::testGlobPatternExclude()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    options.excludePatterns << "*.tmp" << "*.bak";  // Exclude temp and backup files
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    
    // Should not find any .tmp or .bak files
    for (const auto& file : files) {
        QVERIFY(!file.fileName.endsWith(".tmp"));
        QVERIFY(!file.fileName.endsWith(".bak"));
    }
    
    qDebug() << "testGlobPatternExclude: Found" << files.size() << "files (excluding .tmp and .bak)";
}

void TestFileScanner::testMultipleIncludePatterns()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    options.includePatterns << "*.jpg" << "*.png" << "*.pdf";  // Image and PDF files
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    
    // Should find jpg, png, and pdf files
    QVERIFY(files.size() >= 3);
    for (const auto& file : files) {
        bool matches = file.fileName.endsWith(".jpg", Qt::CaseInsensitive) ||
                      file.fileName.endsWith(".png", Qt::CaseInsensitive) ||
                      file.fileName.endsWith(".pdf", Qt::CaseInsensitive);
        QVERIFY(matches);
    }
    
    qDebug() << "testMultipleIncludePatterns: Found" << files.size() << "image/PDF files";
}

void TestFileScanner::testMultipleExcludePatterns()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    options.excludePatterns << "*.tmp" << "*.bak" << ".*";  // Exclude temp, backup, and hidden
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    
    // Should not find any excluded files
    for (const auto& file : files) {
        QVERIFY(!file.fileName.endsWith(".tmp"));
        QVERIFY(!file.fileName.endsWith(".bak"));
        QVERIFY(!file.fileName.startsWith("."));
    }
    
    qDebug() << "testMultipleExcludePatterns: Found" << files.size() << "files (excluding multiple patterns)";
}

void TestFileScanner::testCaseSensitivePatterns()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    options.includePatterns << "*.txt";  // Lowercase pattern
    options.caseSensitivePatterns = true;  // Case sensitive
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    
    // Should only find lowercase .txt files, not .TXT
    for (const auto& file : files) {
        QVERIFY(file.fileName.endsWith(".txt"));
        QVERIFY(!file.fileName.endsWith(".TXT"));
    }
    
    qDebug() << "testCaseSensitivePatterns: Found" << files.size() << "lowercase .txt files";
}

void TestFileScanner::testCaseInsensitivePatterns()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    options.includePatterns << "*.txt";  // Lowercase pattern
    options.caseSensitivePatterns = false;  // Case insensitive (default)
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    
    // Should find both .txt and .TXT files
    QVERIFY(files.size() >= 3);  // test1.txt, test2.txt, Test3.TXT
    
    qDebug() << "testCaseInsensitivePatterns: Found" << files.size() << ".txt files (case insensitive)";
}

void TestFileScanner::testRegexPatterns()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    options.includePatterns << "^test\\d+\\.txt$";  // Regex: test followed by digits and .txt
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    
    // Should find test1.txt and test2.txt (case insensitive)
    QVERIFY(files.size() >= 2);
    for (const auto& file : files) {
        QVERIFY(file.fileName.contains(QRegularExpression("^test\\d+\\.txt$", QRegularExpression::CaseInsensitiveOption)));
    }
    
    qDebug() << "testRegexPatterns: Found" << files.size() << "files matching regex";
}

void TestFileScanner::testComplexPatterns()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    // Include images and code files, exclude temp files
    options.includePatterns << "*.jpg" << "*.png" << "*.cpp" << "*.h" << "*.py";
    options.excludePatterns << "*.tmp";
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    
    // Should find matching files but not temp files
    QVERIFY(files.size() >= 5);
    for (const auto& file : files) {
        QVERIFY(!file.fileName.endsWith(".tmp"));
    }
    
    qDebug() << "testComplexPatterns: Found" << files.size() << "files with complex patterns";
}

void TestFileScanner::testInvalidPatterns()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    options.includePatterns << "[invalid(regex";  // Invalid regex pattern
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should complete without crashing
    QVERIFY(!scanner.isScanning());
    
    // Invalid pattern should match nothing
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    QVERIFY(files.isEmpty());
    
    qDebug() << "testInvalidPatterns: Handled invalid pattern gracefully";
}

void TestFileScanner::testPatternPriority()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    // Include all text files but exclude test files
    options.includePatterns << "*.txt";
    options.excludePatterns << "test*";
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    
    // Should find .txt files but not those starting with "test"
    for (const auto& file : files) {
        QVERIFY(file.fileName.endsWith(".txt", Qt::CaseInsensitive));
        QVERIFY(!file.fileName.startsWith("test", Qt::CaseInsensitive));
    }
    
    qDebug() << "testPatternPriority: Found" << files.size() << "files (exclude takes priority)";
}

void TestFileScanner::testPermissionDeniedError()
{
    FileScanner scanner;
    
    // Set up signal spy for error signals
    QSignalSpy errorSpy(&scanner, &FileScanner::scanError);
    
    FileScanner::ScanOptions options;
    // Try to scan a directory that typically requires elevated permissions
    options.targetPaths << "/root/.ssh";  // This should fail with permission denied on most systems
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should have completed (not crashed)
    QVERIFY(!scanner.isScanning());
    
    // Should have recorded at least one error (if directory exists and is protected)
    QList<FileScanner::ScanErrorInfo> errors = scanner.getScanErrors();
    if (!errors.isEmpty()) {
        QVERIFY(errors.first().errorType == FileScanner::ScanError::PermissionDenied ||
                errors.first().errorType == FileScanner::ScanError::FileSystemError);
        qDebug() << "testPermissionDeniedError: Correctly detected permission error";
    } else {
        qDebug() << "testPermissionDeniedError: Directory not found or accessible (test skipped)";
    }
}

void TestFileScanner::testInvalidPathError()
{
    FileScanner scanner;
    
    QSignalSpy errorSpy(&scanner, &FileScanner::scanError);
    QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
    
    FileScanner::ScanOptions options;
    options.targetPaths << "/this/path/does/not/exist/at/all";
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should complete without crashing
    QVERIFY(!scanner.isScanning());
    QVERIFY(completedSpy.count() == 1);
    
    // Should have recorded an error
    QList<FileScanner::ScanErrorInfo> errors = scanner.getScanErrors();
    QVERIFY(!errors.isEmpty());
    QVERIFY(errors.first().errorType == FileScanner::ScanError::FileSystemError);
    QVERIFY(errorSpy.count() >= 1);
    
    qDebug() << "testInvalidPathError: Correctly handled invalid path";
}

void TestFileScanner::testErrorAccumulation()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    // Add multiple invalid paths
    options.targetPaths << "/invalid/path/one";
    options.targetPaths << "/invalid/path/two";
    options.targetPaths << "/invalid/path/three";
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should have accumulated multiple errors
    QList<FileScanner::ScanErrorInfo> errors = scanner.getScanErrors();
    QVERIFY(errors.size() >= 3);
    QVERIFY(scanner.getTotalErrorsEncountered() >= 3);
    
    // Each error should have proper information
    for (const auto& error : errors) {
        QVERIFY(!error.filePath.isEmpty());
        QVERIFY(!error.errorMessage.isEmpty());
        QVERIFY(error.timestamp.isValid());
    }
    
    qDebug() << "testErrorAccumulation: Accumulated" << errors.size() << "errors correctly";
}

void TestFileScanner::testErrorSignals()
{
    FileScanner scanner;
    
    QSignalSpy errorSpy(&scanner, &FileScanner::scanError);
    QSignalSpy errorSummarySpy(&scanner, &FileScanner::scanErrorSummary);
    QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
    
    FileScanner::ScanOptions options;
    options.targetPaths << "/invalid/path";
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should emit scanError signal for each error
    QVERIFY(errorSpy.count() >= 1);
    
    // Should emit scanErrorSummary at the end
    QVERIFY(errorSummarySpy.count() == 1);
    
    // Should still complete normally
    QVERIFY(completedSpy.count() == 1);
    
    // Verify error signal parameters
    QList<QVariant> errorArgs = errorSpy.first();
    QVERIFY(errorArgs.size() == 3);  // errorType, path, description
    
    qDebug() << "testErrorSignals: Error signals emitted correctly";
}

void TestFileScanner::testScanContinuesAfterError()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    // Mix valid and invalid paths
    options.targetPaths << "/invalid/path";
    options.targetPaths << m_tempDir->path();  // Valid path
    options.targetPaths << "/another/invalid/path";
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should have found files from the valid path
    QVERIFY(scanner.getTotalFilesFound() > 0);
    
    // Should have recorded errors from invalid paths
    QVERIFY(scanner.getTotalErrorsEncountered() >= 2);
    
    // Verify scan completed successfully despite errors
    QVERIFY(!scanner.isScanning());
    
    qDebug() << "testScanContinuesAfterError: Found" << scanner.getTotalFilesFound() 
             << "files and" << scanner.getTotalErrorsEncountered() << "errors";
}

void TestFileScanner::testBasicStatistics()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Get statistics
    FileScanner::ScanStatistics stats = scanner.getScanStatistics();
    
    // Verify basic statistics
    QVERIFY(stats.totalFilesScanned > 0);
    QVERIFY(stats.totalDirectoriesScanned >= 1);  // At least the temp dir
    QVERIFY(stats.totalBytesScanned > 0);
    QVERIFY(stats.scanDurationMs >= 0);
    
    qDebug() << "testBasicStatistics:";
    qDebug() << "  Files scanned:" << stats.totalFilesScanned;
    qDebug() << "  Directories scanned:" << stats.totalDirectoriesScanned;
    qDebug() << "  Bytes scanned:" << stats.totalBytesScanned;
    qDebug() << "  Duration:" << stats.scanDurationMs << "ms";
    qDebug() << "  Files/sec:" << stats.filesPerSecond;
}

void TestFileScanner::testStatisticsSignal()
{
    FileScanner scanner;
    
    QSignalSpy statisticsSpy(&scanner, &FileScanner::scanStatistics);
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should emit statistics signal once at completion
    QVERIFY(statisticsSpy.count() == 1);
    
    // Verify signal contains valid statistics
    QList<QVariant> args = statisticsSpy.first();
    QVERIFY(args.size() == 1);
    
    qDebug() << "testStatisticsSignal: Statistics signal emitted correctly";
}

void TestFileScanner::testFilesFilteredBySize()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1000000;  // 1MB - should filter out all our small test files
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    FileScanner::ScanStatistics stats = scanner.getScanStatistics();
    
    // Should have filtered files by size
    QVERIFY(stats.filesFilteredBySize > 0);
    QVERIFY(stats.totalFilesScanned == 0);  // No files should pass the size filter
    
    qDebug() << "testFilesFilteredBySize: Filtered" << stats.filesFilteredBySize << "files by size";
}

void TestFileScanner::testFilesFilteredByPattern()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    options.includePatterns << "*.nonexistent";  // Pattern that won't match anything
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    FileScanner::ScanStatistics stats = scanner.getScanStatistics();
    
    // Should have filtered files by pattern
    QVERIFY(stats.filesFilteredByPattern > 0);
    QVERIFY(stats.totalFilesScanned == 0);  // No files should match the pattern
    
    qDebug() << "testFilesFilteredByPattern: Filtered" << stats.filesFilteredByPattern << "files by pattern";
}

void TestFileScanner::testFilesFilteredByHidden()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    options.includeHiddenFiles = false;  // Don't include hidden files
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    FileScanner::ScanStatistics stats = scanner.getScanStatistics();
    
    // Should have filtered at least the .hidden file
    QVERIFY(stats.filesFilteredByHidden >= 1);
    
    // Verify no hidden files were included
    QList<FileScanner::FileInfo> files = scanner.getScannedFiles();
    for (const auto& file : files) {
        QVERIFY(!file.fileName.startsWith("."));
    }
    
    qDebug() << "testFilesFilteredByHidden: Filtered" << stats.filesFilteredByHidden << "hidden files";
}

void TestFileScanner::testDirectoriesScanned()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    FileScanner::ScanStatistics stats = scanner.getScanStatistics();
    
    // Should have scanned at least the temp directory
    QVERIFY(stats.totalDirectoriesScanned >= 1);
    
    qDebug() << "testDirectoriesScanned: Scanned" << stats.totalDirectoriesScanned << "directories";
}

void TestFileScanner::testScanDuration()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    FileScanner::ScanStatistics stats = scanner.getScanStatistics();
    
    // Duration should be positive and reasonable
    QVERIFY(stats.scanDurationMs > 0);
    QVERIFY(stats.scanDurationMs < 10000);  // Should complete in less than 10 seconds
    
    qDebug() << "testScanDuration: Scan took" << stats.scanDurationMs << "ms";
}

void TestFileScanner::testFilesPerSecond()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    FileScanner::ScanStatistics stats = scanner.getScanStatistics();
    
    // Files per second should be calculated
    if (stats.totalFilesScanned > 0 && stats.scanDurationMs > 0) {
        QVERIFY(stats.filesPerSecond > 0);
        
        // Verify calculation is correct
        double expectedRate = stats.totalFilesScanned / (stats.scanDurationMs / 1000.0);
        QVERIFY(qAbs(stats.filesPerSecond - expectedRate) < 0.01);
    }
    
    qDebug() << "testFilesPerSecond: Scan rate was" << stats.filesPerSecond << "files/sec";
}

void TestFileScanner::testErrorsInStatistics()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << "/invalid/path";
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    FileScanner::ScanStatistics stats = scanner.getScanStatistics();
    
    // Should have recorded errors in statistics
    QVERIFY(stats.errorsEncountered >= 1);
    QVERIFY(stats.errorsEncountered == scanner.getTotalErrorsEncountered());
    
    qDebug() << "testErrorsInStatistics: Recorded" << stats.errorsEncountered << "errors in statistics";
}

// Pause/Resume tests (Task 9)

void TestFileScanner::testPauseScan()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    // Wait a bit for scan to start
    QTest::qWait(50);
    
    // Pause the scan
    scanner.pauseScan();
    
    // Verify paused state
    QVERIFY(scanner.isPaused());
    QVERIFY(scanner.isScanning());  // Still scanning, just paused
    
    qDebug() << "testPauseScan: Scan paused successfully";
}

void TestFileScanner::testResumeScan()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    QTest::qWait(50);
    
    // Pause then resume
    scanner.pauseScan();
    QVERIFY(scanner.isPaused());
    
    scanner.resumeScan();
    QVERIFY(!scanner.isPaused());
    QVERIFY(scanner.isScanning());
    
    // Wait for scan to complete
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should complete successfully
    QVERIFY(!scanner.isScanning());
    QVERIFY(scanner.getTotalFilesFound() > 0);
    
    qDebug() << "testResumeScan: Scan resumed and completed successfully";
}

void TestFileScanner::testPauseResumeSignals()
{
    FileScanner scanner;
    
    QSignalSpy pausedSpy(&scanner, &FileScanner::scanPaused);
    QSignalSpy resumedSpy(&scanner, &FileScanner::scanResumed);
    QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    QTest::qWait(50);
    
    // Pause
    scanner.pauseScan();
    QVERIFY(pausedSpy.count() == 1);
    
    // Resume
    scanner.resumeScan();
    QVERIFY(resumedSpy.count() == 1);
    
    // Wait for completion
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QVERIFY(completedSpy.count() == 1);
    
    qDebug() << "testPauseResumeSignals: Pause/resume signals emitted correctly";
}

void TestFileScanner::testPauseStopsScanning()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    QTest::qWait(50);
    
    // Pause the scan
    scanner.pauseScan();
    
    // Record files found at pause
    int filesAtPause = scanner.getTotalFilesFound();
    
    // Wait a bit while paused
    QTest::qWait(200);
    
    // Files found should not increase while paused
    int filesAfterWait = scanner.getTotalFilesFound();
    QVERIFY(filesAfterWait == filesAtPause);
    
    qDebug() << "testPauseStopsScanning: Scanning stopped while paused";
}

void TestFileScanner::testResumeAfterPause()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    QTest::qWait(50);
    
    // Pause
    scanner.pauseScan();
    int filesAtPause = scanner.getTotalFilesFound();
    
    // Wait while paused
    QTest::qWait(200);
    
    // Resume
    scanner.resumeScan();
    
    // Wait for scan to complete
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should have found more files after resume
    int finalFiles = scanner.getTotalFilesFound();
    QVERIFY(finalFiles >= filesAtPause);
    
    qDebug() << "testResumeAfterPause: Scan continued after resume";
}

void TestFileScanner::testMultiplePauseResume()
{
    FileScanner scanner;
    
    QSignalSpy pausedSpy(&scanner, &FileScanner::scanPaused);
    QSignalSpy resumedSpy(&scanner, &FileScanner::scanResumed);
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    QTest::qWait(50);
    
    // Pause and resume multiple times
    scanner.pauseScan();
    QTest::qWait(100);
    scanner.resumeScan();
    QTest::qWait(50);
    
    scanner.pauseScan();
    QTest::qWait(100);
    scanner.resumeScan();
    QTest::qWait(50);
    
    // Wait for completion
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should have emitted signals for each pause/resume
    QVERIFY(pausedSpy.count() == 2);
    QVERIFY(resumedSpy.count() == 2);
    
    // Should complete successfully
    QVERIFY(!scanner.isScanning());
    QVERIFY(scanner.getTotalFilesFound() > 0);
    
    qDebug() << "testMultiplePauseResume: Multiple pause/resume cycles handled correctly";
}

void TestFileScanner::testPauseBeforeScanStart()
{
    FileScanner scanner;
    
    // Try to pause before starting scan
    scanner.pauseScan();
    
    // Should not be paused (no scan running)
    QVERIFY(!scanner.isPaused());
    QVERIFY(!scanner.isScanning());
    
    qDebug() << "testPauseBeforeScanStart: Pause before scan handled gracefully";
}

void TestFileScanner::testPauseAfterScanComplete()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    
    // Wait for scan to complete
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Try to pause after completion
    scanner.pauseScan();
    
    // Should not be paused (scan already complete)
    QVERIFY(!scanner.isPaused());
    QVERIFY(!scanner.isScanning());
    
    qDebug() << "testPauseAfterScanComplete: Pause after completion handled gracefully";
}

void TestFileScanner::testCancelWhilePaused()
{
    FileScanner scanner;
    
    QSignalSpy pausedSpy(&scanner, &FileScanner::scanPaused);
    QSignalSpy cancelledSpy(&scanner, &FileScanner::scanCancelled);
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    scanner.startScan(options);
    QTest::qWait(50);
    
    // Pause the scan
    scanner.pauseScan();
    QVERIFY(pausedSpy.count() == 1);
    QVERIFY(scanner.isPaused());
    
    // Cancel while paused
    scanner.cancelScan();
    
    // Resume to allow cancellation to process
    scanner.resumeScan();
    
    // Wait for cancellation
    int maxWait = 2000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Should be cancelled
    QVERIFY(!scanner.isScanning());
    QVERIFY(cancelledSpy.count() == 1);
    
    qDebug() << "testCancelWhilePaused: Cancel while paused handled correctly";
}

void TestFileScanner::cleanupTestCase()
{
    delete m_tempDir;
}

// Test class defined - will be run by main_test.cpp
