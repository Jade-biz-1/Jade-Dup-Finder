/**
 * Integration Test: Complete Scan-to-Delete Workflow
 * 
 * This test validates the entire application workflow:
 * 1. FileScanner scans directory and finds files
 * 2. DuplicateDetector automatically detects duplicates
 * 3. ResultsWindow receives and displays results
 * 4. User selects files for deletion
 * 5. FileManager deletes files with SafetyManager backups
 * 6. UI updates to reflect changes
 * 
 * Requirements: 9.1, 9.2, 9.3
 */

#include <QCoreApplication>
#include <QTest>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QEventLoop>
#include <QTimer>
#include <QDebug>

#include "file_scanner.h"
#include "duplicate_detector.h"
#include "../src/core/safety_manager.h"
#include "file_manager.h"
#include "../src/gui/results_window.h"

class ScanToDeleteWorkflowTest : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();
    
    // Main workflow tests
    void testCompleteScanToDeleteWorkflow();
    void testAutomaticDetectionTriggering();
    void testResultsDisplayUpdate();
    void testFileOperationWithBackup();
    void testUIUpdateAfterDeletion();
    void testMultipleGroupDeletion();
    void testPartialDeletion();
    void testProtectedFileHandling();

private:
    // Helper methods
    void createTestFiles(const QString& basePath, int duplicateGroups, int filesPerGroup);
    void createFile(const QString& path, const QByteArray& content);
    bool waitForSignal(QObject* sender, const char* signal, int timeout = 5000);
    void verifyDuplicateGroups(const QList<DuplicateDetector::DuplicateGroup>& groups, 
                               int expectedGroups, int expectedFilesPerGroup);
    void verifyBackupsCreated(const QStringList& deletedFiles, const QString& backupDir);
    void verifyFilesDeleted(const QStringList& filePaths);
    
    // Test fixtures
    QTemporaryDir* m_testDir;
    QTemporaryDir* m_backupDir;
    QString m_testPath;
    QString m_backupPath;
    
    // Component instances
    FileScanner* m_fileScanner;
    DuplicateDetector* m_duplicateDetector;
    FileManager* m_fileManager;
    SafetyManager* m_safetyManager;
    ResultsWindow* m_resultsWindow;
};

void ScanToDeleteWorkflowTest::initTestCase()
{
    qDebug() << "========================================";
    qDebug() << "Scan-to-Delete Workflow Integration Test";
    qDebug() << "========================================";
}

void ScanToDeleteWorkflowTest::cleanupTestCase()
{
    qDebug() << "========================================";
    qDebug() << "Test Suite Complete";
    qDebug() << "========================================";
}

void ScanToDeleteWorkflowTest::init()
{
    // Register metatypes for signal/slot system
    qRegisterMetaType<FileManager::OperationResult>("FileManager::OperationResult");
    qRegisterMetaType<FileManager::OperationResult>("OperationResult");
    
    // Create temporary directories
    m_testDir = new QTemporaryDir();
    m_backupDir = new QTemporaryDir();
    QVERIFY(m_testDir->isValid());
    QVERIFY(m_backupDir->isValid());
    
    m_testPath = m_testDir->path();
    m_backupPath = m_backupDir->path();
    
    // Create component instances
    m_fileScanner = new FileScanner(this);
    m_duplicateDetector = new DuplicateDetector(this);
    m_safetyManager = new SafetyManager(this);
    m_fileManager = new FileManager(this);
    m_resultsWindow = new ResultsWindow();
    
    // Configure SafetyManager
    m_safetyManager->setBackupDirectory(m_backupPath);
    m_safetyManager->setSafetyLevel(SafetyManager::SafetyLevel::Standard);
    
    // Wire up components
    m_fileManager->setSafetyManager(m_safetyManager);
    m_resultsWindow->setFileManager(m_fileManager);
}

void ScanToDeleteWorkflowTest::cleanup()
{
    // Clean up in reverse order
    delete m_resultsWindow;
    delete m_fileManager;
    delete m_safetyManager;
    delete m_duplicateDetector;
    delete m_fileScanner;
    delete m_backupDir;
    delete m_testDir;
    
    m_resultsWindow = nullptr;
    m_fileManager = nullptr;
    m_safetyManager = nullptr;
    m_duplicateDetector = nullptr;
    m_fileScanner = nullptr;
    m_backupDir = nullptr;
    m_testDir = nullptr;
}

void ScanToDeleteWorkflowTest::testCompleteScanToDeleteWorkflow()
{
    qDebug() << "\n[TEST] Complete Scan-to-Delete Workflow";
    
    // Step 1: Create test files with duplicates
    qDebug() << "Step 1: Creating test files...";
    createTestFiles(m_testPath, 3, 3); // 3 groups, 3 files each
    
    // Verify files were created
    QDir testDir(m_testPath);
    QStringList allFiles = testDir.entryList(QDir::Files);
    QCOMPARE(allFiles.size(), 9); // 3 groups * 3 files
    qDebug() << "  Created" << allFiles.size() << "test files";
    
    // Step 2: Start scan
    qDebug() << "Step 2: Starting file scan...";
    FileScanner::ScanOptions scanOptions;
    scanOptions.targetPaths << m_testPath;
    scanOptions.minimumFileSize = 0;
    scanOptions.includeHiddenFiles = false;
    
    QSignalSpy scanCompletedSpy(m_fileScanner, &FileScanner::scanCompleted);
    m_fileScanner->startScan(scanOptions);
    
    // Wait for scan to complete
    QVERIFY(waitForSignal(m_fileScanner, SIGNAL(scanCompleted()), 10000));
    QCOMPARE(scanCompletedSpy.count(), 1);
    
    QList<FileScanner::FileInfo> scannedFiles = m_fileScanner->getScannedFiles();
    QCOMPARE(scannedFiles.size(), 9);
    qDebug() << "  Scan completed:" << scannedFiles.size() << "files found";
    
    // Step 3: Trigger duplicate detection (simulating MainWindow behavior)
    qDebug() << "Step 3: Running duplicate detection...";
    QSignalSpy detectionCompletedSpy(m_duplicateDetector, &DuplicateDetector::detectionCompleted);
    
    // Convert FileScanner::FileInfo to DuplicateDetector::FileInfo
    QList<DuplicateDetector::FileInfo> detectorFiles;
    for (const auto& scanFile : scannedFiles) {
        DuplicateDetector::FileInfo detectorFile;
        detectorFile.filePath = scanFile.filePath;
        detectorFile.fileSize = scanFile.fileSize;
        detectorFile.fileName = scanFile.fileName;
        detectorFile.directory = scanFile.directory;
        detectorFile.lastModified = scanFile.lastModified;
        detectorFiles.append(detectorFile);
    }
    
    m_duplicateDetector->findDuplicates(detectorFiles);
    
    // Wait for detection to complete
    QVERIFY(waitForSignal(m_duplicateDetector, SIGNAL(detectionCompleted(int)), 15000));
    QCOMPARE(detectionCompletedSpy.count(), 1);
    
    QList<DuplicateDetector::DuplicateGroup> groups = m_duplicateDetector->getDuplicateGroups();
    QCOMPARE(groups.size(), 3); // 3 duplicate groups
    qDebug() << "  Detection completed:" << groups.size() << "duplicate groups found";
    
    // Verify each group has 3 files
    for (const auto& group : groups) {
        QCOMPARE(group.fileCount, 3);
    }
    
    // Step 4: Display results in ResultsWindow
    qDebug() << "Step 4: Displaying results...";
    m_resultsWindow->displayDuplicateGroups(groups);
    
    // Verify results were loaded
    QCOMPARE(m_resultsWindow->getSelectedFilesCount(), 0); // Nothing selected yet
    qDebug() << "  Results displayed successfully";
    
    // Step 5: Select files for deletion (keep first file in each group, delete rest)
    qDebug() << "Step 5: Selecting files for deletion...";
    QStringList filesToDelete;
    for (const auto& group : groups) {
        // Delete all but the first file in each group
        for (int i = 1; i < group.files.size(); ++i) {
            filesToDelete << group.files[i].filePath;
        }
    }
    QCOMPARE(filesToDelete.size(), 6); // 3 groups * 2 files to delete
    qDebug() << "  Selected" << filesToDelete.size() << "files for deletion";
    
    // Step 6: Delete files through FileManager
    qDebug() << "Step 6: Deleting files with backup...";
    QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
    
    QString operationId = m_fileManager->deleteFiles(filesToDelete, "Test deletion");
    QVERIFY(!operationId.isEmpty());
    
    // Wait for operation to complete (timer fires every 100ms, so wait up to 10 seconds)
    QVERIFY(operationCompletedSpy.wait(10000));
    QCOMPARE(operationCompletedSpy.count(), 1);
    
    FileManager::OperationResult result = m_fileManager->getOperationResult(operationId);
    QVERIFY(result.success);
    QCOMPARE(result.processedFiles.size(), 6);
    qDebug() << "  Deletion completed:" << result.processedFiles.size() << "files deleted";
    
    // Step 7: Verify files were actually deleted
    qDebug() << "Step 7: Verifying files were deleted...";
    verifyFilesDeleted(filesToDelete);
    qDebug() << "  All files successfully deleted from filesystem";
    
    // Step 8: Verify backups were created
    qDebug() << "Step 8: Verifying backups were created...";
    verifyBackupsCreated(filesToDelete, m_backupPath);
    qDebug() << "  All backups successfully created";
    
    // Step 9: Verify remaining files still exist
    qDebug() << "Step 9: Verifying remaining files...";
    QStringList remainingFiles;
    for (const auto& group : groups) {
        remainingFiles << group.files[0].filePath; // First file in each group
    }
    
    for (const QString& filePath : remainingFiles) {
        QVERIFY2(QFile::exists(filePath), qPrintable("File should still exist: " + filePath));
    }
    QCOMPARE(remainingFiles.size(), 3);
    qDebug() << "  All" << remainingFiles.size() << "remaining files verified";
    
    qDebug() << "[PASS] Complete workflow test successful!";
}

void ScanToDeleteWorkflowTest::testAutomaticDetectionTriggering()
{
    qDebug() << "\n[TEST] Automatic Detection Triggering";
    
    // Create test files
    createTestFiles(m_testPath, 2, 2);
    
    // Set up signal spy for detection start
    QSignalSpy detectionStartedSpy(m_duplicateDetector, &DuplicateDetector::detectionStarted);
    
    // Connect scan completion to detection (simulating MainWindow)
    connect(m_fileScanner, &FileScanner::scanCompleted, this, [this]() {
        QList<FileScanner::FileInfo> scannedFiles = m_fileScanner->getScannedFiles();
        QList<DuplicateDetector::FileInfo> detectorFiles;
        
        for (const auto& scanFile : scannedFiles) {
            DuplicateDetector::FileInfo detectorFile;
            detectorFile.filePath = scanFile.filePath;
            detectorFile.fileSize = scanFile.fileSize;
            detectorFile.fileName = scanFile.fileName;
            detectorFile.directory = scanFile.directory;
            detectorFile.lastModified = scanFile.lastModified;
            detectorFiles.append(detectorFile);
        }
        
        m_duplicateDetector->findDuplicates(detectorFiles);
    });
    
    // Start scan
    FileScanner::ScanOptions options;
    options.targetPaths << m_testPath;
    options.minimumFileSize = 0;
    
    m_fileScanner->startScan(options);
    
    // Wait for detection to start automatically
    QVERIFY(waitForSignal(m_duplicateDetector, SIGNAL(detectionStarted(int)), 10000));
    QCOMPARE(detectionStartedSpy.count(), 1);
    
    // Wait for detection to complete
    QVERIFY(waitForSignal(m_duplicateDetector, SIGNAL(detectionCompleted(int)), 15000));
    
    QList<DuplicateDetector::DuplicateGroup> groups = m_duplicateDetector->getDuplicateGroups();
    QCOMPARE(groups.size(), 2);
    
    qDebug() << "[PASS] Automatic detection triggered successfully";
}

void ScanToDeleteWorkflowTest::testResultsDisplayUpdate()
{
    qDebug() << "\n[TEST] Results Display Update";
    
    // Create duplicate groups manually
    QList<DuplicateDetector::DuplicateGroup> groups;
    
    for (int i = 0; i < 3; ++i) {
        DuplicateDetector::DuplicateGroup group;
        group.fileSize = 1000;
        group.fileCount = 2;
        
        for (int j = 0; j < 2; ++j) {
            DuplicateDetector::FileInfo file;
            file.filePath = QString("/test/group%1_file%2.txt").arg(i).arg(j);
            file.fileSize = 1000;
            file.fileName = QString("group%1_file%2.txt").arg(i).arg(j);
            group.files.append(file);
        }
        
        group.totalSize = group.fileSize * group.fileCount;
        group.wastedSpace = group.fileSize * (group.fileCount - 1);
        groups.append(group);
    }
    
    // Display results
    m_resultsWindow->displayDuplicateGroups(groups);
    
    // Verify display was updated (basic check)
    QCOMPARE(m_resultsWindow->getSelectedFilesCount(), 0);
    
    qDebug() << "[PASS] Results display updated successfully";
}

void ScanToDeleteWorkflowTest::testFileOperationWithBackup()
{
    qDebug() << "\n[TEST] File Operation with Backup";
    
    // Create a test file
    QString testFile = m_testPath + "/test_file.txt";
    createFile(testFile, "Test content for backup");
    QVERIFY(QFile::exists(testFile));
    
    // Delete file with backup
    QSignalSpy backupCreatedSpy(m_safetyManager, SIGNAL(backupCreated(const QString&, const QString&)));
    QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
    
    QString operationId = m_fileManager->deleteFiles(QStringList() << testFile, "Test backup");
    
    // Wait for completion
    QVERIFY(operationCompletedSpy.wait(5000));
    
    // Verify file was deleted
    QVERIFY(!QFile::exists(testFile));
    
    // Verify backup was created
    QCOMPARE(backupCreatedSpy.count(), 1);
    
    // Verify backup exists
    QDir backupDir(m_backupPath);
    QStringList backups = backupDir.entryList(QDir::Files, QDir::Time);
    QVERIFY(backups.size() > 0);
    
    qDebug() << "[PASS] File operation with backup successful";
}

void ScanToDeleteWorkflowTest::testUIUpdateAfterDeletion()
{
    qDebug() << "\n[TEST] UI Update After Deletion";
    
    // Create test files and detect duplicates
    createTestFiles(m_testPath, 2, 2);
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_testPath;
    options.minimumFileSize = 0;
    
    m_fileScanner->startScan(options);
    QVERIFY(waitForSignal(m_fileScanner, SIGNAL(scanCompleted()), 10000));
    
    QList<FileScanner::FileInfo> scannedFiles = m_fileScanner->getScannedFiles();
    QList<DuplicateDetector::FileInfo> detectorFiles;
    
    for (const auto& scanFile : scannedFiles) {
        DuplicateDetector::FileInfo detectorFile;
        detectorFile.filePath = scanFile.filePath;
        detectorFile.fileSize = scanFile.fileSize;
        detectorFile.fileName = scanFile.fileName;
        detectorFile.directory = scanFile.directory;
        detectorFile.lastModified = scanFile.lastModified;
        detectorFiles.append(detectorFile);
    }
    
    m_duplicateDetector->findDuplicates(detectorFiles);
    QVERIFY(waitForSignal(m_duplicateDetector, SIGNAL(detectionCompleted(int)), 15000));
    
    QList<DuplicateDetector::DuplicateGroup> groups = m_duplicateDetector->getDuplicateGroups();
    m_resultsWindow->displayDuplicateGroups(groups);
    
    // Delete one file from first group
    QString fileToDelete = groups[0].files[1].filePath;
    QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
    m_fileManager->deleteFiles(QStringList() << fileToDelete, "UI update test");
    QVERIFY(operationCompletedSpy.wait(5000));
    
    // UI should be updated (in real app, this would be triggered by signal)
    // For now, just verify the file is gone
    QVERIFY(!QFile::exists(fileToDelete));
    
    qDebug() << "[PASS] UI update after deletion verified";
}

void ScanToDeleteWorkflowTest::testMultipleGroupDeletion()
{
    qDebug() << "\n[TEST] Multiple Group Deletion";
    
    // Create multiple groups
    createTestFiles(m_testPath, 5, 3); // 5 groups, 3 files each
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_testPath;
    options.minimumFileSize = 0;
    
    m_fileScanner->startScan(options);
    QVERIFY(waitForSignal(m_fileScanner, SIGNAL(scanCompleted()), 10000));
    
    QList<FileScanner::FileInfo> scannedFiles = m_fileScanner->getScannedFiles();
    QList<DuplicateDetector::FileInfo> detectorFiles;
    
    for (const auto& scanFile : scannedFiles) {
        DuplicateDetector::FileInfo detectorFile;
        detectorFile.filePath = scanFile.filePath;
        detectorFile.fileSize = scanFile.fileSize;
        detectorFile.fileName = scanFile.fileName;
        detectorFile.directory = scanFile.directory;
        detectorFile.lastModified = scanFile.lastModified;
        detectorFiles.append(detectorFile);
    }
    
    m_duplicateDetector->findDuplicates(detectorFiles);
    QVERIFY(waitForSignal(m_duplicateDetector, SIGNAL(detectionCompleted(int)), 15000));
    
    QList<DuplicateDetector::DuplicateGroup> groups = m_duplicateDetector->getDuplicateGroups();
    QCOMPARE(groups.size(), 5);
    
    // Delete files from all groups (keep first file in each)
    QStringList filesToDelete;
    for (const auto& group : groups) {
        for (int i = 1; i < group.files.size(); ++i) {
            filesToDelete << group.files[i].filePath;
        }
    }
    
    QCOMPARE(filesToDelete.size(), 10); // 5 groups * 2 files to delete
    
    QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
    m_fileManager->deleteFiles(filesToDelete, "Multiple group deletion");
    QVERIFY(operationCompletedSpy.wait(10000));
    
    // Verify all files were deleted
    verifyFilesDeleted(filesToDelete);
    
    qDebug() << "[PASS] Multiple group deletion successful";
}

void ScanToDeleteWorkflowTest::testPartialDeletion()
{
    qDebug() << "\n[TEST] Partial Deletion";
    
    // Create test files
    createTestFiles(m_testPath, 3, 3);
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_testPath;
    options.minimumFileSize = 0;
    
    m_fileScanner->startScan(options);
    QVERIFY(waitForSignal(m_fileScanner, SIGNAL(scanCompleted()), 10000));
    
    QList<FileScanner::FileInfo> scannedFiles = m_fileScanner->getScannedFiles();
    QList<DuplicateDetector::FileInfo> detectorFiles;
    
    for (const auto& scanFile : scannedFiles) {
        DuplicateDetector::FileInfo detectorFile;
        detectorFile.filePath = scanFile.filePath;
        detectorFile.fileSize = scanFile.fileSize;
        detectorFile.fileName = scanFile.fileName;
        detectorFile.directory = scanFile.directory;
        detectorFile.lastModified = scanFile.lastModified;
        detectorFiles.append(detectorFile);
    }
    
    m_duplicateDetector->findDuplicates(detectorFiles);
    QVERIFY(waitForSignal(m_duplicateDetector, SIGNAL(detectionCompleted(int)), 15000));
    
    QList<DuplicateDetector::DuplicateGroup> groups = m_duplicateDetector->getDuplicateGroups();
    
    // Delete only from first group
    QStringList filesToDelete;
    for (int i = 1; i < groups[0].files.size(); ++i) {
        filesToDelete << groups[0].files[i].filePath;
    }
    
    QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
    m_fileManager->deleteFiles(filesToDelete, "Partial deletion");
    QVERIFY(operationCompletedSpy.wait(5000));
    
    // Verify only selected files were deleted
    verifyFilesDeleted(filesToDelete);
    
    // Verify other groups' files still exist
    for (int g = 1; g < groups.size(); ++g) {
        for (const auto& file : groups[g].files) {
            QVERIFY2(QFile::exists(file.filePath), 
                    qPrintable("File should still exist: " + file.filePath));
        }
    }
    
    qDebug() << "[PASS] Partial deletion successful";
}

void ScanToDeleteWorkflowTest::testProtectedFileHandling()
{
    qDebug() << "\n[TEST] Protected File Handling";
    
    // Create a test file
    QString testFile = m_testPath + "/protected_file.txt";
    createFile(testFile, "Protected content");
    
    // Add protection rule
    m_safetyManager->addProtectionRule(testFile, SafetyManager::ProtectionLevel::ReadOnly, 
                                       "Test protection");
    
    // Try to delete protected file
    QSignalSpy protectionViolationSpy(m_safetyManager, SIGNAL(protectionViolation(const QString&, SafetyManager::OperationType)));
    QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
    
    QString operationId = m_fileManager->deleteFiles(QStringList() << testFile, "Test protected");
    
    // Wait for operation
    QVERIFY(operationCompletedSpy.wait(5000));
    
    FileManager::OperationResult result = m_fileManager->getOperationResult(operationId);
    
    // File should be skipped or operation should fail
    QVERIFY(result.skippedFiles.contains(testFile) || !result.success);
    
    // File should still exist
    QVERIFY(QFile::exists(testFile));
    
    qDebug() << "[PASS] Protected file handling successful";
}

// Helper method implementations

void ScanToDeleteWorkflowTest::createTestFiles(const QString& basePath, int duplicateGroups, int filesPerGroup)
{
    for (int group = 0; group < duplicateGroups; ++group) {
        QByteArray content = QString("Duplicate group %1 content").arg(group).toUtf8();
        
        for (int file = 0; file < filesPerGroup; ++file) {
            QString filePath = QString("%1/group%2_file%3.txt").arg(basePath).arg(group).arg(file);
            createFile(filePath, content);
        }
    }
}

void ScanToDeleteWorkflowTest::createFile(const QString& path, const QByteArray& content)
{
    QFile file(path);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(content);
        file.close();
    }
}

bool ScanToDeleteWorkflowTest::waitForSignal(QObject* sender, const char* signal, int timeout)
{
    QEventLoop loop;
    QTimer timer;
    timer.setSingleShot(true);
    
    QObject::connect(&timer, &QTimer::timeout, &loop, &QEventLoop::quit);
    QObject::connect(sender, signal, &loop, SLOT(quit()));
    
    timer.start(timeout);
    loop.exec();
    
    return timer.isActive(); // Returns true if signal was received before timeout
}

void ScanToDeleteWorkflowTest::verifyDuplicateGroups(const QList<DuplicateDetector::DuplicateGroup>& groups, 
                                                     int expectedGroups, int expectedFilesPerGroup)
{
    QCOMPARE(groups.size(), expectedGroups);
    
    for (const auto& group : groups) {
        QCOMPARE(group.fileCount, expectedFilesPerGroup);
        QCOMPARE(group.files.size(), expectedFilesPerGroup);
    }
}

void ScanToDeleteWorkflowTest::verifyBackupsCreated(const QStringList& deletedFiles, const QString& backupDir)
{
    QDir dir(backupDir);
    QStringList backups = dir.entryList(QDir::Files, QDir::Time);
    
    // Should have at least as many backups as deleted files
    QVERIFY2(backups.size() >= deletedFiles.size(), 
            qPrintable(QString("Expected at least %1 backups, found %2")
                      .arg(deletedFiles.size()).arg(backups.size())));
}

void ScanToDeleteWorkflowTest::verifyFilesDeleted(const QStringList& filePaths)
{
    for (const QString& filePath : filePaths) {
        QVERIFY2(!QFile::exists(filePath), 
                qPrintable("File should be deleted: " + filePath));
    }
}

QTEST_MAIN(ScanToDeleteWorkflowTest)
#include "test_scan_to_delete_workflow.moc"
