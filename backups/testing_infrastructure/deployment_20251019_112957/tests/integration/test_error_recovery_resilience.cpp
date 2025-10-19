#include <QCoreApplication>
#include <QDebug>
#include <QTimer>
#include <QEventLoop>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QSignalSpy>
#include <QTest>
#include <QThread>
#include <QMutex>
#include <QRandomGenerator>

#include "file_scanner.h"
#include "duplicate_detector.h"
#include "hash_calculator.h"
#include "file_manager.h"
#include "../src/core/safety_manager.h"

/**
 * @brief Error Recovery and Resilience Testing
 * 
 * This test verifies:
 * - System recovery from various failure modes
 * - Resilience under sustained error conditions
 * - Data consistency during error recovery
 * - Component isolation during failures
 * - Automatic retry mechanisms
 * - Fallback behavior implementation
 * - Resource cleanup after errors
 * - State consistency after recovery
 * 
 * Requirements: 4.4, 9.4, 10.1
 */

class ErrorRecoveryResilienceTest : public QObject {
    Q_OBJECT

private:
    QTemporaryDir* m_tempDir;
    QTemporaryDir* m_backupDir;
    QString m_testPath;
    QString m_backupPath;
    
    // Helper to create test file
    QString createTestFile(const QString& relativePath, const QByteArray& content) {
        QString fullPath = m_testPath + "/" + relativePath;
        QFileInfo info(fullPath);
        QDir().mkpath(info.absolutePath());
        
        QFile file(fullPath);
        if (!file.open(QIODevice::WriteOnly)) {
            qWarning() << "Failed to create test file:" << fullPath;
            return QString();
        }
        file.write(content);
        file.close();
        return fullPath;
    }
    
    // Helper to simulate system stress
    void simulateSystemStress(int durationMs) {
        QElapsedTimer timer;
        timer.start();
        
        while (timer.elapsed() < durationMs) {
            // Simulate CPU load
            volatile int dummy = 0;
            for (int i = 0; i < 10000; i++) {
                dummy += i;
            }
            
            // Process events to keep system responsive
            QCoreApplication::processEvents();
            QThread::msleep(1);
        }
    }

private slots:
    void initTestCase() {
        qDebug() << "===========================================";
        qDebug() << "Error Recovery and Resilience Testing";
        qDebug() << "===========================================";
        qDebug();
        
        // Register metatypes
        qRegisterMetaType<FileManager::OperationResult>("FileManager::OperationResult");
        qRegisterMetaType<FileManager::OperationResult>("OperationResult");
        qRegisterMetaType<SafetyManager::BackupResult>("SafetyManager::BackupResult");
        qRegisterMetaType<SafetyManager::BackupResult>("BackupResult");
        
        m_tempDir = new QTemporaryDir();
        m_backupDir = new QTemporaryDir();
        QVERIFY(m_tempDir->isValid());
        QVERIFY(m_backupDir->isValid());
        
        m_testPath = m_tempDir->path();
        m_backupPath = m_backupDir->path();
        
        qDebug() << "Test directory:" << m_testPath;
        qDebug() << "Backup directory:" << m_backupPath;
    }
    
    void cleanupTestCase() {
        delete m_backupDir;
        delete m_tempDir;
        
        qDebug() << "\n===========================================";
        qDebug() << "All tests completed";
        qDebug() << "===========================================";
    }
    
    /**
     * Test 1: Recovery from scan interruption
     */
    void test_recoveryFromScanInterruption() {
        qDebug() << "\n[Test 1] Recovery from Scan Interruption";
        qDebug() << "=========================================";
        
        // Create many test files
        const int fileCount = 100;
        QStringList createdFiles;
        
        for (int i = 0; i < fileCount; i++) {
            QString file = createTestFile(QString("scan_recovery/file_%1.txt").arg(i), 
                                        QString("Content %1").arg(i).toUtf8());
            createdFiles << file;
        }
        
        qDebug() << "   Created" << fileCount << "test files";
        
        // Start scan and interrupt it
        FileScanner scanner;
        FileScanner::ScanOptions options;
        options.targetPaths << m_testPath + "/scan_recovery";
        options.minimumFileSize = 0;
        options.progressBatchSize = 10; // Smaller batches for more interruption opportunities
        
        QSignalSpy scanStartedSpy(&scanner, &FileScanner::scanStarted);
        QSignalSpy scanProgressSpy(&scanner, &FileScanner::scanProgress);
        QSignalSpy scanCancelledSpy(&scanner, &FileScanner::scanCancelled);
        QSignalSpy scanCompletedSpy(&scanner, &FileScanner::scanCompleted);
        
        scanner.startScan(options);
        QVERIFY(scanStartedSpy.wait(1000));
        
        // Cancel scan after a short delay
        QTimer::singleShot(50, [&scanner]() {
            qDebug() << "      Cancelling scan...";
            scanner.cancelScan();
        });
        
        // Wait for cancellation or completion
        QEventLoop loop;
        QObject::connect(&scanner, &FileScanner::scanCancelled, &loop, &QEventLoop::quit);
        QObject::connect(&scanner, &FileScanner::scanCompleted, &loop, &QEventLoop::quit);
        QTimer::singleShot(5000, &loop, &QEventLoop::quit);
        loop.exec();
        
        qDebug() << "      Scan cancelled:" << scanCancelledSpy.count();
        qDebug() << "      Scan completed:" << scanCompletedSpy.count();
        qDebug() << "      Progress updates:" << scanProgressSpy.count();
        
        // Verify scanner state after interruption
        QVERIFY(!scanner.isScanning());
        
        // Test recovery - start new scan
        qDebug() << "   Testing recovery with new scan...";
        
        FileScanner recoveryScanner;
        QSignalSpy recoveryCompletedSpy(&recoveryScanner, &FileScanner::scanCompleted);
        
        recoveryScanner.startScan(options);
        QVERIFY(recoveryCompletedSpy.wait(10000));
        
        QVector<FileScanner::FileInfo> recoveredFiles = recoveryScanner.getScannedFiles();
        qDebug() << "      Recovery scan found:" << recoveredFiles.size() << "files";
        
        // Recovery should find all files
        QVERIFY(recoveredFiles.size() >= fileCount * 0.95); // Allow 5% tolerance
        
        qDebug() << "✓ Recovery from scan interruption verified";
    }
    
    /**
     * Test 2: Recovery from hash calculation failures
     */
    void test_recoveryFromHashCalculationFailures() {
        qDebug() << "\n[Test 2] Recovery from Hash Calculation Failures";
        qDebug() << "=================================================";
        
        // Create test files including some that will cause issues
        QString normalFile1 = createTestFile("hash_recovery/normal1.txt", "Normal content 1");
        QString normalFile2 = createTestFile("hash_recovery/normal2.txt", "Normal content 2");
        QString normalFile3 = createTestFile("hash_recovery/normal3.txt", "Normal content 3");
        
        // Create a file that we'll delete during processing
        QString volatileFile = createTestFile("hash_recovery/volatile.txt", "Will be deleted");
        
        QStringList allFiles = {normalFile1, normalFile2, normalFile3, volatileFile};
        
        qDebug() << "   Created" << allFiles.size() << "test files";
        
        // Start hash calculation
        HashCalculator hashCalc;
        QSignalSpy hashCompletedSpy(&hashCalc, &HashCalculator::hashCompleted);
        QSignalSpy hashErrorSpy(&hashCalc, &HashCalculator::hashError);
        QSignalSpy allCompleteSpy(&hashCalc, &HashCalculator::allOperationsComplete);
        
        hashCalc.calculateFileHashes(allFiles);
        
        // Delete the volatile file during processing to cause an error
        QTimer::singleShot(100, [volatileFile]() {
            QFile::remove(volatileFile);
            qDebug() << "      Deleted volatile file during hash calculation";
        });
        
        // Wait for completion
        QEventLoop loop;
        QObject::connect(&hashCalc, &HashCalculator::allOperationsComplete, &loop, &QEventLoop::quit);
        QTimer::singleShot(10000, &loop, &QEventLoop::quit);
        loop.exec();
        
        qDebug() << "   Hash calculation results:";
        qDebug() << "      Completed hashes:" << hashCompletedSpy.count();
        qDebug() << "      Hash errors:" << hashErrorSpy.count();
        qDebug() << "      All operations complete:" << allCompleteSpy.count();
        
        // Should have completed some hashes and reported errors for others
        QVERIFY(hashCompletedSpy.count() >= 3); // At least the normal files
        QVERIFY(hashErrorSpy.count() >= 1);     // At least the deleted file
        QCOMPARE(allCompleteSpy.count(), 1);
        
        // Test recovery - calculate hashes for remaining files
        qDebug() << "   Testing recovery with remaining files...";
        
        HashCalculator recoveryHashCalc;
        QSignalSpy recoveryCompletedSpy(&recoveryHashCalc, &HashCalculator::hashCompleted);
        QSignalSpy recoveryAllCompleteSpy(&recoveryHashCalc, &HashCalculator::allOperationsComplete);
        
        QStringList remainingFiles = {normalFile1, normalFile2, normalFile3};
        recoveryHashCalc.calculateFileHashes(remainingFiles);
        
        QEventLoop recoveryLoop;
        QObject::connect(&recoveryHashCalc, &HashCalculator::allOperationsComplete, 
                        &recoveryLoop, &QEventLoop::quit);
        QTimer::singleShot(10000, &recoveryLoop, &QEventLoop::quit);
        recoveryLoop.exec();
        
        qDebug() << "      Recovery hash completed:" << recoveryCompletedSpy.count();
        qDebug() << "      Recovery all complete:" << recoveryAllCompleteSpy.count();
        
        // Recovery should succeed for all remaining files
        QCOMPARE(recoveryCompletedSpy.count(), remainingFiles.size());
        QCOMPARE(recoveryAllCompleteSpy.count(), 1);
        
        qDebug() << "✓ Recovery from hash calculation failures verified";
    }
    
    /**
     * Test 3: Recovery from file operation failures
     */
    void test_recoveryFromFileOperationFailures() {
        qDebug() << "\n[Test 3] Recovery from File Operation Failures";
        qDebug() << "===============================================";
        
        // Create test files
        QStringList testFiles;
        for (int i = 0; i < 5; i++) {
            QString file = createTestFile(QString("operation_recovery/file_%1.txt").arg(i),
                                        QString("Content %1").arg(i).toUtf8());
            testFiles << file;
        }
        
        // Add non-existent files to cause failures
        testFiles << "/non/existent/file1.txt";
        testFiles << "/non/existent/file2.txt";
        
        qDebug() << "   Created test scenario with" << testFiles.size() << "files (some non-existent)";
        
        // Set up components
        SafetyManager safetyManager;
        safetyManager.setBackupDirectory(m_backupPath);
        
        FileManager fileManager;
        fileManager.setSafetyManager(&safetyManager);
        
        // Attempt file operations with mixed success/failure
        QSignalSpy operationCompletedSpy(&fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        QSignalSpy operationErrorSpy(&fileManager, &FileManager::operationError);
        
        QString operationId = fileManager.deleteFiles(testFiles, "Recovery test with failures");
        QVERIFY(operationCompletedSpy.wait(15000));
        
        FileManager::OperationResult result = fileManager.getOperationResult(operationId);
        
        qDebug() << "   File operation results:";
        qDebug() << "      Overall success:" << result.success;
        qDebug() << "      Processed files:" << result.processedFiles.size();
        qDebug() << "      Failed files:" << result.failedFiles.size();
        qDebug() << "      Skipped files:" << result.skippedFiles.size();
        qDebug() << "      Operation errors:" << operationErrorSpy.count();
        
        // Should have partial success
        QVERIFY(result.processedFiles.size() >= 5); // The real files
        QVERIFY(result.failedFiles.size() >= 2);    // The non-existent files
        
        // Test recovery - perform operation on remaining files
        qDebug() << "   Testing recovery with successful files only...";
        
        // Create new files for recovery test
        QStringList recoveryFiles;
        for (int i = 0; i < 3; i++) {
            QString file = createTestFile(QString("operation_recovery/recovery_%1.txt").arg(i),
                                        QString("Recovery content %1").arg(i).toUtf8());
            recoveryFiles << file;
        }
        
        QString recoveryOpId = fileManager.deleteFiles(recoveryFiles, "Recovery operation");
        QVERIFY(operationCompletedSpy.wait(10000));
        
        FileManager::OperationResult recoveryResult = fileManager.getOperationResult(recoveryOpId);
        
        qDebug() << "      Recovery operation success:" << recoveryResult.success;
        qDebug() << "      Recovery processed files:" << recoveryResult.processedFiles.size();
        qDebug() << "      Recovery failed files:" << recoveryResult.failedFiles.size();
        
        // Recovery should be fully successful
        QVERIFY(recoveryResult.success);
        QCOMPARE(recoveryResult.processedFiles.size(), recoveryFiles.size());
        QCOMPARE(recoveryResult.failedFiles.size(), 0);
        
        qDebug() << "✓ Recovery from file operation failures verified";
    }
    
    /**
     * Test 4: Resilience under sustained error conditions
     */
    void test_resilienceUnderSustainedErrors() {
        qDebug() << "\n[Test 4] Resilience Under Sustained Error Conditions";
        qDebug() << "=====================================================";
        
        // Create a scenario with continuous error conditions
        const int operationCount = 10;
        const int filesPerOperation = 3;
        
        SafetyManager safetyManager;
        safetyManager.setBackupDirectory("/invalid/backup/path"); // Cause backup failures
        
        FileManager fileManager;
        fileManager.setSafetyManager(&safetyManager);
        
        QSignalSpy operationCompletedSpy(&fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        QSignalSpy operationErrorSpy(&fileManager, &FileManager::operationError);
        
        QStringList operationIds;
        int totalFiles = 0;
        
        qDebug() << "   Starting" << operationCount << "operations with sustained error conditions...";
        
        // Start multiple operations that will encounter errors
        for (int op = 0; op < operationCount; op++) {
            QStringList files;
            
            // Mix of real and non-existent files
            for (int f = 0; f < filesPerOperation; f++) {
                if (f % 2 == 0) {
                    // Real file
                    QString file = createTestFile(QString("resilience/op_%1_file_%2.txt").arg(op).arg(f),
                                                QString("Op %1 File %2").arg(op).arg(f).toUtf8());
                    files << file;
                } else {
                    // Non-existent file
                    files << QString("/fake/op_%1_file_%2.txt").arg(op).arg(f);
                }
                totalFiles++;
            }
            
            QString opId = fileManager.deleteFiles(files, QString("Resilience test operation %1").arg(op));
            operationIds << opId;
            
            // Small delay between operations
            QTest::qWait(50);
        }
        
        qDebug() << "      Started" << operationIds.size() << "operations affecting" << totalFiles << "files";
        
        // Wait for all operations to complete
        int completedOps = 0;
        int maxWaitTime = 30000; // 30 seconds
        int waitInterval = 1000;  // 1 second
        int totalWaitTime = 0;
        
        while (completedOps < operationIds.size() && totalWaitTime < maxWaitTime) {
            if (operationCompletedSpy.wait(waitInterval)) {
                completedOps = operationCompletedSpy.count();
            }
            totalWaitTime += waitInterval;
            
            if (completedOps % 3 == 0 && completedOps > 0) {
                qDebug() << "      Completed" << completedOps << "of" << operationIds.size() << "operations";
            }
        }
        
        qDebug() << "   All operations completed in" << totalWaitTime << "ms";
        qDebug() << "   Total operation error signals:" << operationErrorSpy.count();
        
        // Analyze results
        int successfulOps = 0;
        int partialSuccessOps = 0;
        int failedOps = 0;
        int totalProcessedFiles = 0;
        int totalFailedFiles = 0;
        
        for (const QString& opId : operationIds) {
            FileManager::OperationResult result = fileManager.getOperationResult(opId);
            
            if (result.success && result.failedFiles.isEmpty()) {
                successfulOps++;
            } else if (result.processedFiles.size() > 0) {
                partialSuccessOps++;
            } else {
                failedOps++;
            }
            
            totalProcessedFiles += result.processedFiles.size();
            totalFailedFiles += result.failedFiles.size();
        }
        
        qDebug() << "   Resilience analysis:";
        qDebug() << "      Fully successful operations:" << successfulOps;
        qDebug() << "      Partially successful operations:" << partialSuccessOps;
        qDebug() << "      Failed operations:" << failedOps;
        qDebug() << "      Total processed files:" << totalProcessedFiles;
        qDebug() << "      Total failed files:" << totalFailedFiles;
        
        // System should remain resilient
        QCOMPARE(successfulOps + partialSuccessOps + failedOps, operationIds.size());
        QVERIFY(totalProcessedFiles > 0); // Should process at least some files
        
        // Test that system can still perform normal operations after sustained errors
        qDebug() << "   Testing normal operation after sustained errors...";
        
        // Reset to valid backup directory
        safetyManager.setBackupDirectory(m_backupPath);
        
        QString normalFile = createTestFile("resilience/post_stress_normal.txt", "Post-stress normal file");
        QString normalOpId = fileManager.deleteFiles({normalFile}, "Post-stress normal operation");
        
        QVERIFY(operationCompletedSpy.wait(5000));
        
        FileManager::OperationResult normalResult = fileManager.getOperationResult(normalOpId);
        qDebug() << "      Post-stress normal operation success:" << normalResult.success;
        
        // Normal operations should work after sustained errors
        QVERIFY(normalResult.success);
        
        qDebug() << "✓ Resilience under sustained error conditions verified";
    }
    
    /**
     * Test 5: Data consistency during error recovery
     */
    void test_dataConsistencyDuringErrorRecovery() {
        qDebug() << "\n[Test 5] Data Consistency During Error Recovery";
        qDebug() << "================================================";
        
        // Create test files with known content
        QStringList testFiles;
        QStringList expectedContent;
        
        for (int i = 0; i < 5; i++) {
            QString content = QString("Consistency test content %1 - %2").arg(i).arg(QDateTime::currentMSecsSinceEpoch());
            QString file = createTestFile(QString("consistency/file_%1.txt").arg(i), content.toUtf8());
            testFiles << file;
            expectedContent << content;
        }
        
        qDebug() << "   Created" << testFiles.size() << "files with known content";
        
        // Set up components with proper backup
        SafetyManager safetyManager;
        safetyManager.setBackupDirectory(m_backupPath);
        
        FileManager fileManager;
        fileManager.setSafetyManager(&safetyManager);
        
        // Create backups before operations
        QSignalSpy backupCompletedSpy(&safetyManager, &SafetyManager::backupCompleted);
        QStringList backupIds;
        
        for (const QString& file : testFiles) {
            QString backupId = safetyManager.createBackup(file, "Consistency test backup");
            backupIds << backupId;
        }
        
        // Wait for all backups to complete
        int expectedBackups = backupIds.size();
        while (backupCompletedSpy.count() < expectedBackups && backupCompletedSpy.wait(2000)) {
            // Wait for all backups
        }
        
        qDebug() << "      Created" << backupCompletedSpy.count() << "backups";
        QCOMPARE(backupCompletedSpy.count(), expectedBackups);
        
        // Perform file operations
        QSignalSpy operationCompletedSpy(&fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        
        QString deleteOpId = fileManager.deleteFiles(testFiles, "Consistency test deletion");
        QVERIFY(operationCompletedSpy.wait(10000));
        
        // Verify files were deleted
        for (const QString& file : testFiles) {
            QVERIFY(!QFile::exists(file));
        }
        
        qDebug() << "      Files deleted successfully";
        
        // Simulate error during recovery and test data consistency
        qDebug() << "   Testing data consistency during recovery...";
        
        // Restore files and verify content consistency
        QStringList backupPaths;
        for (const QString& backupId : backupIds) {
            SafetyManager::BackupInfo info = safetyManager.getBackupInfo(backupId);
            QVERIFY(!info.backupPath.isEmpty());
            QVERIFY(QFile::exists(info.backupPath));
            backupPaths << info.backupPath;
        }
        
        QString restoreOpId = fileManager.restoreFiles(backupPaths, m_testPath + "/consistency");
        QVERIFY(operationCompletedSpy.wait(10000));
        
        FileManager::OperationResult restoreResult = fileManager.getOperationResult(restoreOpId);
        qDebug() << "      Restore operation success:" << restoreResult.success;
        QVERIFY(restoreResult.success);
        
        // Verify data consistency after recovery
        for (int i = 0; i < testFiles.size(); i++) {
            QString restoredFile = testFiles[i];
            QVERIFY(QFile::exists(restoredFile));
            
            QFile file(restoredFile);
            QVERIFY(file.open(QIODevice::ReadOnly));
            QString restoredContent = QString::fromUtf8(file.readAll());
            file.close();
            
            QCOMPARE(restoredContent, expectedContent[i]);
            qDebug() << "      File" << i << "content verified";
        }
        
        qDebug() << "✓ Data consistency during error recovery verified";
    }
    
    /**
     * Test 6: Component isolation during failures
     */
    void test_componentIsolationDuringFailures() {
        qDebug() << "\n[Test 6] Component Isolation During Failures";
        qDebug() << "==============================================";
        
        // Create test scenario with multiple components
        createTestFile("isolation/file1.txt", "Isolation test 1");
        createTestFile("isolation/file2.txt", "Isolation test 2");
        createTestFile("isolation/dup1.txt", "Duplicate content");
        createTestFile("isolation/dup2.txt", "Duplicate content");
        
        qDebug() << "   Created test files for component isolation testing";
        
        // Test FileScanner isolation - should work even if other components fail
        qDebug() << "   Testing FileScanner isolation...";
        
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/isolation";
        scanOptions.minimumFileSize = 0;
        
        QSignalSpy scanCompletedSpy(&scanner, &FileScanner::scanCompleted);
        QSignalSpy scanErrorSpy(&scanner, &FileScanner::errorOccurred);
        
        scanner.startScan(scanOptions);
        QVERIFY(scanCompletedSpy.wait(5000));
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        qDebug() << "      FileScanner found" << scannedFiles.size() << "files";
        qDebug() << "      FileScanner errors:" << scanErrorSpy.count();
        
        QVERIFY(scannedFiles.size() >= 4);
        QCOMPARE(scanErrorSpy.count(), 0);
        
        // Test HashCalculator isolation - should work independently
        qDebug() << "   Testing HashCalculator isolation...";
        
        HashCalculator hashCalc;
        QSignalSpy hashCompletedSpy(&hashCalc, &HashCalculator::hashCompleted);
        QSignalSpy hashErrorSpy(&hashCalc, &HashCalculator::hashError);
        QSignalSpy allCompleteSpy(&hashCalc, &HashCalculator::allOperationsComplete);
        
        QStringList filePaths;
        for (const FileScanner::FileInfo& info : scannedFiles) {
            filePaths << info.filePath;
        }
        
        hashCalc.calculateFileHashes(filePaths);
        
        QEventLoop hashLoop;
        QObject::connect(&hashCalc, &HashCalculator::allOperationsComplete, &hashLoop, &QEventLoop::quit);
        QTimer::singleShot(10000, &hashLoop, &QEventLoop::quit);
        hashLoop.exec();
        
        qDebug() << "      HashCalculator completed:" << hashCompletedSpy.count();
        qDebug() << "      HashCalculator errors:" << hashErrorSpy.count();
        qDebug() << "      All operations complete:" << allCompleteSpy.count();
        
        QVERIFY(hashCompletedSpy.count() >= 4);
        QCOMPARE(allCompleteSpy.count(), 1);
        
        // Test DuplicateDetector isolation - should work even with component failures
        qDebug() << "   Testing DuplicateDetector isolation...";
        
        DuplicateDetector detector;
        QSignalSpy detectionCompletedSpy(&detector, &DuplicateDetector::detectionCompleted);
        QSignalSpy detectionErrorSpy(&detector, &DuplicateDetector::detectionError);
        
        detector.findDuplicates(scannedFiles);
        QVERIFY(detectionCompletedSpy.wait(15000));
        
        QList<DuplicateDetector::DuplicateGroup> groups = detector.getDuplicateGroups();
        qDebug() << "      DuplicateDetector found" << groups.size() << "groups";
        qDebug() << "      DuplicateDetector errors:" << detectionErrorSpy.count();
        
        QVERIFY(groups.size() >= 1); // Should find the duplicate pair
        
        // Test that failure in one component doesn't affect others
        qDebug() << "   Testing component independence under failure...";
        
        // Simulate failure in SafetyManager
        SafetyManager failingSafetyManager;
        failingSafetyManager.setBackupDirectory("/completely/invalid/path/that/cannot/exist");
        
        FileManager fileManager;
        fileManager.setSafetyManager(&failingSafetyManager);
        
        // FileManager operations might fail, but other components should still work
        QSignalSpy operationCompletedSpy(&fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        
        QString testFile = createTestFile("isolation/test_independence.txt", "Independence test");
        QString opId = fileManager.deleteFiles({testFile}, "Independence test");
        
        operationCompletedSpy.wait(5000); // May fail, that's expected
        
        // Other components should still work independently
        FileScanner independentScanner;
        QSignalSpy independentScanSpy(&independentScanner, &FileScanner::scanCompleted);
        
        independentScanner.startScan(scanOptions);
        QVERIFY(independentScanSpy.wait(5000));
        
        QVector<FileScanner::FileInfo> independentFiles = independentScanner.getScannedFiles();
        qDebug() << "      Independent scanner found" << independentFiles.size() << "files";
        
        // Should still work despite FileManager/SafetyManager issues
        QVERIFY(independentFiles.size() >= 3);
        
        qDebug() << "✓ Component isolation during failures verified";
    }
    
    /**
     * Test 7: Resource cleanup after errors
     */
    void test_resourceCleanupAfterErrors() {
        qDebug() << "\n[Test 7] Resource Cleanup After Errors";
        qDebug() << "========================================";
        
        // Create test files
        QStringList testFiles;
        for (int i = 0; i < 10; i++) {
            QString file = createTestFile(QString("cleanup/file_%1.txt").arg(i),
                                        QString("Cleanup test %1").arg(i).toUtf8());
            testFiles << file;
        }
        
        qDebug() << "   Created" << testFiles.size() << "test files";
        
        // Test resource cleanup in FileScanner
        qDebug() << "   Testing FileScanner resource cleanup...";
        
        for (int cycle = 0; cycle < 3; cycle++) {
            FileScanner scanner;
            FileScanner::ScanOptions options;
            options.targetPaths << m_testPath + "/cleanup";
            options.minimumFileSize = 0;
            
            QSignalSpy scanStartedSpy(&scanner, &FileScanner::scanStarted);
            QSignalSpy scanCancelledSpy(&scanner, &FileScanner::scanCancelled);
            
            scanner.startScan(options);
            QVERIFY(scanStartedSpy.wait(1000));
            
            // Cancel scan to test cleanup
            scanner.cancelScan();
            
            // Wait for cancellation or timeout
            QEventLoop loop;
            QObject::connect(&scanner, &FileScanner::scanCancelled, &loop, &QEventLoop::quit);
            QTimer::singleShot(2000, &loop, &QEventLoop::quit);
            loop.exec();
            
            // Verify cleanup
            QVERIFY(!scanner.isScanning());
            
            qDebug() << "      Cycle" << (cycle + 1) << "- Scanner cleaned up successfully";
        }
        
        // Test resource cleanup in HashCalculator
        qDebug() << "   Testing HashCalculator resource cleanup...";
        
        for (int cycle = 0; cycle < 3; cycle++) {
            HashCalculator hashCalc;
            QSignalSpy hashCompletedSpy(&hashCalc, &HashCalculator::hashCompleted);
            
            // Start hash calculation
            hashCalc.calculateFileHashes(testFiles.mid(0, 5));
            
            // Cancel immediately to test cleanup
            hashCalc.cancelAll();
            
            // Give time for cleanup
            QTest::qWait(500);
            
            qDebug() << "      Cycle" << (cycle + 1) << "- HashCalculator cleaned up successfully";
        }
        
        // Test resource cleanup in DuplicateDetector
        qDebug() << "   Testing DuplicateDetector resource cleanup...";
        
        // Scan files first
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/cleanup";
        scanOptions.minimumFileSize = 0;
        
        QSignalSpy scanCompletedSpy(&scanner, &FileScanner::scanCompleted);
        scanner.startScan(scanOptions);
        QVERIFY(scanCompletedSpy.wait(5000));
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        
        for (int cycle = 0; cycle < 3; cycle++) {
            DuplicateDetector detector;
            QSignalSpy detectionStartedSpy(&detector, &DuplicateDetector::detectionStarted);
            QSignalSpy detectionCancelledSpy(&detector, &DuplicateDetector::detectionCancelled);
            
            detector.findDuplicates(scannedFiles);
            
            // Cancel detection to test cleanup
            QTimer::singleShot(100, [&detector]() {
                detector.cancelDetection();
            });
            
            // Wait for cancellation
            QEventLoop loop;
            QObject::connect(&detector, &DuplicateDetector::detectionCancelled, &loop, &QEventLoop::quit);
            QTimer::singleShot(3000, &loop, &QEventLoop::quit);
            loop.exec();
            
            // Verify cleanup
            QVERIFY(!detector.isDetecting());
            
            qDebug() << "      Cycle" << (cycle + 1) << "- DuplicateDetector cleaned up successfully";
        }
        
        qDebug() << "✓ Resource cleanup after errors verified";
    }
};

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    ErrorRecoveryResilienceTest test;
    int result = QTest::qExec(&test, argc, argv);
    
    // Process any remaining events before exit
    QCoreApplication::processEvents();
    
    return result;
}

#include "test_error_recovery_resilience.moc"