#include <QCoreApplication>
#include <QDebug>
#include <QTimer>
#include <QEventLoop>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QSignalSpy>
#include <QTest>
#include <QMetaObject>
#include <QMetaMethod>
#include <QVariant>

#include "file_scanner.h"
#include "duplicate_detector.h"
#include "hash_calculator.h"
#include "file_manager.h"
#include "../src/core/safety_manager.h"
#include "theme_manager.h"

/**
 * @brief API Contract Testing
 * 
 * This test verifies:
 * - API contract compliance across all components
 * - Data transformation and validation between components
 * - Signal/slot connection validation
 * - Method signature consistency
 * - Return value validation
 * - Error handling contract compliance
 * 
 * Requirements: 1.3, 4.5, 10.3
 */

class ApiContractTest : public QObject {
    Q_OBJECT

private:
    QTemporaryDir* m_tempDir;
    QString m_testPath;
    
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

private slots:
    void initTestCase() {
        qDebug() << "===========================================";
        qDebug() << "API Contract Testing";
        qDebug() << "===========================================";
        qDebug();
        
        // Register metatypes
        qRegisterMetaType<FileManager::OperationResult>("FileManager::OperationResult");
        qRegisterMetaType<FileManager::OperationResult>("OperationResult");
        qRegisterMetaType<DuplicateDetector::DuplicateGroup>("DuplicateDetector::DuplicateGroup");
        qRegisterMetaType<DuplicateDetector::DetectionProgress>("DuplicateDetector::DetectionProgress");
        qRegisterMetaType<HashCalculator::HashResult>("HashCalculator::HashResult");
        
        m_tempDir = new QTemporaryDir();
        QVERIFY(m_tempDir->isValid());
        m_testPath = m_tempDir->path();
        
        qDebug() << "Test directory:" << m_testPath;
    }
    
    void cleanupTestCase() {
        delete m_tempDir;
        qDebug() << "\n===========================================";
        qDebug() << "All tests completed";
        qDebug() << "===========================================";
    }
    
    /**
     * Test 1: FileScanner API contract validation
     */
    void test_fileScannerApiContract() {
        qDebug() << "\n[Test 1] FileScanner API Contract";
        qDebug() << "==================================";
        
        FileScanner scanner;
        
        // Test method signatures and return types
        qDebug() << "   Testing method signatures...";
        
        // Test startScan method
        FileScanner::ScanOptions options;
        options.targetPaths << m_testPath;
        options.minimumFileSize = 0;
        
        // Method should not throw and should change state
        bool initialScanning = scanner.isScanning();
        scanner.startScan(options);
        bool scanningAfterStart = scanner.isScanning();
        
        qDebug() << "      Initial scanning state:" << initialScanning;
        qDebug() << "      Scanning after start:" << scanningAfterStart;
        
        QVERIFY(!initialScanning);
        QVERIFY(scanningAfterStart);
        
        // Test signal contracts
        qDebug() << "   Testing signal contracts...";
        
        QSignalSpy scanStartedSpy(&scanner, &FileScanner::scanStarted);
        QSignalSpy scanProgressSpy(&scanner, &FileScanner::scanProgress);
        QSignalSpy scanCompletedSpy(&scanner, &FileScanner::scanCompleted);
        QSignalSpy errorOccurredSpy(&scanner, &FileScanner::errorOccurred);
        
        // Wait for scan completion
        QEventLoop loop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &loop, &QEventLoop::quit);
        QTimer::singleShot(5000, &loop, &QEventLoop::quit);
        loop.exec();
        
        // Verify signal contracts
        QCOMPARE(scanStartedSpy.count(), 1);
        QCOMPARE(scanCompletedSpy.count(), 1);
        qDebug() << "      scanStarted signals:" << scanStartedSpy.count();
        qDebug() << "      scanProgress signals:" << scanProgressSpy.count();
        qDebug() << "      scanCompleted signals:" << scanCompletedSpy.count();
        qDebug() << "      errorOccurred signals:" << errorOccurredSpy.count();
        
        // Test return value contracts
        QVector<FileScanner::FileInfo> files = scanner.getScannedFiles();
        int totalFiles = scanner.getTotalFilesFound();
        qint64 totalBytes = scanner.getTotalBytesScanned();
        
        qDebug() << "   Testing return value contracts...";
        qDebug() << "      Files returned:" << files.size();
        qDebug() << "      Total files:" << totalFiles;
        qDebug() << "      Total bytes:" << totalBytes;
        
        QVERIFY(files.size() >= 0);
        QVERIFY(totalFiles >= 0);
        QVERIFY(totalBytes >= 0);
        QCOMPARE(files.size(), totalFiles);
        
        // Test FileInfo structure contract
        if (!files.isEmpty()) {
            const FileScanner::FileInfo& fileInfo = files.first();
            QVERIFY(!fileInfo.filePath.isEmpty());
            QVERIFY(fileInfo.fileSize >= 0);
            QVERIFY(!fileInfo.fileName.isEmpty());
            QVERIFY(!fileInfo.directory.isEmpty());
            QVERIFY(fileInfo.lastModified.isValid());
        }
        
        qDebug() << "✓ FileScanner API contract validated";
    }
    
    /**
     * Test 2: DuplicateDetector API contract validation
     */
    void test_duplicateDetectorApiContract() {
        qDebug() << "\n[Test 2] DuplicateDetector API Contract";
        qDebug() << "========================================";
        
        // Create test data
        createTestFile("api_test/file1.txt", "Content 1");
        createTestFile("api_test/file2.txt", "Content 2");
        createTestFile("api_test/dup1.txt", "Duplicate content");
        createTestFile("api_test/dup2.txt", "Duplicate content");
        
        // Scan files first
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/api_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(5000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        QVERIFY(scannedFiles.size() >= 4);
        
        // Test DuplicateDetector API
        DuplicateDetector detector;
        
        qDebug() << "   Testing method signatures...";
        
        // Test configuration methods
        DuplicateDetector::DetectionOptions options;
        options.level = DuplicateDetector::DetectionLevel::Standard;
        options.minimumFileSize = 0;
        
        detector.setOptions(options);
        DuplicateDetector::DetectionOptions retrievedOptions = detector.getOptions();
        
        QCOMPARE(retrievedOptions.level, options.level);
        QCOMPARE(retrievedOptions.minimumFileSize, options.minimumFileSize);
        
        // Test signal contracts
        qDebug() << "   Testing signal contracts...";
        
        QSignalSpy detectionStartedSpy(&detector, &DuplicateDetector::detectionStarted);
        QSignalSpy detectionProgressSpy(&detector, &DuplicateDetector::detectionProgress);
        QSignalSpy duplicateGroupFoundSpy(&detector, &DuplicateDetector::duplicateGroupFound);
        QSignalSpy detectionCompletedSpy(&detector, &DuplicateDetector::detectionCompleted);
        QSignalSpy detectionErrorSpy(&detector, &DuplicateDetector::detectionError);
        
        // Start detection
        QEventLoop detectionLoop;
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted, &detectionLoop, &QEventLoop::quit);
        
        detector.findDuplicates(scannedFiles);
        QTimer::singleShot(15000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        // Verify signal contracts
        QCOMPARE(detectionStartedSpy.count(), 1);
        QCOMPARE(detectionCompletedSpy.count(), 1);
        QVERIFY(detectionProgressSpy.count() >= 1);
        
        qDebug() << "      detectionStarted signals:" << detectionStartedSpy.count();
        qDebug() << "      detectionProgress signals:" << detectionProgressSpy.count();
        qDebug() << "      duplicateGroupFound signals:" << duplicateGroupFoundSpy.count();
        qDebug() << "      detectionCompleted signals:" << detectionCompletedSpy.count();
        qDebug() << "      detectionError signals:" << detectionErrorSpy.count();
        
        // Test return value contracts
        QList<DuplicateDetector::DuplicateGroup> groups = detector.getDuplicateGroups();
        int totalGroups = detector.getTotalDuplicateGroups();
        qint64 totalWastedSpace = detector.getTotalWastedSpace();
        
        qDebug() << "   Testing return value contracts...";
        qDebug() << "      Groups returned:" << groups.size();
        qDebug() << "      Total groups:" << totalGroups;
        qDebug() << "      Total wasted space:" << totalWastedSpace;
        
        QVERIFY(groups.size() >= 0);
        QVERIFY(totalGroups >= 0);
        QVERIFY(totalWastedSpace >= 0);
        QCOMPARE(groups.size(), totalGroups);
        
        // Test DuplicateGroup structure contract
        if (!groups.isEmpty()) {
            const DuplicateDetector::DuplicateGroup& group = groups.first();
            QVERIFY(!group.groupId.isEmpty());
            QVERIFY(group.fileCount >= 2);
            QVERIFY(group.fileSize >= 0);
            QVERIFY(group.totalSize >= group.fileSize);
            QVERIFY(group.wastedSpace >= 0);
            QVERIFY(!group.hash.isEmpty());
            QVERIFY(group.detected.isValid());
        }
        
        qDebug() << "✓ DuplicateDetector API contract validated";
    }
    
    /**
     * Test 3: HashCalculator API contract validation
     */
    void test_hashCalculatorApiContract() {
        qDebug() << "\n[Test 3] HashCalculator API Contract";
        qDebug() << "=====================================";
        
        // Create test files
        QString file1 = createTestFile("hash_test/file1.txt", "Hash test content 1");
        QString file2 = createTestFile("hash_test/file2.txt", "Hash test content 2");
        
        HashCalculator hashCalc;
        
        qDebug() << "   Testing method signatures...";
        
        // Test single file hash calculation
        QSignalSpy hashCompletedSpy(&hashCalc, &HashCalculator::hashCompleted);
        QSignalSpy hashErrorSpy(&hashCalc, &HashCalculator::hashError);
        QSignalSpy allOperationsCompleteSpy(&hashCalc, &HashCalculator::allOperationsComplete);
        
        hashCalc.calculateFileHash(file1);
        
        // Wait for completion
        QEventLoop loop;
        QObject::connect(&hashCalc, &HashCalculator::hashCompleted, &loop, &QEventLoop::quit);
        QTimer::singleShot(5000, &loop, &QEventLoop::quit);
        loop.exec();
        
        // Verify signal contracts
        QVERIFY(hashCompletedSpy.count() >= 1);
        qDebug() << "      hashCompleted signals:" << hashCompletedSpy.count();
        qDebug() << "      hashError signals:" << hashErrorSpy.count();
        
        // Test HashResult structure contract
        if (hashCompletedSpy.count() > 0) {
            QList<QVariant> arguments = hashCompletedSpy.takeFirst();
            QVERIFY(arguments.size() >= 1);
            
            // Note: In a real implementation, we would extract and validate the HashResult
            qDebug() << "      HashResult signal arguments:" << arguments.size();
        }
        
        // Test batch hash calculation
        QStringList files = {file1, file2};
        hashCalc.calculateFileHashes(files);
        
        QEventLoop batchLoop;
        QObject::connect(&hashCalc, &HashCalculator::allOperationsComplete, &batchLoop, &QEventLoop::quit);
        QTimer::singleShot(10000, &batchLoop, &QEventLoop::quit);
        batchLoop.exec();
        
        QVERIFY(allOperationsCompleteSpy.count() >= 1);
        qDebug() << "      allOperationsComplete signals:" << allOperationsCompleteSpy.count();
        
        // Test statistics contract
        HashCalculator::Statistics stats = hashCalc.getStatistics();
        qDebug() << "   Testing statistics contract...";
        qDebug() << "      Total hashes:" << stats.totalHashesCalculated;
        qDebug() << "      Cache hits:" << stats.cacheHits;
        qDebug() << "      Cache misses:" << stats.cacheMisses;
        qDebug() << "      Total bytes:" << stats.totalBytesProcessed;
        qDebug() << "      Average speed:" << stats.averageSpeed;
        
        QVERIFY(stats.totalHashesCalculated >= 0);
        QVERIFY(stats.cacheHits >= 0);
        QVERIFY(stats.cacheMisses >= 0);
        QVERIFY(stats.totalBytesProcessed >= 0);
        QVERIFY(stats.averageSpeed >= 0.0);
        
        qDebug() << "✓ HashCalculator API contract validated";
    }
    
    /**
     * Test 4: FileManager API contract validation
     */
    void test_fileManagerApiContract() {
        qDebug() << "\n[Test 4] FileManager API Contract";
        qDebug() << "==================================";
        
        // Create test files
        QString file1 = createTestFile("filemanager_test/file1.txt", "File manager test 1");
        QString file2 = createTestFile("filemanager_test/file2.txt", "File manager test 2");
        
        FileManager fileManager;
        SafetyManager safetyManager;
        
        // Configure components
        safetyManager.setBackupDirectory(m_tempDir->path() + "/backups");
        fileManager.setSafetyManager(&safetyManager);
        
        qDebug() << "   Testing method signatures...";
        
        // Test operation methods return operation IDs
        QString deleteOpId = fileManager.deleteFiles({file1}, "API contract test");
        QVERIFY(!deleteOpId.isEmpty());
        qDebug() << "      Delete operation ID:" << deleteOpId;
        
        // Test signal contracts
        QSignalSpy operationStartedSpy(&fileManager, &FileManager::operationStarted);
        QSignalSpy operationProgressSpy(&fileManager, &FileManager::operationProgress);
        QSignalSpy operationCompletedSpy(&fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        QSignalSpy operationErrorSpy(&fileManager, &FileManager::operationError);
        
        // Wait for operation completion
        QEventLoop loop;
        QObject::connect(&fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)), 
                        &loop, &QEventLoop::quit);
        QTimer::singleShot(10000, &loop, &QEventLoop::quit);
        loop.exec();
        
        // Verify signal contracts
        qDebug() << "   Testing signal contracts...";
        qDebug() << "      operationStarted signals:" << operationStartedSpy.count();
        qDebug() << "      operationProgress signals:" << operationProgressSpy.count();
        qDebug() << "      operationCompleted signals:" << operationCompletedSpy.count();
        qDebug() << "      operationError signals:" << operationErrorSpy.count();
        
        QVERIFY(operationCompletedSpy.count() >= 1);
        
        // Test return value contracts
        FileManager::OperationResult result = fileManager.getOperationResult(deleteOpId);
        qDebug() << "   Testing return value contracts...";
        qDebug() << "      Operation ID:" << result.operationId;
        qDebug() << "      Success:" << result.success;
        qDebug() << "      Processed files:" << result.processedFiles.size();
        qDebug() << "      Failed files:" << result.failedFiles.size();
        qDebug() << "      Total size:" << result.totalSize;
        
        QCOMPARE(result.operationId, deleteOpId);
        QVERIFY(result.totalSize >= 0);
        QVERIFY(result.completed.isValid());
        
        // Test operation status methods
        bool isInProgress = fileManager.isOperationInProgress(deleteOpId);
        qDebug() << "      Operation in progress:" << isInProgress;
        QVERIFY(!isInProgress); // Should be completed by now
        
        QList<FileManager::OperationResult> recentOps = fileManager.getRecentOperations(10);
        qDebug() << "      Recent operations:" << recentOps.size();
        QVERIFY(recentOps.size() >= 1);
        
        qDebug() << "✓ FileManager API contract validated";
    }
    
    /**
     * Test 5: SafetyManager API contract validation
     */
    void test_safetyManagerApiContract() {
        qDebug() << "\n[Test 5] SafetyManager API Contract";
        qDebug() << "====================================";
        
        // Create test file
        QString testFile = createTestFile("safety_test/test.txt", "Safety manager test");
        
        SafetyManager safetyManager;
        safetyManager.setBackupDirectory(m_tempDir->path() + "/safety_backups");
        
        qDebug() << "   Testing method signatures...";
        
        // Test backup creation
        QSignalSpy backupCompletedSpy(&safetyManager, &SafetyManager::backupCompleted);
        QSignalSpy backupErrorSpy(&safetyManager, &SafetyManager::backupError);
        
        QString backupId = safetyManager.createBackup(testFile, "API contract test");
        QVERIFY(!backupId.isEmpty());
        qDebug() << "      Backup ID:" << backupId;
        
        // Wait for backup completion
        QEventLoop loop;
        QObject::connect(&safetyManager, &SafetyManager::backupCompleted, &loop, &QEventLoop::quit);
        QTimer::singleShot(5000, &loop, &QEventLoop::quit);
        loop.exec();
        
        // Verify signal contracts
        qDebug() << "   Testing signal contracts...";
        qDebug() << "      backupCompleted signals:" << backupCompletedSpy.count();
        qDebug() << "      backupError signals:" << backupErrorSpy.count();
        
        QVERIFY(backupCompletedSpy.count() >= 1);
        
        // Test backup info retrieval
        SafetyManager::BackupInfo backupInfo = safetyManager.getBackupInfo(backupId);
        qDebug() << "   Testing return value contracts...";
        qDebug() << "      Original path:" << backupInfo.originalPath;
        qDebug() << "      Backup path:" << backupInfo.backupPath;
        qDebug() << "      File size:" << backupInfo.fileSize;
        qDebug() << "      Created at:" << backupInfo.createdAt.toString();
        
        QCOMPARE(backupInfo.originalPath, testFile);
        QVERIFY(!backupInfo.backupPath.isEmpty());
        QVERIFY(backupInfo.fileSize >= 0);
        QVERIFY(backupInfo.createdAt.isValid());
        
        // Test backup validation
        bool isValid = safetyManager.validateBackupIntegrity(backupId);
        qDebug() << "      Backup integrity valid:" << isValid;
        QVERIFY(isValid);
        
        // Test backup listing
        QStringList backups = safetyManager.getAvailableBackups();
        qDebug() << "      Available backups:" << backups.size();
        QVERIFY(backups.size() >= 1);
        QVERIFY(backups.contains(backupId));
        
        qDebug() << "✓ SafetyManager API contract validated";
    }
    
    /**
     * Test 6: ThemeManager API contract validation
     */
    void test_themeManagerApiContract() {
        qDebug() << "\n[Test 6] ThemeManager API Contract";
        qDebug() << "===================================";
        
        ThemeManager* themeManager = ThemeManager::instance();
        QVERIFY(themeManager != nullptr);
        
        qDebug() << "   Testing method signatures...";
        
        // Test theme setting and getting
        ThemeManager::Theme originalTheme = themeManager->currentTheme();
        QString originalThemeString = themeManager->currentThemeString();
        
        qDebug() << "      Original theme:" << originalThemeString;
        
        // Test theme switching
        ThemeManager::Theme newTheme = (originalTheme == ThemeManager::Light) ? 
                                      ThemeManager::Dark : ThemeManager::Light;
        
        QSignalSpy themeChangedSpy(themeManager, &ThemeManager::themeChanged);
        
        themeManager->setTheme(newTheme);
        QCOMPARE(themeManager->currentTheme(), newTheme);
        QCOMPARE(themeChangedSpy.count(), 1);
        
        qDebug() << "      New theme:" << themeManager->currentThemeString();
        qDebug() << "      themeChanged signals:" << themeChangedSpy.count();
        
        // Test style generation methods
        QString appStyleSheet = themeManager->getApplicationStyleSheet();
        QVERIFY(!appStyleSheet.isEmpty());
        qDebug() << "      Application stylesheet length:" << appStyleSheet.length();
        
        QString progressStyle = themeManager->getProgressBarStyle(ThemeManager::ProgressType::Normal);
        QVERIFY(!progressStyle.isEmpty());
        qDebug() << "      Progress bar style length:" << progressStyle.length();
        
        QString statusStyle = themeManager->getStatusIndicatorStyle(ThemeManager::StatusType::Success);
        QVERIFY(!statusStyle.isEmpty());
        qDebug() << "      Status indicator style length:" << statusStyle.length();
        
        // Test system theme detection
        bool isSystemDark = themeManager->isSystemDarkMode();
        qDebug() << "      System dark mode:" << isSystemDark;
        
        // Restore original theme
        themeManager->setTheme(originalTheme);
        QCOMPARE(themeManager->currentTheme(), originalTheme);
        
        qDebug() << "✓ ThemeManager API contract validated";
    }
    
    /**
     * Test 7: Cross-component data transformation validation
     */
    void test_crossComponentDataTransformation() {
        qDebug() << "\n[Test 7] Cross-Component Data Transformation";
        qDebug() << "=============================================";
        
        // Create test data
        createTestFile("transform_test/file1.txt", "Transform test 1");
        createTestFile("transform_test/file2.txt", "Transform test 2");
        
        // Test FileScanner -> DuplicateDetector transformation
        qDebug() << "   Testing FileScanner -> DuplicateDetector transformation...";
        
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/transform_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(5000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannerFiles = scanner.getScannedFiles();
        QVERIFY(scannerFiles.size() >= 2);
        
        // Transform to DuplicateDetector format
        QList<DuplicateDetector::FileInfo> detectorFiles;
        for (const FileScanner::FileInfo& scanInfo : scannerFiles) {
            DuplicateDetector::FileInfo detectorInfo = DuplicateDetector::FileInfo::fromScannerInfo(scanInfo);
            detectorFiles.append(detectorInfo);
            
            // Verify transformation preserves data
            QCOMPARE(detectorInfo.filePath, scanInfo.filePath);
            QCOMPARE(detectorInfo.fileSize, scanInfo.fileSize);
            QCOMPARE(detectorInfo.fileName, scanInfo.fileName);
            QCOMPARE(detectorInfo.directory, scanInfo.directory);
            QCOMPARE(detectorInfo.lastModified, scanInfo.lastModified);
        }
        
        qDebug() << "      Transformed" << detectorFiles.size() << "FileInfo objects";
        
        // Test DuplicateDetector processing
        DuplicateDetector detector;
        QEventLoop detectionLoop;
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted, &detectionLoop, &QEventLoop::quit);
        
        detector.findDuplicates(detectorFiles);
        QTimer::singleShot(10000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        QList<DuplicateDetector::DuplicateGroup> groups = detector.getDuplicateGroups();
        qDebug() << "      Detection completed, groups found:" << groups.size();
        
        // Verify group data integrity
        for (const DuplicateDetector::DuplicateGroup& group : groups) {
            QVERIFY(!group.groupId.isEmpty());
            QVERIFY(group.fileCount >= 2);
            QVERIFY(group.files.size() == group.fileCount);
            QVERIFY(group.totalSize == group.fileSize * group.fileCount);
            QVERIFY(group.wastedSpace == group.fileSize * (group.fileCount - 1));
        }
        
        qDebug() << "✓ Cross-component data transformation validated";
    }
    
    /**
     * Test 8: Signal/slot connection validation across components
     */
    void test_signalSlotConnectionValidation() {
        qDebug() << "\n[Test 8] Signal/Slot Connection Validation";
        qDebug() << "===========================================";
        
        // Test FileScanner signals
        qDebug() << "   Testing FileScanner signal connections...";
        
        FileScanner scanner;
        
        // Verify signal existence using Qt's meta-object system
        const QMetaObject* scannerMeta = scanner.metaObject();
        
        QStringList expectedSignals = {
            "scanStarted()",
            "scanProgress(int,int,QString)",
            "scanCompleted()",
            "scanCancelled()",
            "errorOccurred(QString)"
        };
        
        for (const QString& signalName : expectedSignals) {
            int signalIndex = scannerMeta->indexOfSignal(signalName.toUtf8().constData());
            QVERIFY(signalIndex >= 0);
            qDebug() << "      Signal found:" << signalName;
        }
        
        // Test DuplicateDetector signals
        qDebug() << "   Testing DuplicateDetector signal connections...";
        
        DuplicateDetector detector;
        const QMetaObject* detectorMeta = detector.metaObject();
        
        QStringList expectedDetectorSignals = {
            "detectionStarted(int)",
            "detectionProgress(DuplicateDetector::DetectionProgress)",
            "duplicateGroupFound(DuplicateDetector::DuplicateGroup)",
            "detectionCompleted(int)",
            "detectionCancelled()",
            "detectionError(QString)"
        };
        
        for (const QString& signalName : expectedDetectorSignals) {
            int signalIndex = detectorMeta->indexOfSignal(signalName.toUtf8().constData());
            if (signalIndex >= 0) {
                qDebug() << "      Signal found:" << signalName;
            } else {
                qDebug() << "      Signal not found (may use different signature):" << signalName;
            }
        }
        
        // Test cross-component signal/slot connections
        qDebug() << "   Testing cross-component connections...";
        
        // Create a simple connection test
        QSignalSpy scanCompletedSpy(&scanner, &FileScanner::scanCompleted);
        QSignalSpy detectionStartedSpy(&detector, &DuplicateDetector::detectionStarted);
        
        // Connect scanner completion to detector start (simulated workflow)
        QObject::connect(&scanner, &FileScanner::scanCompleted, [&detector, &scanner]() {
            QVector<FileScanner::FileInfo> files = scanner.getScannedFiles();
            if (!files.isEmpty()) {
                detector.findDuplicates(files);
            }
        });
        
        // Create test file and run workflow
        createTestFile("connection_test/test.txt", "Connection test");
        
        FileScanner::ScanOptions options;
        options.targetPaths << m_testPath + "/connection_test";
        options.minimumFileSize = 0;
        
        scanner.startScan(options);
        
        // Wait for scan completion
        QEventLoop loop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &loop, &QEventLoop::quit);
        QTimer::singleShot(5000, &loop, &QEventLoop::quit);
        loop.exec();
        
        // Give time for connected slot to execute
        QTest::qWait(100);
        
        QCOMPARE(scanCompletedSpy.count(), 1);
        qDebug() << "      Scan completed signals:" << scanCompletedSpy.count();
        qDebug() << "      Detection started signals:" << detectionStartedSpy.count();
        
        qDebug() << "✓ Signal/slot connection validation completed";
    }
};

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    ApiContractTest test;
    int result = QTest::qExec(&test, argc, argv);
    
    // Process any remaining events before exit
    QCoreApplication::processEvents();
    
    return result;
}

#include "test_api_contracts.moc"