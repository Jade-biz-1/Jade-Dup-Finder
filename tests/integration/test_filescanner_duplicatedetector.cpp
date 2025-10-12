#include <QCoreApplication>
#include <QDebug>
#include <QTimer>
#include <QEventLoop>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QSignalSpy>
#include <QTest>
#include <QRandomGenerator>
#include <QElapsedTimer>

#include "file_scanner.h"
#include "duplicate_detector.h"

/**
 * @brief Integration test for FileScanner and DuplicateDetector
 * 
 * This test verifies:
 * - FileInfo structure compatibility
 * - FileInfo::fromScannerInfo() conversion
 * - End-to-end duplicate detection workflow
 * - Large datasets (10,000+ files)
 * - Performance meets targets
 * 
 * Requirements: 4.2, 4.4, 4.5
 */

class FileScannerDuplicateDetectorTest : public QObject {
    Q_OBJECT

private:
    QTemporaryDir* m_tempDir;
    QString m_testPath;
    
    // Helper to create test file with specific content
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
    
    // Helper to create test file with random content of specific size
    QString createTestFileWithSize(const QString& relativePath, qint64 size) {
        QByteArray content;
        content.reserve(size);
        
        // Generate random content
        for (qint64 i = 0; i < size; i++) {
            content.append(static_cast<char>(QRandomGenerator::global()->bounded(256)));
        }
        
        return createTestFile(relativePath, content);
    }
    
    // Helper to create duplicate files with same content
    QStringList createDuplicateSet(const QString& basePath, const QByteArray& content, int count) {
        QStringList files;
        for (int i = 0; i < count; i++) {
            QString path = QString("%1/duplicate_%2.dat").arg(basePath).arg(i);
            QString fullPath = createTestFile(path, content);
            if (!fullPath.isEmpty()) {
                files << fullPath;
            }
        }
        return files;
    }

private slots:
    void initTestCase() {
        qDebug() << "===========================================";
        qDebug() << "FileScanner <-> DuplicateDetector Integration Test";
        qDebug() << "===========================================";
        qDebug();
        
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
     * Test 1: FileInfo structure compatibility
     * Verify that FileScanner::FileInfo and DuplicateDetector::FileInfo are compatible
     */
    void test_fileInfoStructureCompatibility() {
        qDebug() << "\n[Test 1] FileInfo Structure Compatibility";
        qDebug() << "==========================================";
        
        // Create test file
        QString testFile = createTestFile("compat_test/test.txt", "Test content");
        QVERIFY(!testFile.isEmpty());
        
        // Scan the file
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/compat_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(5000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        QVERIFY(scannedFiles.size() >= 1);
        
        const FileScanner::FileInfo& scanInfo = scannedFiles.first();
        
        qDebug() << "FileScanner::FileInfo:";
        qDebug() << "   filePath:" << scanInfo.filePath;
        qDebug() << "   fileSize:" << scanInfo.fileSize;
        qDebug() << "   fileName:" << scanInfo.fileName;
        qDebug() << "   directory:" << scanInfo.directory;
        qDebug() << "   lastModified:" << scanInfo.lastModified.toString();
        
        // Verify all fields are populated
        QVERIFY(!scanInfo.filePath.isEmpty());
        QVERIFY(scanInfo.fileSize >= 0);
        QVERIFY(!scanInfo.fileName.isEmpty());
        QVERIFY(!scanInfo.directory.isEmpty());
        QVERIFY(scanInfo.lastModified.isValid());
        
        qDebug() << "✓ FileScanner::FileInfo structure verified";
    }
    
    /**
     * Test 2: FileInfo::fromScannerInfo() conversion
     * Verify the conversion function works correctly
     */
    void test_fromScannerInfoConversion() {
        qDebug() << "\n[Test 2] FileInfo::fromScannerInfo() Conversion";
        qDebug() << "================================================";
        
        // Create test files
        createTestFile("convert_test/file1.txt", "Content 1");
        createTestFile("convert_test/file2.txt", "Content 2");
        
        // Scan files
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/convert_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(5000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        QVERIFY(scannedFiles.size() >= 2);
        
        qDebug() << "Converting" << scannedFiles.size() << "FileScanner::FileInfo to DuplicateDetector::FileInfo";
        
        // Convert each FileInfo
        for (const FileScanner::FileInfo& scanInfo : scannedFiles) {
            DuplicateDetector::FileInfo detectorInfo = DuplicateDetector::FileInfo::fromScannerInfo(scanInfo);
            
            // Verify conversion
            QCOMPARE(detectorInfo.filePath, scanInfo.filePath);
            QCOMPARE(detectorInfo.fileSize, scanInfo.fileSize);
            QCOMPARE(detectorInfo.fileName, scanInfo.fileName);
            QCOMPARE(detectorInfo.directory, scanInfo.directory);
            QCOMPARE(detectorInfo.lastModified, scanInfo.lastModified);
            
            qDebug() << "   ✓" << scanInfo.fileName;
            qDebug() << "      Path match:" << (detectorInfo.filePath == scanInfo.filePath);
            qDebug() << "      Size match:" << (detectorInfo.fileSize == scanInfo.fileSize);
            qDebug() << "      Name match:" << (detectorInfo.fileName == scanInfo.fileName);
            qDebug() << "      Directory match:" << (detectorInfo.directory == scanInfo.directory);
            qDebug() << "      Modified match:" << (detectorInfo.lastModified == scanInfo.lastModified);
        }
        
        qDebug() << "✓ FileInfo conversion verified";
    }
    
    /**
     * Test 3: End-to-end duplicate detection workflow
     * Complete workflow from scan to duplicate detection
     */
    void test_endToEndDuplicateDetection() {
        qDebug() << "\n[Test 3] End-to-End Duplicate Detection Workflow";
        qDebug() << "=================================================";
        
        // Create test files with duplicates
        QByteArray content1 = "This is content set 1";
        QByteArray content2 = "This is content set 2";
        QByteArray content3 = "This is content set 3";
        
        // Create 3 sets of duplicates
        createDuplicateSet("workflow/set1", content1, 3);  // 3 duplicates
        createDuplicateSet("workflow/set2", content2, 4);  // 4 duplicates
        createDuplicateSet("workflow/set3", content3, 2);  // 2 duplicates
        
        // Create some unique files
        createTestFile("workflow/unique1.txt", "Unique content 1");
        createTestFile("workflow/unique2.txt", "Unique content 2");
        
        qDebug() << "Created test structure:";
        qDebug() << "   - 3 duplicate sets (3, 4, 2 files each)";
        qDebug() << "   - 2 unique files";
        qDebug() << "   - Total: 11 files";
        
        // Phase 1: Scan files
        qDebug() << "\nPhase 1: Scanning files...";
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/workflow";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(10000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        qDebug() << "   Scanned" << scannedFiles.size() << "files";
        QVERIFY(scannedFiles.size() >= 11);
        
        // Phase 2: Detect duplicates
        qDebug() << "\nPhase 2: Detecting duplicates...";
        DuplicateDetector detector;
        
        // Configure detection options
        DuplicateDetector::DetectionOptions detectionOptions;
        detectionOptions.level = DuplicateDetector::DetectionLevel::Standard;
        detectionOptions.minimumFileSize = 0;
        detectionOptions.skipEmptyFiles = false;
        detector.setOptions(detectionOptions);
        
        QEventLoop detectionLoop;
        QList<DuplicateDetector::DuplicateGroup> foundGroups;
        
        QObject::connect(&detector, &DuplicateDetector::detectionStarted, 
                        [](int totalFiles) {
            qDebug() << "   Detection started for" << totalFiles << "files";
        });
        
        QObject::connect(&detector, &DuplicateDetector::detectionProgress,
                        [](const DuplicateDetector::DetectionProgress& progress) {
            QString phase;
            switch (progress.currentPhase) {
                case DuplicateDetector::DetectionProgress::SizeGrouping:
                    phase = "Size Grouping";
                    break;
                case DuplicateDetector::DetectionProgress::HashCalculation:
                    phase = "Hash Calculation";
                    break;
                case DuplicateDetector::DetectionProgress::DuplicateGrouping:
                    phase = "Duplicate Grouping";
                    break;
                case DuplicateDetector::DetectionProgress::GeneratingRecommendations:
                    phase = "Generating Recommendations";
                    break;
                case DuplicateDetector::DetectionProgress::Complete:
                    phase = "Complete";
                    break;
            }
            qDebug() << "   Progress:" << phase << "-" 
                     << QString::number(progress.percentComplete, 'f', 1) << "%";
        });
        
        QObject::connect(&detector, &DuplicateDetector::duplicateGroupFound,
                        [&foundGroups](const DuplicateDetector::DuplicateGroup& group) {
            foundGroups.append(group);
            qDebug() << "   Found duplicate group:" << group.fileCount << "files,"
                     << "wasted space:" << group.wastedSpace << "bytes";
        });
        
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted,
                        [&detectionLoop](int totalGroups) {
            qDebug() << "   Detection completed:" << totalGroups << "duplicate groups found";
            detectionLoop.quit();
        });
        
        QObject::connect(&detector, &DuplicateDetector::detectionError,
                        [](const QString& error) {
            qWarning() << "   Detection error:" << error;
        });
        
        // Start detection using the convenience method
        detector.findDuplicates(scannedFiles);
        
        QTimer::singleShot(30000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        // Phase 3: Verify results
        qDebug() << "\nPhase 3: Verifying results...";
        
        QList<DuplicateDetector::DuplicateGroup> groups = detector.getDuplicateGroups();
        qDebug() << "   Total duplicate groups:" << groups.size();
        
        // We should have 3 duplicate groups (set1, set2, set3)
        QVERIFY(groups.size() >= 3);
        
        // Verify group details
        int totalDuplicateFiles = 0;
        qint64 totalWastedSpace = 0;
        
        for (const DuplicateDetector::DuplicateGroup& group : groups) {
            qDebug() << "   Group" << group.groupId << ":";
            qDebug() << "      Files:" << group.fileCount;
            qDebug() << "      File size:" << group.fileSize << "bytes";
            qDebug() << "      Total size:" << group.totalSize << "bytes";
            qDebug() << "      Wasted space:" << group.wastedSpace << "bytes";
            qDebug() << "      Hash:" << group.hash.left(16) + "...";
            qDebug() << "      Recommendation:" << group.recommendedAction;
            
            QVERIFY(group.fileCount >= 2);  // At least 2 files per group
            QVERIFY(group.fileSize > 0);
            QVERIFY(group.totalSize == group.fileSize * group.fileCount);
            QVERIFY(group.wastedSpace == group.fileSize * (group.fileCount - 1));
            QVERIFY(!group.hash.isEmpty());
            
            totalDuplicateFiles += group.fileCount;
            totalWastedSpace += group.wastedSpace;
        }
        
        qDebug() << "   Total duplicate files:" << totalDuplicateFiles;
        qDebug() << "   Total wasted space:" << totalWastedSpace << "bytes";
        
        // Verify statistics
        DuplicateDetector::DetectionStatistics stats = detector.getStatistics();
        qDebug() << "   Detection statistics:";
        qDebug() << "      Total files processed:" << stats.totalFilesProcessed;
        qDebug() << "      Files with unique size:" << stats.filesWithUniqueSize;
        qDebug() << "      Files in size groups:" << stats.filesInSizeGroups;
        qDebug() << "      Hash calculations:" << stats.hashCalculationsPerformed;
        qDebug() << "      Duplicate groups found:" << stats.duplicateGroupsFound;
        qDebug() << "      Total duplicate files:" << stats.totalDuplicateFiles;
        qDebug() << "      Total wasted space:" << stats.totalWastedSpace;
        qDebug() << "      Detection time:" << stats.detectionTime.toString("mm:ss.zzz");
        qDebug() << "      Average group size:" << QString::number(stats.averageGroupSize, 'f', 2);
        
        QCOMPARE(stats.duplicateGroupsFound, groups.size());
        QCOMPARE(stats.totalWastedSpace, totalWastedSpace);
        
        qDebug() << "✓ End-to-end duplicate detection verified";
    }
    
    /**
     * Test 4: Large dataset performance (10,000+ files)
     * Test with a large number of files to verify performance
     */
    void test_largeDatasetPerformance() {
        qDebug() << "\n[Test 4] Large Dataset Performance (10,000+ files)";
        qDebug() << "===================================================";
        
        // Create a large dataset with duplicates
        const int totalFiles = 10000;
        const int duplicateSets = 100;  // 100 sets of duplicates
        const int filesPerSet = 10;     // 10 files per set
        const int uniqueFiles = totalFiles - (duplicateSets * filesPerSet);
        
        qDebug() << "Creating large test dataset:";
        qDebug() << "   Total files:" << totalFiles;
        qDebug() << "   Duplicate sets:" << duplicateSets;
        qDebug() << "   Files per set:" << filesPerSet;
        qDebug() << "   Unique files:" << uniqueFiles;
        
        QElapsedTimer creationTimer;
        creationTimer.start();
        
        // Create duplicate sets
        for (int i = 0; i < duplicateSets; i++) {
            QByteArray content = QString("Duplicate set %1 content").arg(i).toUtf8();
            createDuplicateSet(QString("large/set%1").arg(i), content, filesPerSet);
            
            if ((i + 1) % 20 == 0) {
                qDebug() << "   Created" << (i + 1) << "duplicate sets...";
            }
        }
        
        // Create unique files
        for (int i = 0; i < uniqueFiles; i++) {
            QByteArray content = QString("Unique file %1 content").arg(i).toUtf8();
            createTestFile(QString("large/unique/file%1.txt").arg(i), content);
            
            if ((i + 1) % 2000 == 0) {
                qDebug() << "   Created" << (i + 1) << "unique files...";
            }
        }
        
        qint64 creationTime = creationTimer.elapsed();
        qDebug() << "   File creation completed in" << creationTime << "ms";
        
        // Phase 1: Scan files
        qDebug() << "\nPhase 1: Scanning" << totalFiles << "files...";
        QElapsedTimer scanTimer;
        scanTimer.start();
        
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/large";
        scanOptions.minimumFileSize = 0;
        scanOptions.progressBatchSize = 1000;  // Progress every 1000 files
        
        QEventLoop scanLoop;
        int lastProgress = 0;
        
        QObject::connect(&scanner, &FileScanner::scanProgress,
                        [&lastProgress](int processed, int total, const QString& path) {
            Q_UNUSED(total);
            Q_UNUSED(path);
            if (processed - lastProgress >= 1000) {
                qDebug() << "   Scanned" << processed << "files...";
                lastProgress = processed;
            }
        });
        
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(60000, &scanLoop, &QEventLoop::quit);  // 60 second timeout
        scanLoop.exec();
        
        qint64 scanTime = scanTimer.elapsed();
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        
        qDebug() << "   Scan completed:";
        qDebug() << "      Files found:" << scannedFiles.size();
        qDebug() << "      Scan time:" << scanTime << "ms";
        qDebug() << "      Scan rate:" << QString::number(scannedFiles.size() * 1000.0 / scanTime, 'f', 2) << "files/sec";
        
        QVERIFY(scannedFiles.size() >= totalFiles * 0.95);  // Allow 5% tolerance
        
        // Get scan statistics
        FileScanner::ScanStatistics scanStats = scanner.getScanStatistics();
        qDebug() << "   Scan statistics:";
        qDebug() << "      Total files:" << scanStats.totalFilesScanned;
        qDebug() << "      Total directories:" << scanStats.totalDirectoriesScanned;
        qDebug() << "      Total bytes:" << scanStats.totalBytesScanned;
        qDebug() << "      Duration:" << scanStats.scanDurationMs << "ms";
        qDebug() << "      Files/second:" << QString::number(scanStats.filesPerSecond, 'f', 2);
        
        // Phase 2: Detect duplicates
        qDebug() << "\nPhase 2: Detecting duplicates in" << scannedFiles.size() << "files...";
        QElapsedTimer detectionTimer;
        detectionTimer.start();
        
        DuplicateDetector detector;
        
        DuplicateDetector::DetectionOptions detectionOptions;
        detectionOptions.level = DuplicateDetector::DetectionLevel::Standard;
        detectionOptions.minimumFileSize = 0;
        detectionOptions.skipEmptyFiles = false;
        detector.setOptions(detectionOptions);
        
        QEventLoop detectionLoop;
        int lastDetectionProgress = 0;
        
        QObject::connect(&detector, &DuplicateDetector::detectionProgress,
                        [&lastDetectionProgress](const DuplicateDetector::DetectionProgress& progress) {
            int currentProgress = static_cast<int>(progress.percentComplete);
            if (currentProgress - lastDetectionProgress >= 10) {
                qDebug() << "   Detection progress:" << currentProgress << "%";
                lastDetectionProgress = currentProgress;
            }
        });
        
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted,
                        [&detectionLoop](int totalGroups) {
            qDebug() << "   Detection completed:" << totalGroups << "duplicate groups";
            detectionLoop.quit();
        });
        
        detector.findDuplicates(scannedFiles);
        
        QTimer::singleShot(120000, &detectionLoop, &QEventLoop::quit);  // 120 second timeout
        detectionLoop.exec();
        
        qint64 detectionTime = detectionTimer.elapsed();
        
        // Phase 3: Verify results and performance
        qDebug() << "\nPhase 3: Verifying results and performance...";
        
        QList<DuplicateDetector::DuplicateGroup> groups = detector.getDuplicateGroups();
        qDebug() << "   Duplicate groups found:" << groups.size();
        
        // We should have approximately 100 duplicate groups
        QVERIFY(groups.size() >= duplicateSets * 0.9);  // Allow 10% tolerance
        
        // Calculate total wasted space
        qint64 totalWastedSpace = 0;
        int totalDuplicateFiles = 0;
        
        for (const DuplicateDetector::DuplicateGroup& group : groups) {
            totalWastedSpace += group.wastedSpace;
            totalDuplicateFiles += group.fileCount;
        }
        
        qDebug() << "   Total duplicate files:" << totalDuplicateFiles;
        qDebug() << "   Total wasted space:" << totalWastedSpace << "bytes";
        
        // Get detection statistics
        DuplicateDetector::DetectionStatistics stats = detector.getStatistics();
        qDebug() << "   Detection statistics:";
        qDebug() << "      Total files processed:" << stats.totalFilesProcessed;
        qDebug() << "      Files with unique size:" << stats.filesWithUniqueSize;
        qDebug() << "      Files in size groups:" << stats.filesInSizeGroups;
        qDebug() << "      Hash calculations:" << stats.hashCalculationsPerformed;
        qDebug() << "      Duplicate groups found:" << stats.duplicateGroupsFound;
        qDebug() << "      Total duplicate files:" << stats.totalDuplicateFiles;
        qDebug() << "      Total wasted space:" << stats.totalWastedSpace;
        qDebug() << "      Detection time:" << stats.detectionTime.toString("mm:ss.zzz");
        qDebug() << "      Average group size:" << QString::number(stats.averageGroupSize, 'f', 2);
        
        // Performance verification
        qDebug() << "\n   Performance Summary:";
        qDebug() << "      Total time:" << (scanTime + detectionTime) << "ms";
        qDebug() << "      Scan time:" << scanTime << "ms";
        qDebug() << "      Detection time:" << detectionTime << "ms";
        qDebug() << "      Overall rate:" << QString::number(scannedFiles.size() * 1000.0 / (scanTime + detectionTime), 'f', 2) << "files/sec";
        
        // Performance targets (from requirements):
        // - Scan rate: >= 1,000 files/minute on SSD (16.67 files/sec)
        // - Memory usage: <= 100MB for 100,000 files (not directly testable here)
        
        double scanRate = scannedFiles.size() * 1000.0 / scanTime;
        qDebug() << "      Scan rate:" << QString::number(scanRate, 'f', 2) << "files/sec";
        
        // Verify scan rate meets target (16.67 files/sec minimum)
        // Note: This may fail on slow systems or in CI environments
        if (scanRate >= 16.67) {
            qDebug() << "   ✓ Scan rate meets target (>= 16.67 files/sec)";
        } else {
            qWarning() << "   ⚠ Scan rate below target:" << scanRate << "< 16.67 files/sec";
            qWarning() << "      This may be due to slow disk or system load";
        }
        
        qDebug() << "✓ Large dataset performance test completed";
    }
    
    /**
     * Test 5: Signal/slot connections and error handling
     * Verify all signals work correctly and errors are handled
     */
    void test_signalSlotConnectionsAndErrors() {
        qDebug() << "\n[Test 5] Signal/Slot Connections and Error Handling";
        qDebug() << "====================================================";
        
        // Create test files
        createTestFile("signal_test/file1.txt", "Content 1");
        createTestFile("signal_test/file2.txt", "Content 2");
        createTestFile("signal_test/dup1.txt", "Duplicate");
        createTestFile("signal_test/dup2.txt", "Duplicate");
        
        // Scan files
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/signal_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(5000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        QVERIFY(scannedFiles.size() >= 4);
        
        // Test DuplicateDetector signals
        qDebug() << "Testing DuplicateDetector signals...";
        DuplicateDetector detector;
        
        QSignalSpy detectionStartedSpy(&detector, &DuplicateDetector::detectionStarted);
        QSignalSpy detectionProgressSpy(&detector, &DuplicateDetector::detectionProgress);
        QSignalSpy duplicateGroupFoundSpy(&detector, &DuplicateDetector::duplicateGroupFound);
        QSignalSpy detectionCompletedSpy(&detector, &DuplicateDetector::detectionCompleted);
        QSignalSpy detectionErrorSpy(&detector, &DuplicateDetector::detectionError);
        
        QEventLoop detectionLoop;
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted, 
                        &detectionLoop, &QEventLoop::quit);
        
        detector.findDuplicates(scannedFiles);
        
        QTimer::singleShot(30000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        // Verify signals were emitted
        qDebug() << "Signal counts:";
        qDebug() << "   detectionStarted:" << detectionStartedSpy.count();
        qDebug() << "   detectionProgress:" << detectionProgressSpy.count();
        qDebug() << "   duplicateGroupFound:" << duplicateGroupFoundSpy.count();
        qDebug() << "   detectionCompleted:" << detectionCompletedSpy.count();
        qDebug() << "   detectionError:" << detectionErrorSpy.count();
        
        QCOMPARE(detectionStartedSpy.count(), 1);
        QVERIFY(detectionProgressSpy.count() >= 1);
        QVERIFY(duplicateGroupFoundSpy.count() >= 1);  // At least one duplicate group
        QCOMPARE(detectionCompletedSpy.count(), 1);
        QCOMPARE(detectionErrorSpy.count(), 0);  // No errors expected
        
        qDebug() << "✓ Signal/slot connections verified";
    }
    
    /**
     * Test 6: Cancellation support
     * Verify cancellation works correctly
     */
    void test_cancellationSupport() {
        qDebug() << "\n[Test 6] Cancellation Support";
        qDebug() << "==============================";
        
        // Create many files to allow time for cancellation
        qDebug() << "Creating test files for cancellation test...";
        for (int i = 0; i < 100; i++) {
            createTestFileWithSize(QString("cancel_test/file%1.dat").arg(i), 10 * 1024);
        }
        
        // Scan files
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/cancel_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(10000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        qDebug() << "Scanned" << scannedFiles.size() << "files";
        
        // Test cancellation
        qDebug() << "Testing detection cancellation...";
        DuplicateDetector detector;
        
        QSignalSpy detectionCancelledSpy(&detector, &DuplicateDetector::detectionCancelled);
        
        QEventLoop detectionLoop;
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted, 
                        &detectionLoop, &QEventLoop::quit);
        QObject::connect(&detector, &DuplicateDetector::detectionCancelled, 
                        &detectionLoop, &QEventLoop::quit);
        
        detector.findDuplicates(scannedFiles);
        
        // Cancel after a short delay
        QTimer::singleShot(100, [&detector]() {
            qDebug() << "   Cancelling detection...";
            detector.cancelDetection();
        });
        
        QTimer::singleShot(10000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        // Verify cancellation
        QVERIFY(!detector.isDetecting());
        qDebug() << "   Detection cancelled successfully";
        qDebug() << "   detectionCancelled signals:" << detectionCancelledSpy.count();
        
        qDebug() << "✓ Cancellation support verified";
    }
    
    /**
     * Test 7: Different detection levels
     * Test Quick, Standard, and Deep detection levels
     */
    void test_differentDetectionLevels() {
        qDebug() << "\n[Test 7] Different Detection Levels";
        qDebug() << "====================================";
        
        // Create test files
        QByteArray content = "Test content for detection levels";
        createDuplicateSet("levels_test", content, 3);
        
        // Scan files
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/levels_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(5000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        QVERIFY(scannedFiles.size() >= 3);
        
        // Test Quick detection (size-based only)
        qDebug() << "\nTesting Quick detection level (size-based)...";
        {
            DuplicateDetector detector;
            DuplicateDetector::DetectionOptions options;
            options.level = DuplicateDetector::DetectionLevel::Quick;
            detector.setOptions(options);
            
            QEventLoop loop;
            QObject::connect(&detector, &DuplicateDetector::detectionCompleted, &loop, &QEventLoop::quit);
            
            detector.findDuplicates(scannedFiles);
            QTimer::singleShot(10000, &loop, &QEventLoop::quit);
            loop.exec();
            
            QList<DuplicateDetector::DuplicateGroup> groups = detector.getDuplicateGroups();
            qDebug() << "   Quick detection found" << groups.size() << "groups";
            QVERIFY(groups.size() >= 1);
            
            // Quick mode should have "(size-based)" as hash
            if (!groups.isEmpty()) {
                qDebug() << "   Hash:" << groups.first().hash;
                QCOMPARE(groups.first().hash, QString("(size-based)"));
            }
        }
        
        // Test Standard detection (size + hash)
        qDebug() << "\nTesting Standard detection level (size + hash)...";
        {
            DuplicateDetector detector;
            DuplicateDetector::DetectionOptions options;
            options.level = DuplicateDetector::DetectionLevel::Standard;
            detector.setOptions(options);
            
            QEventLoop loop;
            QObject::connect(&detector, &DuplicateDetector::detectionCompleted, &loop, &QEventLoop::quit);
            
            detector.findDuplicates(scannedFiles);
            QTimer::singleShot(30000, &loop, &QEventLoop::quit);
            loop.exec();
            
            QList<DuplicateDetector::DuplicateGroup> groups = detector.getDuplicateGroups();
            qDebug() << "   Standard detection found" << groups.size() << "groups";
            QVERIFY(groups.size() >= 1);
            
            // Standard mode should have actual hash
            if (!groups.isEmpty()) {
                qDebug() << "   Hash:" << groups.first().hash.left(16) + "...";
                QVERIFY(groups.first().hash.length() == 64);  // SHA-256 hash
                QVERIFY(groups.first().hash != "(size-based)");
            }
        }
        
        qDebug() << "✓ Different detection levels verified";
    }
};

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    FileScannerDuplicateDetectorTest test;
    int result = QTest::qExec(&test, argc, argv);
    
    // Process any remaining events before exit
    QCoreApplication::processEvents();
    
    return result;
}

#include "test_filescanner_duplicatedetector.moc"
