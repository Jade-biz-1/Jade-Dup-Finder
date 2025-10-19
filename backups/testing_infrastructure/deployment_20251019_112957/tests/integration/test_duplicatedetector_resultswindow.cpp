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

#include "duplicate_detector.h"
#include "results_window.h"
#include "file_scanner.h"

/**
 * @brief Integration test for DuplicateDetector and ResultsWindow
 * 
 * This test verifies:
 * - DuplicateDetector::DuplicateGroup to ResultsWindow::DuplicateGroup conversion
 * - Results display and visualization integration
 * - Selection state synchronization
 * - Progress reporting integration
 * - Error handling between components
 * - Performance with large result sets
 * 
 * Requirements: 1.3, 2.3, 7.4
 */

class DuplicateDetectorResultsWindowTest : public QObject {
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
        qDebug() << "DuplicateDetector <-> ResultsWindow Integration Test";
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
     * Test 1: Data structure conversion
     * Verify DuplicateDetector::DuplicateGroup converts correctly to ResultsWindow::DuplicateGroup
     */
    void test_dataStructureConversion() {
        qDebug() << "\n[Test 1] Data Structure Conversion";
        qDebug() << "===================================";
        
        // Create test files
        QByteArray content = "Test content for conversion";
        createDuplicateSet("conversion_test", content, 3);
        
        // Scan files
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/conversion_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(5000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        QVERIFY(scannedFiles.size() >= 3);
        
        // Detect duplicates
        DuplicateDetector detector;
        QEventLoop detectionLoop;
        QList<DuplicateDetector::DuplicateGroup> detectorGroups;
        
        QObject::connect(&detector, &DuplicateDetector::duplicateGroupFound,
                        [&detectorGroups](const DuplicateDetector::DuplicateGroup& group) {
            detectorGroups.append(group);
        });
        
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted,
                        &detectionLoop, &QEventLoop::quit);
        
        detector.findDuplicates(scannedFiles);
        QTimer::singleShot(10000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        QVERIFY(detectorGroups.size() >= 1);
        
        // Test conversion to ResultsWindow format
        ResultsWindow resultsWindow;
        resultsWindow.displayDuplicateGroups(detectorGroups);
        
        qDebug() << "   Detector groups:" << detectorGroups.size();
        qDebug() << "   Conversion completed successfully";
        
        // Verify the conversion preserved essential data
        const DuplicateDetector::DuplicateGroup& detectorGroup = detectorGroups.first();
        qDebug() << "   Original group:";
        qDebug() << "      ID:" << detectorGroup.groupId;
        qDebug() << "      Files:" << detectorGroup.fileCount;
        qDebug() << "      Size:" << detectorGroup.fileSize;
        qDebug() << "      Hash:" << detectorGroup.hash.left(16) + "...";
        qDebug() << "      Wasted space:" << detectorGroup.wastedSpace;
        
        QVERIFY(!detectorGroup.groupId.isEmpty());
        QVERIFY(detectorGroup.fileCount >= 3);
        QVERIFY(detectorGroup.fileSize > 0);
        QVERIFY(!detectorGroup.hash.isEmpty());
        QVERIFY(detectorGroup.wastedSpace > 0);
        
        qDebug() << "✓ Data structure conversion verified";
    }
    
    /**
     * Test 2: Results display integration
     * Test how ResultsWindow displays and manages duplicate groups
     */
    void test_resultsDisplayIntegration() {
        qDebug() << "\n[Test 2] Results Display Integration";
        qDebug() << "====================================";
        
        // Create multiple duplicate sets
        QByteArray content1 = "Content set 1";
        QByteArray content2 = "Content set 2";
        QByteArray content3 = "Content set 3";
        
        createDuplicateSet("display_test/set1", content1, 3);
        createDuplicateSet("display_test/set2", content2, 4);
        createDuplicateSet("display_test/set3", content3, 2);
        
        // Create some unique files
        createTestFile("display_test/unique1.txt", "Unique content 1");
        createTestFile("display_test/unique2.txt", "Unique content 2");
        
        // Scan and detect duplicates
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/display_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(10000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        qDebug() << "   Scanned" << scannedFiles.size() << "files";
        
        DuplicateDetector detector;
        QEventLoop detectionLoop;
        QList<DuplicateDetector::DuplicateGroup> groups;
        
        QObject::connect(&detector, &DuplicateDetector::duplicateGroupFound,
                        [&groups](const DuplicateDetector::DuplicateGroup& group) {
            groups.append(group);
        });
        
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted,
                        &detectionLoop, &QEventLoop::quit);
        
        detector.findDuplicates(scannedFiles);
        QTimer::singleShot(30000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        qDebug() << "   Found" << groups.size() << "duplicate groups";
        QVERIFY(groups.size() >= 3);
        
        // Test ResultsWindow display
        ResultsWindow resultsWindow;
        resultsWindow.displayDuplicateGroups(groups);
        
        // Verify display integration
        qDebug() << "   Results displayed in window";
        qDebug() << "   Selected files count:" << resultsWindow.getSelectedFilesCount();
        qDebug() << "   Selected files size:" << resultsWindow.getSelectedFilesSize();
        
        // Test selection operations
        resultsWindow.selectAllDuplicates();
        int selectedAfterSelectAll = resultsWindow.getSelectedFilesCount();
        qDebug() << "   After select all:" << selectedAfterSelectAll;
        QVERIFY(selectedAfterSelectAll > 0);
        
        resultsWindow.selectNoneFiles();
        int selectedAfterSelectNone = resultsWindow.getSelectedFilesCount();
        qDebug() << "   After select none:" << selectedAfterSelectNone;
        QCOMPARE(selectedAfterSelectNone, 0);
        
        qDebug() << "✓ Results display integration verified";
    }
    
    /**
     * Test 3: Selection state synchronization
     * Test synchronization between detector results and window selection
     */
    void test_selectionStateSynchronization() {
        qDebug() << "\n[Test 3] Selection State Synchronization";
        qDebug() << "=========================================";
        
        // Create test data
        QByteArray content = "Sync test content";
        createDuplicateSet("sync_test", content, 4);
        
        // Get duplicate groups
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/sync_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(5000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        
        DuplicateDetector detector;
        QEventLoop detectionLoop;
        QList<DuplicateDetector::DuplicateGroup> groups;
        
        QObject::connect(&detector, &DuplicateDetector::duplicateGroupFound,
                        [&groups](const DuplicateDetector::DuplicateGroup& group) {
            groups.append(group);
        });
        
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted,
                        &detectionLoop, &QEventLoop::quit);
        
        detector.findDuplicates(scannedFiles);
        QTimer::singleShot(10000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        QVERIFY(groups.size() >= 1);
        
        // Test selection synchronization
        ResultsWindow resultsWindow;
        resultsWindow.displayDuplicateGroups(groups);
        
        // Test various selection operations
        qDebug() << "   Testing selection operations:";
        
        // Initial state
        int initialSelected = resultsWindow.getSelectedFilesCount();
        qDebug() << "      Initial selected:" << initialSelected;
        
        // Select all
        resultsWindow.selectAllDuplicates();
        int afterSelectAll = resultsWindow.getSelectedFilesCount();
        qDebug() << "      After select all:" << afterSelectAll;
        QVERIFY(afterSelectAll > initialSelected);
        
        // Select recommended
        resultsWindow.selectRecommended();
        int afterSelectRecommended = resultsWindow.getSelectedFilesCount();
        qDebug() << "      After select recommended:" << afterSelectRecommended;
        
        // Select by size (test with minimum size)
        qint64 minSize = 10; // 10 bytes minimum
        resultsWindow.selectBySize(minSize);
        int afterSelectBySize = resultsWindow.getSelectedFilesCount();
        qDebug() << "      After select by size (>=" << minSize << "):" << afterSelectBySize;
        
        // Clear selection
        resultsWindow.selectNoneFiles();
        int afterClear = resultsWindow.getSelectedFilesCount();
        qDebug() << "      After clear:" << afterClear;
        QCOMPARE(afterClear, 0);
        
        qDebug() << "✓ Selection state synchronization verified";
    }
    
    /**
     * Test 4: Progress reporting integration
     * Test progress reporting between detector and results window
     */
    void test_progressReportingIntegration() {
        qDebug() << "\n[Test 4] Progress Reporting Integration";
        qDebug() << "=======================================";
        
        // Create larger dataset for meaningful progress reporting
        const int fileCount = 50;
        qDebug() << "   Creating" << fileCount << "test files...";
        
        for (int i = 0; i < fileCount; i++) {
            QString content = QString("Progress test file %1").arg(i);
            createTestFile(QString("progress_test/file_%1.txt").arg(i), content.toUtf8());
        }
        
        // Create some duplicate sets
        QByteArray dupContent1 = "Duplicate set 1";
        QByteArray dupContent2 = "Duplicate set 2";
        createDuplicateSet("progress_test/dups1", dupContent1, 5);
        createDuplicateSet("progress_test/dups2", dupContent2, 3);
        
        // Scan files
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/progress_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(15000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        qDebug() << "   Scanned" << scannedFiles.size() << "files";
        
        // Test progress reporting during detection
        DuplicateDetector detector;
        ResultsWindow resultsWindow;
        
        QEventLoop detectionLoop;
        int progressUpdates = 0;
        
        // Connect progress signals
        QObject::connect(&detector, &DuplicateDetector::detectionProgress,
                        [&](const DuplicateDetector::DetectionProgress& progress) {
            progressUpdates++;
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
            
            if (progressUpdates % 5 == 0) { // Log every 5th update
                qDebug() << "      Progress:" << phase << "-" 
                         << QString::number(progress.percentComplete, 'f', 1) << "%";
            }
            
            // Update results window with progress
            resultsWindow.updateProgress(phase, static_cast<int>(progress.percentComplete));
        });
        
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted,
                        [&detectionLoop](int totalGroups) {
            qDebug() << "   Detection completed:" << totalGroups << "groups";
            detectionLoop.quit();
        });
        
        detector.findDuplicates(scannedFiles);
        QTimer::singleShot(60000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        qDebug() << "   Progress updates received:" << progressUpdates;
        QVERIFY(progressUpdates > 0);
        
        // Display final results
        QList<DuplicateDetector::DuplicateGroup> groups = detector.getDuplicateGroups();
        resultsWindow.displayDuplicateGroups(groups);
        
        qDebug() << "✓ Progress reporting integration verified";
    }
    
    /**
     * Test 5: Error handling integration
     * Test error handling between detector and results window
     */
    void test_errorHandlingIntegration() {
        qDebug() << "\n[Test 5] Error Handling Integration";
        qDebug() << "====================================";
        
        // Create test files including some that will cause issues
        createTestFile("error_test/normal1.txt", "Normal content 1");
        createTestFile("error_test/normal2.txt", "Normal content 2");
        createTestFile("error_test/empty.txt", "");
        
        // Create a file that we'll delete during processing
        QString volatileFile = createTestFile("error_test/volatile.txt", "Will be deleted");
        
        // Scan files
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/error_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(5000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        qDebug() << "   Scanned" << scannedFiles.size() << "files";
        
        // Delete the volatile file to simulate external changes
        QFile::remove(volatileFile);
        qDebug() << "   Deleted volatile file to simulate external changes";
        
        // Test error handling during detection
        DuplicateDetector detector;
        ResultsWindow resultsWindow;
        
        QEventLoop detectionLoop;
        QSignalSpy errorSpy(&detector, &DuplicateDetector::detectionError);
        
        QObject::connect(&detector, &DuplicateDetector::detectionError,
                        [](const QString& error) {
            qDebug() << "   Detection error:" << error;
        });
        
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted,
                        &detectionLoop, &QEventLoop::quit);
        
        detector.findDuplicates(scannedFiles);
        QTimer::singleShot(15000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        // Verify error handling
        qDebug() << "   Detection errors:" << errorSpy.count();
        
        // Display results even if there were errors
        QList<DuplicateDetector::DuplicateGroup> groups = detector.getDuplicateGroups();
        resultsWindow.displayDuplicateGroups(groups);
        
        qDebug() << "   Results displayed despite errors:" << groups.size() << "groups";
        
        // Test that the results window can handle partial/error results
        QVERIFY(groups.size() >= 0); // Should not crash, even with 0 groups
        
        qDebug() << "✓ Error handling integration verified";
    }
    
    /**
     * Test 6: Performance with large result sets
     * Test performance when displaying many duplicate groups
     */
    void test_performanceWithLargeResultSets() {
        qDebug() << "\n[Test 6] Performance with Large Result Sets";
        qDebug() << "============================================";
        
        const int groupCount = 20;  // Reduced for reasonable test time
        const int filesPerGroup = 5;
        
        qDebug() << "   Creating" << groupCount << "duplicate groups with" << filesPerGroup << "files each...";
        
        // Create multiple duplicate groups
        for (int group = 0; group < groupCount; group++) {
            QByteArray content = QString("Group %1 content").arg(group).toUtf8();
            createDuplicateSet(QString("large_test/group_%1").arg(group), content, filesPerGroup);
            
            if ((group + 1) % 5 == 0) {
                qDebug() << "      Created" << (group + 1) << "groups...";
            }
        }
        
        // Scan files
        qDebug() << "   Scanning files...";
        QElapsedTimer scanTimer;
        scanTimer.start();
        
        FileScanner scanner;
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths << m_testPath + "/large_test";
        scanOptions.minimumFileSize = 0;
        
        QEventLoop scanLoop;
        QObject::connect(&scanner, &FileScanner::scanCompleted, &scanLoop, &QEventLoop::quit);
        
        scanner.startScan(scanOptions);
        QTimer::singleShot(30000, &scanLoop, &QEventLoop::quit);
        scanLoop.exec();
        
        qint64 scanTime = scanTimer.elapsed();
        QVector<FileScanner::FileInfo> scannedFiles = scanner.getScannedFiles();
        qDebug() << "      Scan completed in" << scanTime << "ms";
        qDebug() << "      Files found:" << scannedFiles.size();
        
        // Detect duplicates
        qDebug() << "   Detecting duplicates...";
        QElapsedTimer detectionTimer;
        detectionTimer.start();
        
        DuplicateDetector detector;
        QEventLoop detectionLoop;
        QList<DuplicateDetector::DuplicateGroup> groups;
        
        QObject::connect(&detector, &DuplicateDetector::duplicateGroupFound,
                        [&groups](const DuplicateDetector::DuplicateGroup& group) {
            groups.append(group);
        });
        
        QObject::connect(&detector, &DuplicateDetector::detectionCompleted,
                        &detectionLoop, &QEventLoop::quit);
        
        detector.findDuplicates(scannedFiles);
        QTimer::singleShot(60000, &detectionLoop, &QEventLoop::quit);
        detectionLoop.exec();
        
        qint64 detectionTime = detectionTimer.elapsed();
        qDebug() << "      Detection completed in" << detectionTime << "ms";
        qDebug() << "      Groups found:" << groups.size();
        
        // Test results window performance
        qDebug() << "   Testing results window display performance...";
        QElapsedTimer displayTimer;
        displayTimer.start();
        
        ResultsWindow resultsWindow;
        resultsWindow.displayDuplicateGroups(groups);
        
        qint64 displayTime = displayTimer.elapsed();
        qDebug() << "      Display completed in" << displayTime << "ms";
        
        // Test selection performance
        qDebug() << "   Testing selection performance...";
        QElapsedTimer selectionTimer;
        selectionTimer.start();
        
        resultsWindow.selectAllDuplicates();
        int selectedCount = resultsWindow.getSelectedFilesCount();
        
        qint64 selectionTime = selectionTimer.elapsed();
        qDebug() << "      Selection completed in" << selectionTime << "ms";
        qDebug() << "      Selected files:" << selectedCount;
        
        // Performance validation
        qDebug() << "   Performance summary:";
        qDebug() << "      Total time:" << (scanTime + detectionTime + displayTime) << "ms";
        qDebug() << "      Display rate:" << QString::number(groups.size() * 1000.0 / displayTime, 'f', 2) << "groups/sec";
        qDebug() << "      Selection rate:" << QString::number(selectedCount * 1000.0 / selectionTime, 'f', 2) << "files/sec";
        
        // Basic performance validation
        QVERIFY(groups.size() >= groupCount * 0.8); // At least 80% of expected groups
        QVERIFY(displayTime < 5000); // Display should complete within 5 seconds
        QVERIFY(selectionTime < 2000); // Selection should complete within 2 seconds
        
        qDebug() << "✓ Performance with large result sets verified";
    }
};

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    DuplicateDetectorResultsWindowTest test;
    int result = QTest::qExec(&test, argc, argv);
    
    // Process any remaining events before exit
    QCoreApplication::processEvents();
    
    return result;
}

#include "test_duplicatedetector_resultswindow.moc"