#include <QCoreApplication>
#include <QDebug>
#include <QTimer>
#include <QEventLoop>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>

#include "file_scanner.h"
#include "hash_calculator.h"
#include "duplicate_detector.h"

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    
    qDebug() << "===========================================";
    qDebug() << "DupFinder Integration Test - Full Workflow";
    qDebug() << "===========================================";
    qDebug();
    
    // Create temporary directory with test files
    QTemporaryDir tempDir;
    if (!tempDir.isValid()) {
        qCritical() << "Failed to create temporary directory";
        return 1;
    }
    
    QString testPath = tempDir.path();
    qDebug() << "Test directory:" << testPath;
    
    // Create test file structure with known duplicates
    QByteArray content1 = "This is the content of duplicate file set 1.";
    QByteArray content2 = "This is different content for unique file set 2.";
    QByteArray content3 = "This is the content of duplicate file set 1."; // Same as content1
    QByteArray content4 = "Another unique file with different content entirely.";
    
    // Create files
    QStringList testFiles = {
        testPath + "/documents/file1.txt",      // content1 (duplicate group 1)
        testPath + "/documents/file2.txt",      // content2 (unique)
        testPath + "/downloads/file1_copy.txt", // content3 (same as content1 - duplicate)
        testPath + "/downloads/file3.txt",      // content4 (unique)
        testPath + "/desktop/backup_file1.txt", // content1 again (duplicate)
        testPath + "/temp/empty.txt"            // empty file
    };
    
    // Create directories
    QDir().mkpath(testPath + "/documents");
    QDir().mkpath(testPath + "/downloads");
    QDir().mkpath(testPath + "/desktop");
    QDir().mkpath(testPath + "/temp");
    
    // Write test files
    QFile file1(testFiles[0]); file1.open(QIODevice::WriteOnly); file1.write(content1); file1.close();
    QFile file2(testFiles[1]); file2.open(QIODevice::WriteOnly); file2.write(content2); file2.close();
    QFile file3(testFiles[2]); file3.open(QIODevice::WriteOnly); file3.write(content3); file3.close();
    QFile file4(testFiles[3]); file4.open(QIODevice::WriteOnly); file4.write(content4); file4.close();
    QFile file5(testFiles[4]); file5.open(QIODevice::WriteOnly); file5.write(content1); file5.close(); // Another duplicate
    QFile file6(testFiles[5]); file6.open(QIODevice::WriteOnly); file6.close(); // Empty file
    
    qDebug() << "Created test files:";
    for (const QString& file : testFiles) {
        QFileInfo info(file);
        qDebug() << "   -" << info.fileName() << "(" << info.size() << "bytes)";
    }
    qDebug();
    
    // Phase 1: FileScanner
    qDebug() << "Phase 1: File Scanning";
    qDebug() << "=========================";
    
    FileScanner scanner;
    FileScanner::ScanOptions scanOptions;
    scanOptions.targetPaths << testPath;
    scanOptions.minimumFileSize = 0; // Include empty files
    scanOptions.includeHiddenFiles = false;
    
    QEventLoop scanLoop;
    QList<FileScanner::FileInfo> scannedFiles;
    
    QObject::connect(&scanner, &FileScanner::scanCompleted, [&]() {
        scannedFiles = scanner.getScannedFiles();
        qDebug() << "Scan completed -" << scannedFiles.size() << "files found";
        scanLoop.quit();
    });
    
    QObject::connect(&scanner, &FileScanner::scanProgress, [](int processed, int total, const QString& path) {
        Q_UNUSED(total)
        if (processed % 2 == 0) { // Show some progress
            qDebug() << "   Progress:" << processed << "files processed, current:" << QFileInfo(path).fileName();
        }
    });
    
    scanner.startScan(scanOptions);
    
    QTimer scanTimeout;
    scanTimeout.setSingleShot(true);
    scanTimeout.setInterval(5000);
    QObject::connect(&scanTimeout, &QTimer::timeout, &scanLoop, &QEventLoop::quit);
    scanTimeout.start();
    
    scanLoop.exec();
    
    if (scannedFiles.isEmpty()) {
        qCritical() << "Scan failed or timed out";
        return 1;
    }
    
    qDebug() << "Scan results:";
    for (const FileScanner::FileInfo& file : scannedFiles) {
        qDebug() << "   -" << QFileInfo(file.filePath).fileName() << "(" << file.fileSize << "bytes)";
    }
    qDebug();
    
    // Phase 2: Hash Calculation
    qDebug() << "Phase 2: Hash Calculation";
    qDebug() << "============================";
    
    HashCalculator hashCalc;
    QEventLoop hashLoop;
    QHash<QString, QString> fileHashes;
    int hashesCompleted = 0;
    
    QObject::connect(&hashCalc, &HashCalculator::hashCompleted, [&](const HashCalculator::HashResult& result) {
        if (result.success) {
            fileHashes[result.filePath] = result.hash;
            hashesCompleted++;
            qDebug() << "   Success" << QFileInfo(result.filePath).fileName() << "->" << result.hash.left(16) + "..."
                     << (result.fromCache ? "(cached)" : "(calculated)");
        } else {
            qWarning() << "   Failed" << QFileInfo(result.filePath).fileName() << "failed:" << result.errorMessage;
        }
    });
    
    QObject::connect(&hashCalc, &HashCalculator::allOperationsComplete, [&]() {
        qDebug() << "All hashes calculated";
        hashLoop.quit();
    });
    
    QObject::connect(&hashCalc, &HashCalculator::hashProgress, [](const HashCalculator::ProgressInfo& progress) {
        qDebug() << "   Progress:" << progress.percentComplete << "% -" << QFileInfo(progress.filePath).fileName();
    });
    
    // Calculate hashes for all files
    QStringList filePaths;
    for (const FileScanner::FileInfo& file : scannedFiles) {
        filePaths << file.filePath;
    }
    
    hashCalc.calculateFileHashes(filePaths);
    
    QTimer hashTimeout;
    hashTimeout.setSingleShot(true);
    hashTimeout.setInterval(10000);
    QObject::connect(&hashTimeout, &QTimer::timeout, &hashLoop, &QEventLoop::quit);
    hashTimeout.start();
    
    hashLoop.exec();
    
    if (fileHashes.size() != scannedFiles.size()) {
        qWarning() << "Not all files were hashed successfully";
        qDebug() << "Expected:" << scannedFiles.size() << "Got:" << fileHashes.size();
    }
    
    // Show hash statistics
    HashCalculator::Statistics hashStats = hashCalc.getStatistics();
    qDebug() << "Hash Statistics:";
    qDebug() << "   Total hashes:" << hashStats.totalHashesCalculated;
    qDebug() << "   Cache hits:" << hashStats.cacheHits;
    qDebug() << "   Cache hit rate:" << QString::number(hashCalc.getCacheHitRate() * 100, 'f', 1) + "%";
    qDebug();
    
    // Phase 3: Duplicate Detection
    qDebug() << "Phase 3: Duplicate Detection";
    qDebug() << "===============================";
    
    DuplicateDetector detector;
    QEventLoop detectionLoop;
    QList<DuplicateDetector::DuplicateGroup> duplicateGroups;
    
    QObject::connect(&detector, &DuplicateDetector::detectionStarted, [](int totalFiles) {
        qDebug() << "Detection started for" << totalFiles << "files";
    });
    
    QObject::connect(&detector, &DuplicateDetector::detectionProgress, [](const DuplicateDetector::DetectionProgress& progress) {
        QString phaseNames[] = {"Size Grouping", "Hash Calculation", "Duplicate Grouping", "Generating Recommendations", "Complete"};
        qDebug() << "   Progress" << phaseNames[progress.currentPhase] << ":" << progress.percentComplete << "%"
                 << "(" << progress.filesProcessed << "/" << progress.totalFiles << ")";
        
        if (progress.currentPhase >= DuplicateDetector::DetectionProgress::DuplicateGrouping) {
            qDebug() << "      Size groups:" << progress.sizeGroupsFound 
                     << "Duplicate groups:" << progress.duplicateGroupsFound
                     << "Wasted space:" << progress.wastedSpaceFound << "bytes";
        }
    });
    
    QObject::connect(&detector, &DuplicateDetector::duplicateGroupFound, [](const DuplicateDetector::DuplicateGroup& group) {
        qDebug() << "   Found duplicate group:" << group.groupId.left(8) + "..."
                 << "(" << group.fileCount << "files," << group.wastedSpace << "bytes wasted)";
        qDebug() << "      Recommendation:" << group.recommendedAction;
        for (const DuplicateDetector::FileInfo& file : group.files) {
            qDebug() << "      -" << QFileInfo(file.filePath).fileName() << "(" << file.filePath << ")";
        }
    });
    
    QObject::connect(&detector, &DuplicateDetector::detectionCompleted, [&](int totalGroups) {
        duplicateGroups = detector.getDuplicateGroups();
        qDebug() << "Detection completed -" << totalGroups << "duplicate groups found";
        detectionLoop.quit();
    });
    
    QObject::connect(&detector, &DuplicateDetector::detectionError, [&](const QString& error) {
        qCritical() << "Detection error:" << error;
        detectionLoop.quit();
    });
    
    // Configure detection options
    DuplicateDetector::DetectionOptions detectionOptions;
    detectionOptions.level = DuplicateDetector::DetectionLevel::Standard; // Size + Hash
    detectionOptions.skipEmptyFiles = false; // Include empty files for testing
    detectionOptions.minimumFileSize = 0;
    
    detector.setOptions(detectionOptions);
    
    // Start detection
    detector.findDuplicates(scannedFiles);
    
    QTimer detectionTimeout;
    detectionTimeout.setSingleShot(true);
    detectionTimeout.setInterval(15000);
    QObject::connect(&detectionTimeout, &QTimer::timeout, &detectionLoop, &QEventLoop::quit);
    detectionTimeout.start();
    
    detectionLoop.exec();
    
    // Show detection statistics
    DuplicateDetector::DetectionStatistics detectionStats = detector.getStatistics();
    qDebug() << "Detection Statistics:";
    qDebug() << "   Total files processed:" << detectionStats.totalFilesProcessed;
    qDebug() << "   Files with unique sizes:" << detectionStats.filesWithUniqueSize;
    qDebug() << "   Files in size groups:" << detectionStats.filesInSizeGroups;
    qDebug() << "   Hash calculations:" << detectionStats.hashCalculationsPerformed;
    qDebug() << "   Duplicate groups found:" << detectionStats.duplicateGroupsFound;
    qDebug() << "   Total duplicate files:" << detectionStats.totalDuplicateFiles;
    qDebug() << "   Total wasted space:" << detectionStats.totalWastedSpace << "bytes";
    qDebug() << "   Average group size:" << QString::number(detectionStats.averageGroupSize, 'f', 1);
    qDebug();
    
    // Verification Phase
    qDebug() << "Phase 4: Verification";
    qDebug() << "========================";
    
    bool testPassed = true;
    
    // Verify we found the expected duplicates
    qDebug() << "Verifying duplicate detection results...";
    
    if (duplicateGroups.size() >= 1) {
        qDebug() << "Found" << duplicateGroups.size() << "duplicate groups";
        
        // Find the group with content1 duplicates (should be 3 files)
        bool foundMainDuplicateGroup = false;
        for (const DuplicateDetector::DuplicateGroup& group : duplicateGroups) {
            if (group.fileCount >= 3) {
                foundMainDuplicateGroup = true;
                qDebug() << "Found main duplicate group with" << group.fileCount << "files";
                qDebug() << "   Wasted space:" << group.wastedSpace << "bytes";
                qDebug() << "   Hash:" << group.hash.left(16) + "...";
                break;
            }
        }
        
        if (!foundMainDuplicateGroup) {
            qWarning() << "Did not find expected main duplicate group (3+ files)";
            testPassed = false;
        }
    } else {
        qWarning() << "No duplicate groups found, expected at least 1";
        testPassed = false;
    }
    
    // Verify file processing
    if (detectionStats.totalFilesProcessed >= 6) {
        qDebug() << "Processed expected number of files";
    } else {
        qWarning() << "Did not process expected number of files";
        testPassed = false;
    }
    
    // Verify hash calculations
    if (hashStats.totalHashesCalculated > 0) {
        qDebug() << "Hash calculations performed successfully";
    } else {
        qWarning() << "No hash calculations performed";
        testPassed = false;
    }
    
    qDebug();
    qDebug() << "===========================================";
    if (testPassed) {
        qDebug() << "INTEGRATION TEST PASSED SUCCESSFULLY!";
        qDebug() << "FileScanner -> HashCalculator -> DuplicateDetector workflow verified";
    } else {
        qCritical() << "INTEGRATION TEST FAILED";
        qCritical() << "Some components did not work as expected";
    }
    qDebug() << "===========================================";
    
    return testPassed ? 0 : 1;
}
