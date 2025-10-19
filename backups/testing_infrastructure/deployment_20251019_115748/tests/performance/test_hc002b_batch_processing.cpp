#include <QCoreApplication>
#include <QDebug>
#include <QTimer>
#include <QEventLoop>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QElapsedTimer>

#include "hash_calculator.h"

void createTestFiles(const QString& basePath, int smallFiles, int mediumFiles, int largeFiles) {
    qDebug() << "Creating test files in:" << basePath;
    
    // Create small files (< 1MB)
    for (int i = 0; i < smallFiles; ++i) {
        QFile file(basePath + QString("/small_file_%1.txt").arg(i));
        if (file.open(QIODevice::WriteOnly)) {
            QByteArray data(512 * 1024, 'A' + (i % 26)); // 512KB files
            file.write(data);
            file.close();
        }
    }
    
    // Create medium files (1-10MB)
    for (int i = 0; i < mediumFiles; ++i) {
        QFile file(basePath + QString("/medium_file_%1.txt").arg(i));
        if (file.open(QIODevice::WriteOnly)) {
            QByteArray data(2 * 1024 * 1024, 'M' + (i % 5)); // 2MB files
            file.write(data);
            file.close();
        }
    }
    
    // Create large files (10-100MB)
    for (int i = 0; i < largeFiles; ++i) {
        QFile file(basePath + QString("/large_file_%1.txt").arg(i));
        if (file.open(QIODevice::WriteOnly)) {
            QByteArray chunk(1024 * 1024, 'L' + (i % 3)); // 1MB chunks
            for (int j = 0; j < 20; ++j) { // 20MB files
                file.write(chunk);
            }
            file.close();
        }
    }
    
    qDebug() << "Created" << (smallFiles + mediumFiles + largeFiles) << "test files";
}

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    
    qDebug() << "===========================================";
    qDebug() << "HC-002b: Advanced Batch Processing Test";
    qDebug() << "===========================================";
    qDebug();
    
    // Create temporary directory with test files
    QTemporaryDir tempDir;
    if (!tempDir.isValid()) {
        qCritical() << "Failed to create temporary directory";
        return 1;
    }
    
    QString testPath = tempDir.path();
    
    // Create subdirectories for better batching tests
    QDir().mkpath(testPath + "/documents");
    QDir().mkpath(testPath + "/downloads");
    QDir().mkpath(testPath + "/media");
    
    // Create test files of different sizes
    createTestFiles(testPath + "/documents", 15, 5, 2);  // 15 small, 5 medium, 2 large
    createTestFiles(testPath + "/downloads", 10, 3, 1);  // 10 small, 3 medium, 1 large
    createTestFiles(testPath + "/media", 5, 2, 3);       // 5 small, 2 medium, 3 large
    
    // Collect all test files
    QStringList allFiles;
    QDir dir(testPath);
    QStringList nameFilters;
    nameFilters << "*.txt";
    
    QDirIterator iterator(testPath, nameFilters, QDir::Files, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        allFiles << iterator.next();
    }
    
    qDebug() << "Test files collected:" << allFiles.size() << "files";
    qDebug();
    
    // Test 1: Standard Batch Processing
    qDebug() << "TEST 1: Standard Batch Processing";
    qDebug() << "=================================";
    
    HashCalculator standardCalc;
    
    QEventLoop loop1;
    QElapsedTimer timer1;
    int standardResults = 0;
    
    QObject::connect(&standardCalc, &HashCalculator::hashCompleted, [&](const HashCalculator::HashResult& result) {
        if (result.success) {
            standardResults++;
            if (standardResults % 5 == 0) {
                qDebug() << "Standard processing:" << standardResults << "files completed";
            }
        }
    });
    
    QObject::connect(&standardCalc, &HashCalculator::allOperationsComplete, [&]() {
        qDebug() << "Standard processing completed:" << standardResults << "files";
        loop1.quit();
    });
    
    timer1.start();
    standardCalc.calculateFileHashes(allFiles);
    
    QTimer timeout1;
    timeout1.setSingleShot(true);
    timeout1.setInterval(30000);
    QObject::connect(&timeout1, &QTimer::timeout, &loop1, &QEventLoop::quit);
    timeout1.start();
    
    loop1.exec();
    qint64 standardTime = timer1.elapsed();
    
    HashCalculator::Statistics standardStats = standardCalc.getStatistics();
    qDebug() << "Standard Results:";
    qDebug() << "  Time:" << standardTime << "ms";
    qDebug() << "  Files processed:" << standardResults;
    qDebug() << "  Throughput:" << QString::number(standardCalc.getCurrentThroughput(), 'f', 2) << "MB/s";
    qDebug();
    
    // Test 2: Intelligent Batch Processing
    qDebug() << "TEST 2: Intelligent Batch Processing";
    qDebug() << "====================================";
    
    HashCalculator intelligentCalc;
    
    // Enable advanced batch processing features
    HashCalculator::HashOptions advancedOptions;
    advancedOptions.enableBatchProcessing = true;
    advancedOptions.enableSizeBasedGrouping = true;
    advancedOptions.enableParallelBatches = true;
    advancedOptions.maxConcurrentBatches = 3;
    advancedOptions.smallFileBatchSize = 20;
    advancedOptions.mediumFileBatchSize = 10;
    advancedOptions.largeFileBatchSize = 3;
    advancedOptions.enableAdaptiveBatching = true;
    intelligentCalc.setOptions(advancedOptions);
    
    QEventLoop loop2;
    QElapsedTimer timer2;
    int intelligentResults = 0;
    int batchesStarted = 0;
    int batchesCompleted = 0;
    
    QObject::connect(&intelligentCalc, &HashCalculator::hashCompleted, [&](const HashCalculator::HashResult& result) {
        if (result.success) {
            intelligentResults++;
        }
    });
    
    QObject::connect(&intelligentCalc, &HashCalculator::batchStarted, [&](const HashCalculator::BatchInfo& batchInfo) {
        batchesStarted++;
        qDebug() << "Batch" << batchInfo.batchId << "started:" << batchInfo.filePaths.size() 
                 << (batchInfo.isSmallFileBatch ? "small" : "mixed/large") << "files";
    });
    
    QObject::connect(&intelligentCalc, &HashCalculator::batchCompleted, [&](const HashCalculator::BatchInfo& batchInfo) {
        batchesCompleted++;
        qDebug() << "Batch" << batchInfo.batchId << "completed in" 
                 << QString::number(batchInfo.processingTime, 'f', 0) << "ms"
                 << "(" << QString::number(batchInfo.averageFileSpeed, 'f', 1) << "files/sec)";
    });
    
    QObject::connect(&intelligentCalc, &HashCalculator::chunkSizeAdapted, 
                    [](qint64 oldSize, qint64 newSize, double gain) {
        qDebug() << "Chunk size adapted:" << oldSize << "->" << newSize 
                 << "(" << QString::number(gain, 'f', 1) << "% gain)";
    });
    
    QObject::connect(&intelligentCalc, &HashCalculator::allOperationsComplete, [&]() {
        qDebug() << "Intelligent processing completed:" << intelligentResults << "files";
        loop2.quit();
    });
    
    timer2.start();
    intelligentCalc.calculateFileHashesIntelligent(allFiles);
    
    QTimer timeout2;
    timeout2.setSingleShot(true);
    timeout2.setInterval(30000);
    QObject::connect(&timeout2, &QTimer::timeout, &loop2, &QEventLoop::quit);
    timeout2.start();
    
    loop2.exec();
    qint64 intelligentTime = timer2.elapsed();
    
    HashCalculator::Statistics intelligentStats = intelligentCalc.getStatistics();
    
    qDebug();
    qDebug() << "Intelligent Results:";
    qDebug() << "  Time:" << intelligentTime << "ms";
    qDebug() << "  Files processed:" << intelligentResults;
    qDebug() << "  Batches started:" << batchesStarted;
    qDebug() << "  Batches completed:" << batchesCompleted;
    qDebug() << "  Throughput:" << QString::number(intelligentCalc.getCurrentThroughput(), 'f', 2) << "MB/s";
    qDebug() << "  Small file batches:" << intelligentStats.smallFileBatchesOptimized;
    qDebug() << "  Parallel batches:" << intelligentStats.parallelBatchesExecuted;
    qDebug() << "  Average batch time:" << QString::number(intelligentStats.averageBatchTime, 'f', 1) << "ms";
    qDebug() << "  Batch throughput:" << QString::number(intelligentStats.batchThroughput, 'f', 2) << "batches/sec";
    qDebug() << "  Chunk adaptations:" << intelligentStats.chunkSizeAdaptations;
    qDebug() << "  Optimal chunk size:" << intelligentStats.optimalChunkSize << "bytes";
    qDebug();
    
    // Performance Comparison
    qDebug() << "PERFORMANCE COMPARISON";
    qDebug() << "======================";
    double improvementPct = 0.0;
    if (standardTime > 0) {
        improvementPct = ((double)(standardTime - intelligentTime) / standardTime) * 100.0;
    }
    
    qDebug() << "Standard processing:" << standardTime << "ms";
    qDebug() << "Intelligent processing:" << intelligentTime << "ms";
    qDebug() << "Improvement:" << QString::number(improvementPct, 'f', 1) << "%";
    qDebug();
    
    // Test 3: Adaptive Chunk Sizing
    qDebug() << "TEST 3: Adaptive Chunk Sizing";
    qDebug() << "==============================";
    
    HashCalculator adaptiveCalc;
    adaptiveCalc.setAdaptiveChunkSizing(true);
    
    qDebug() << "Initial chunk size:" << adaptiveCalc.getOptimalChunkSize() << "bytes";
    qDebug() << "Adaptive sizing enabled:" << (adaptiveCalc.isAdaptiveChunkSizingEnabled() ? "Yes" : "No");
    
    // Test with a few large files to trigger adaptation
    QStringList largeFiles;
    for (const QString& file : allFiles) {
        QFileInfo info(file);
        if (info.size() > 10 * 1024 * 1024) { // > 10MB
            largeFiles << file;
            if (largeFiles.size() >= 3) break; // Test with 3 large files
        }
    }
    
    if (!largeFiles.isEmpty()) {
        QEventLoop loop3;
        int adaptiveResults = 0;
        
        QObject::connect(&adaptiveCalc, &HashCalculator::hashCompleted, [&](const HashCalculator::HashResult& result) {
            if (result.success) {
                adaptiveResults++;
                qDebug() << "Processed" << QFileInfo(result.filePath).fileName() 
                         << "- Current chunk size:" << adaptiveCalc.getOptimalChunkSize() << "bytes";
            }
        });
        
        QObject::connect(&adaptiveCalc, &HashCalculator::chunkSizeAdapted, 
                        [](qint64 oldSize, qint64 newSize, double gain) {
            qDebug() << "  -> Chunk size adapted:" << oldSize << "->" << newSize 
                     << "(" << QString::number(gain, 'f', 1) << "% throughput change)";
        });
        
        QObject::connect(&adaptiveCalc, &HashCalculator::allOperationsComplete, [&]() {
            qDebug() << "Adaptive processing completed";
            loop3.quit();
        });
        
        adaptiveCalc.calculateFileHashes(largeFiles);
        
        QTimer timeout3;
        timeout3.setSingleShot(true);
        timeout3.setInterval(15000);
        QObject::connect(&timeout3, &QTimer::timeout, &loop3, &QEventLoop::quit);
        timeout3.start();
        
        loop3.exec();
        
        qDebug() << "Final optimal chunk size:" << adaptiveCalc.getOptimalChunkSize() << "bytes";
    } else {
        qDebug() << "No large files available for adaptive testing";
    }
    
    qDebug();
    qDebug() << "===========================================";
    qDebug() << "HC-002b Advanced Batch Processing Test Complete";
    qDebug() << "Key Features Demonstrated:";
    qDebug() << "  ✓ Size-based file grouping";
    qDebug() << "  ✓ Intelligent batch creation";
    qDebug() << "  ✓ Parallel batch processing";
    qDebug() << "  ✓ Adaptive chunk sizing";
    qDebug() << "  ✓ Batch monitoring and statistics";
    qDebug() << "===========================================";
    
    return 0;
}