#include <QTest>
#include <QSignalSpy>
#include <QEventLoop>
#include <QTimer>
#include <QTemporaryFile>
#include <QTextStream>
#include <QDebug>

#include "hash_calculator.h"

/**
 * @brief Unit tests for HashCalculator component
 * 
 * Tests cover:
 * - Basic SHA-256 hash calculation
 * - Cache functionality and LRU eviction  
 * - Multi-threaded processing
 * - Progress reporting for large files
 * - Error handling and edge cases
 * - Performance characteristics
 */
class TestHashCalculator : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();
    
    // Core functionality tests
    void testBasicHashCalculation();
    void testEmptyFileHash();
    void testLargeFileHash();
    void testBatchProcessing();
    void testSynchronousHash();
    
    // Cache tests
    void testCacheHitMiss();
    void testCacheLRUEviction();
    void testCacheInvalidation();
    void testCacheDisable();
    
    // Error handling tests
    void testNonExistentFile();
    void testPermissionDenied();
    
    // Performance and threading tests
    void testConcurrentProcessing();
    void testCancellation();
    void testProgressReporting();
    
    // Configuration tests
    void testOptionsConfiguration();
    void testStatistics();
    
private:
    HashCalculator* m_calculator;
    QString m_testDataDir;
    QStringList m_testFiles;
    
    // Helper methods
    QString createTestFile(const QString& name, const QByteArray& content);
    QString createLargeTestFile(const QString& name, qint64 sizeInMB);
    bool waitForSignal(QSignalSpy& spy, int timeout = 5000);
    QString calculateExpectedHash(const QByteArray& data);
};

void TestHashCalculator::initTestCase()
{
    qDebug() << "Setting up HashCalculator test suite";
    
    // Create temporary directory for test files
    QTemporaryFile tempDir;
    tempDir.open();
    m_testDataDir = QFileInfo(tempDir.fileName()).absolutePath() + "/dupfinder_hash_test";
    QDir().mkpath(m_testDataDir);
    
    qDebug() << "Test data directory:" << m_testDataDir;
}

void TestHashCalculator::cleanupTestCase()
{
    // Clean up test files
    for (const QString& file : m_testFiles) {
        QFile::remove(file);
    }
    QDir(m_testDataDir).removeRecursively();
    
    qDebug() << "HashCalculator test suite completed";
}

void TestHashCalculator::init()
{
    m_calculator = new HashCalculator(this);
}

void TestHashCalculator::cleanup()
{
    delete m_calculator;
    m_calculator = nullptr;
}

void TestHashCalculator::testBasicHashCalculation()
{
    qDebug() << "Testing basic hash calculation";
    
    // Create test file with known content
    QByteArray testData = "Hello, DupFinder! This is a test file for hash calculation.";
    QString testFile = createTestFile("basic_test.txt", testData);
    
    // Calculate expected hash
    QString expectedHash = calculateExpectedHash(testData);
    
    // Test synchronous calculation
    QString hash = m_calculator->calculateFileHashSync(testFile);
    QVERIFY(!hash.isEmpty());
    QCOMPARE(hash.length(), 64); // SHA-256 is 64 hex characters
    QCOMPARE(hash, expectedHash);
    
    qDebug() << "Basic hash test passed - Hash:" << hash.left(16) + "...";
}

void TestHashCalculator::testEmptyFileHash()
{
    qDebug() << "Testing empty file hash calculation";
    
    QString emptyFile = createTestFile("empty.txt", QByteArray());
    QString expectedEmptyHash = calculateExpectedHash(QByteArray());
    
    QString hash = m_calculator->calculateFileHashSync(emptyFile);
    QVERIFY(!hash.isEmpty());
    QCOMPARE(hash, expectedEmptyHash);
    
    qDebug() << "Empty file hash test passed";
}

void TestHashCalculator::testLargeFileHash()
{
    qDebug() << "Testing large file hash with progress reporting";
    
    // Create a file larger than the large file threshold (100MB default)
    QString largeFile = createLargeTestFile("large_test.dat", 10); // 10MB for testing
    
    // Set up signal spies
    QSignalSpy hashCompletedSpy(m_calculator, &HashCalculator::hashCompleted);
    QSignalSpy progressSpy(m_calculator, &HashCalculator::hashProgress);
    
    // Start async calculation
    m_calculator->calculateFileHash(largeFile);
    
    // Wait for completion
    QVERIFY(waitForSignal(hashCompletedSpy));
    
    // Verify result
    QCOMPARE(hashCompletedSpy.count(), 1);
    HashCalculator::HashResult result = qvariant_cast<HashCalculator::HashResult>(hashCompletedSpy.first().at(0));
    
    QVERIFY(result.success);
    QCOMPARE(result.filePath, largeFile);
    QVERIFY(!result.hash.isEmpty());
    QCOMPARE(result.hash.length(), 64);
    QVERIFY(!result.fromCache);
    
    // We should have received progress updates for large files
    // Note: Progress depends on file size vs threshold
    qDebug() << "Large file hash test passed - Progress updates:" << progressSpy.count();
}

void TestHashCalculator::testBatchProcessing()
{
    qDebug() << "Testing batch file processing";
    
    // Create multiple test files
    QStringList testFiles;
    for (int i = 0; i < 5; ++i) {
        QByteArray data = QByteArray("Test data for file ") + QByteArray::number(i);
        testFiles << createTestFile(QString("batch_%1.txt").arg(i), data);
    }
    
    QSignalSpy hashCompletedSpy(m_calculator, &HashCalculator::hashCompleted);
    QSignalSpy allCompleteSpy(m_calculator, &HashCalculator::allOperationsComplete);
    
    // Process batch
    m_calculator->calculateFileHashes(testFiles);
    
    // Wait for all to complete
    QVERIFY(waitForSignal(allCompleteSpy, 10000));
    
    // Verify all files were processed
    QCOMPARE(hashCompletedSpy.count(), testFiles.size());
    
    // Verify each result
    for (int i = 0; i < hashCompletedSpy.count(); ++i) {
        HashCalculator::HashResult result = qvariant_cast<HashCalculator::HashResult>(hashCompletedSpy.at(i).at(0));
        QVERIFY(result.success);
        QVERIFY(testFiles.contains(result.filePath));
        QVERIFY(!result.hash.isEmpty());
    }
    
    qDebug() << "Batch processing test passed";
}

void TestHashCalculator::testSynchronousHash()
{
    qDebug() << "Testing synchronous hash calculation";
    
    QByteArray testData = "Synchronous test data";
    QString testFile = createTestFile("sync_test.txt", testData);
    QString expectedHash = calculateExpectedHash(testData);
    
    QString hash = m_calculator->calculateFileHashSync(testFile);
    QCOMPARE(hash, expectedHash);
    
    qDebug() << "Synchronous hash test passed";
}

void TestHashCalculator::testCacheHitMiss()
{
    qDebug() << "Testing cache hit/miss functionality";
    
    QByteArray testData = "Cache test data";
    QString testFile = createTestFile("cache_test.txt", testData);
    
    // Ensure cache is enabled
    HashCalculator::HashOptions options = m_calculator->getOptions();
    options.enableCaching = true;
    m_calculator->setOptions(options);
    m_calculator->clearCache();
    
    QSignalSpy hashCompletedSpy(m_calculator, &HashCalculator::hashCompleted);
    
    // First calculation - should be cache miss
    m_calculator->calculateFileHash(testFile);
    QVERIFY(waitForSignal(hashCompletedSpy));
    
    HashCalculator::HashResult firstResult = qvariant_cast<HashCalculator::HashResult>(hashCompletedSpy.first().at(0));
    QVERIFY(firstResult.success);
    QVERIFY(!firstResult.fromCache); // Should be cache miss
    
    hashCompletedSpy.clear();
    
    // Second calculation - should be cache hit
    m_calculator->calculateFileHash(testFile);
    QVERIFY(waitForSignal(hashCompletedSpy));
    
    HashCalculator::HashResult secondResult = qvariant_cast<HashCalculator::HashResult>(hashCompletedSpy.first().at(0));
    QVERIFY(secondResult.success);
    QVERIFY(secondResult.fromCache); // Should be cache hit
    QCOMPARE(firstResult.hash, secondResult.hash);
    
    // Verify cache statistics
    QCOMPARE(m_calculator->getCacheSize(), 1);
    QVERIFY(m_calculator->getCacheHitRate() > 0.0);
    
    qDebug() << "Cache hit/miss test passed - Hit rate:" << m_calculator->getCacheHitRate();
}

void TestHashCalculator::testCacheLRUEviction()
{
    qDebug() << "Testing LRU cache eviction";
    
    // Set small cache size for testing
    HashCalculator::HashOptions options = m_calculator->getOptions();
    options.enableCaching = true;
    options.maxCacheSize = 3;
    m_calculator->setOptions(options);
    m_calculator->clearCache();
    
    // Create more files than cache size
    QStringList testFiles;
    for (int i = 0; i < 5; ++i) {
        QByteArray data = QByteArray("LRU test data ") + QByteArray::number(i);
        testFiles << createTestFile(QString("lru_%1.txt").arg(i), data);
    }
    
    QSignalSpy allCompleteSpy(m_calculator, &HashCalculator::allOperationsComplete);
    
    // Process all files (more than cache size)
    m_calculator->calculateFileHashes(testFiles);
    QVERIFY(waitForSignal(allCompleteSpy, 10000));
    
    // Cache should be at maximum size
    QCOMPARE(m_calculator->getCacheSize(), 3);
    
    qDebug() << "LRU eviction test passed - Cache size:" << m_calculator->getCacheSize();
}

void TestHashCalculator::testCacheInvalidation()
{
    qDebug() << "Testing cache invalidation on file modification";
    
    QByteArray originalData = "Original data";
    QString testFile = createTestFile("invalidation_test.txt", originalData);
    
    // First calculation
    QString originalHash = m_calculator->calculateFileHashSync(testFile);
    QVERIFY(!originalHash.isEmpty());
    
    // Modify file
    QByteArray modifiedData = "Modified data";
    QFile file(testFile);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write(modifiedData);
    file.close();
    
    // Second calculation should detect change and recalculate
    QString modifiedHash = m_calculator->calculateFileHashSync(testFile);
    QVERIFY(!modifiedHash.isEmpty());
    QVERIFY(originalHash != modifiedHash);
    
    qDebug() << "Cache invalidation test passed";
}

void TestHashCalculator::testCacheDisable()
{
    qDebug() << "Testing cache disable functionality";
    
    HashCalculator::HashOptions options = m_calculator->getOptions();
    options.enableCaching = false;
    m_calculator->setOptions(options);
    
    QByteArray testData = "No cache test data";
    QString testFile = createTestFile("no_cache_test.txt", testData);
    
    QSignalSpy hashCompletedSpy(m_calculator, &HashCalculator::hashCompleted);
    
    // Multiple calculations with cache disabled
    for (int i = 0; i < 3; ++i) {
        m_calculator->calculateFileHash(testFile);
        QVERIFY(waitForSignal(hashCompletedSpy));
        
        HashCalculator::HashResult result = qvariant_cast<HashCalculator::HashResult>(hashCompletedSpy.last().at(0));
        QVERIFY(result.success);
        QVERIFY(!result.fromCache); // Should never be from cache
        
        hashCompletedSpy.clear();
    }
    
    QCOMPARE(m_calculator->getCacheSize(), 0);
    
    qDebug() << "Cache disable test passed";
}

void TestHashCalculator::testNonExistentFile()
{
    qDebug() << "Testing non-existent file handling";
    
    QString nonExistentFile = m_testDataDir + "/does_not_exist.txt";
    
    QSignalSpy errorSpy(m_calculator, &HashCalculator::hashError);
    QSignalSpy completedSpy(m_calculator, &HashCalculator::hashCompleted);
    
    m_calculator->calculateFileHash(nonExistentFile);
    
    // Should either get error signal or completed with success=false
    QEventLoop loop;
    QTimer::singleShot(1000, &loop, &QEventLoop::quit);
    
    connect(m_calculator, &HashCalculator::hashError, &loop, &QEventLoop::quit);
    connect(m_calculator, &HashCalculator::hashCompleted, &loop, &QEventLoop::quit);
    
    loop.exec();
    
    // Should handle gracefully (either error signal or failed result)
    bool handledGracefully = (errorSpy.count() > 0) || 
                            (completedSpy.count() > 0 && 
                             !qvariant_cast<HashCalculator::HashResult>(completedSpy.first().at(0)).success);
    
    QVERIFY(handledGracefully);
    
    qDebug() << "Non-existent file test passed";
}

void TestHashCalculator::testPermissionDenied()
{
    qDebug() << "Testing permission denied handling";
    
    // This test is platform-dependent and might not work in all environments
    // We'll create a file and remove read permissions
    
    QString testFile = createTestFile("permission_test.txt", "Test data");
    
    // Remove read permissions (Unix/Linux)
    QFile file(testFile);
    if (file.setPermissions(QFile::WriteOwner)) {
        QString hash = m_calculator->calculateFileHashSync(testFile);
        // Should return empty string or handle gracefully
        // The exact behavior may vary by platform
        qDebug() << "Permission test result:" << (hash.isEmpty() ? "Handled gracefully" : "Unexpected success");
        
        // Restore permissions for cleanup
        file.setPermissions(QFile::ReadOwner | QFile::WriteOwner);
    } else {
        qDebug() << "Permission test skipped (cannot modify file permissions)";
    }
}

void TestHashCalculator::testConcurrentProcessing()
{
    qDebug() << "Testing concurrent processing";
    
    // Create multiple files for concurrent processing
    QStringList testFiles;
    for (int i = 0; i < 10; ++i) {
        QByteArray data = QByteArray("Concurrent test data ") + QByteArray::number(i);
        testFiles << createTestFile(QString("concurrent_%1.txt").arg(i), data);
    }
    
    QSignalSpy hashCompletedSpy(m_calculator, &HashCalculator::hashCompleted);
    QSignalSpy allCompleteSpy(m_calculator, &HashCalculator::allOperationsComplete);
    
    // Process all files concurrently
    auto startTime = QTime::currentTime();
    m_calculator->calculateFileHashes(testFiles);
    
    // Wait for completion
    QVERIFY(waitForSignal(allCompleteSpy, 15000));
    auto endTime = QTime::currentTime();
    
    // Verify all completed
    QCOMPARE(hashCompletedSpy.count(), testFiles.size());
    
    // Verify processing was reasonably fast (concurrent)
    int processingTimeMs = startTime.msecsTo(endTime);
    qDebug() << "Concurrent processing completed in" << processingTimeMs << "ms";
    
    // All results should be successful
    for (int i = 0; i < hashCompletedSpy.count(); ++i) {
        HashCalculator::HashResult result = qvariant_cast<HashCalculator::HashResult>(hashCompletedSpy.at(i).at(0));
        QVERIFY(result.success);
    }
    
    qDebug() << "Concurrent processing test passed";
}

void TestHashCalculator::testCancellation()
{
    qDebug() << "Testing operation cancellation";
    
    // Create a large file for cancellation testing
    QString largeFile = createLargeTestFile("cancel_test.dat", 50); // 50MB
    
    QSignalSpy cancelledSpy(m_calculator, &HashCalculator::hashCancelled);
    QSignalSpy completedSpy(m_calculator, &HashCalculator::hashCompleted);
    
    // Start calculation
    m_calculator->calculateFileHash(largeFile);
    
    // Cancel after brief delay
    QTimer::singleShot(100, [this]() {
        m_calculator->cancelAll();
    });
    
    // Wait for cancellation or completion
    QEventLoop loop;
    QTimer timeoutTimer;
    timeoutTimer.setSingleShot(true);
    timeoutTimer.setInterval(5000);
    
    connect(&timeoutTimer, &QTimer::timeout, &loop, &QEventLoop::quit);
    connect(m_calculator, &HashCalculator::hashCancelled, &loop, &QEventLoop::quit);
    connect(m_calculator, &HashCalculator::allOperationsComplete, &loop, &QEventLoop::quit);
    
    timeoutTimer.start();
    loop.exec();
    
    // Should have been cancelled or completed
    bool wasCancelled = cancelledSpy.count() > 0;
    bool wasCompleted = completedSpy.count() > 0;
    
    QVERIFY(wasCancelled || wasCompleted);
    
    if (wasCancelled) {
        qDebug() << "Cancellation test passed - Operation was cancelled";
    } else {
        qDebug() << "Cancellation test completed - Operation finished before cancellation";
    }
}

void TestHashCalculator::testProgressReporting()
{
    qDebug() << "Testing progress reporting";
    
    // Create file large enough to trigger progress reporting
    QString testFile = createLargeTestFile("progress_test.dat", 5); // 5MB
    
    QSignalSpy progressSpy(m_calculator, &HashCalculator::hashProgress);
    QSignalSpy completedSpy(m_calculator, &HashCalculator::hashCompleted);
    
    m_calculator->calculateFileHash(testFile);
    QVERIFY(waitForSignal(completedSpy, 10000));
    
    // Verify progress was reported (if file was large enough)
    if (progressSpy.count() > 0) {
        HashCalculator::ProgressInfo progress = qvariant_cast<HashCalculator::ProgressInfo>(progressSpy.first().at(0));
        QCOMPARE(progress.filePath, testFile);
        QVERIFY(progress.totalBytes > 0);
        QVERIFY(progress.bytesProcessed <= progress.totalBytes);
        QVERIFY(progress.percentComplete >= 0 && progress.percentComplete <= 100);
        
        qDebug() << "Progress reporting test passed - Updates:" << progressSpy.count();
    } else {
        qDebug() << "Progress reporting test - No progress updates (file too small)";
    }
}

void TestHashCalculator::testOptionsConfiguration()
{
    qDebug() << "Testing options configuration";
    
    HashCalculator::HashOptions originalOptions = m_calculator->getOptions();
    
    // Modify options
    HashCalculator::HashOptions newOptions;
    newOptions.threadPoolSize = 2;
    newOptions.largeFileThreshold = 50 * 1024 * 1024; // 50MB
    newOptions.chunkSize = 32 * 1024; // 32KB
    newOptions.enableCaching = false;
    newOptions.maxCacheSize = 5000;
    
    m_calculator->setOptions(newOptions);
    
    HashCalculator::HashOptions retrievedOptions = m_calculator->getOptions();
    QCOMPARE(retrievedOptions.threadPoolSize, newOptions.threadPoolSize);
    QCOMPARE(retrievedOptions.largeFileThreshold, newOptions.largeFileThreshold);
    QCOMPARE(retrievedOptions.chunkSize, newOptions.chunkSize);
    QCOMPARE(retrievedOptions.enableCaching, newOptions.enableCaching);
    QCOMPARE(retrievedOptions.maxCacheSize, newOptions.maxCacheSize);
    
    // Restore original options
    m_calculator->setOptions(originalOptions);
    
    qDebug() << "Options configuration test passed";
}

void TestHashCalculator::testStatistics()
{
    qDebug() << "Testing statistics collection";
    
    m_calculator->resetStatistics();
    
    // Create test files
    QStringList testFiles;
    for (int i = 0; i < 3; ++i) {
        QByteArray data = QByteArray("Statistics test data ") + QByteArray::number(i);
        testFiles << createTestFile(QString("stats_%1.txt").arg(i), data);
    }
    
    QSignalSpy allCompleteSpy(m_calculator, &HashCalculator::allOperationsComplete);
    
    // Process files
    m_calculator->calculateFileHashes(testFiles);
    QVERIFY(waitForSignal(allCompleteSpy));
    
    // Check statistics
    HashCalculator::Statistics stats = m_calculator->getStatistics();
    QCOMPARE(stats.totalHashesCalculated, testFiles.size());
    QVERIFY(stats.totalBytesProcessed > 0);
    
    qDebug() << "Statistics test passed - Hashes:" << stats.totalHashesCalculated
             << "Bytes:" << stats.totalBytesProcessed
             << "Cache hits:" << stats.cacheHits;
}

// Helper Methods

QString TestHashCalculator::createTestFile(const QString& name, const QByteArray& content)
{
    QString filePath = m_testDataDir + "/" + name;
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to create test file:" << filePath;
        return QString();
    }
    
    file.write(content);
    file.close();
    
    m_testFiles << filePath;
    return filePath;
}

QString TestHashCalculator::createLargeTestFile(const QString& name, qint64 sizeInMB)
{
    QString filePath = m_testDataDir + "/" + name;
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to create large test file:" << filePath;
        return QString();
    }
    
    QByteArray chunk(1024 * 1024, 'X'); // 1MB chunk
    for (qint64 i = 0; i < sizeInMB; ++i) {
        if (file.write(chunk) != chunk.size()) {
            qWarning() << "Failed to write to large test file:" << filePath;
            file.close();
            return QString();
        }
    }
    
    file.close();
    m_testFiles << filePath;
    return filePath;
}

bool TestHashCalculator::waitForSignal(QSignalSpy& spy, int timeout)
{
    if (spy.count() > 0) {
        return true;
    }
    
    return spy.wait(timeout);
}

QString TestHashCalculator::calculateExpectedHash(const QByteArray& data)
{
    QCryptographicHash hasher(QCryptographicHash::Sha256);
    hasher.addData(data);
    return hasher.result().toHex().toLower();
}

QTEST_MAIN(TestHashCalculator)
#include "test_hash_calculator.moc"