#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QElapsedTimer>
#include <QRandomGenerator>
#include <QThread>
#include <QTextStream>
#include "hash_calculator.h"

class IOOptimizationTester : public QObject {
    Q_OBJECT

public:
    IOOptimizationTester(QObject* parent = nullptr) : QObject(parent) {
        m_calculator = new HashCalculator(this);
        
        // Connect signals for monitoring
        connect(m_calculator, &HashCalculator::hashCompleted,
                this, &IOOptimizationTester::onHashCompleted);
        
        connect(m_calculator, &HashCalculator::hashError,
                this, &IOOptimizationTester::onHashError);
        
        connect(m_calculator, &HashCalculator::allOperationsComplete,
                this, &IOOptimizationTester::onAllComplete);
    }
    
    void runTests() {
        qDebug() << "=== HashCalculator I/O Optimization Tests (HC-002c) ===";
        
        // Create test directory and files
        createTestFiles();
        
        // Test 1: Baseline performance without I/O optimizations
        testBaseline();
        
        // Test 2: Memory mapping for large files
        testMemoryMapping();
        
        // Test 3: Read-ahead caching
        testReadAhead();
        
        // Test 4: Asynchronous I/O
        testAsyncIO();
        
        // Test 5: Combined I/O optimizations
        testCombinedOptimizations();
        
        // Cleanup
        cleanupTestFiles();
        
        qDebug() << "=== All I/O Optimization Tests Complete ===";
    }

private slots:
    void onHashCompleted(const HashCalculator::HashResult& result) {
        if (result.success) {
            m_completedHashes++;
            qDebug() << "Hash completed:" << result.filePath.split('/').last() 
                     << "->" << result.hash.left(16) + "..."
                     << "Size:" << (result.fileSize / 1024) << "KB"
                     << "From cache:" << result.fromCache;
        }
    }
    
    void onHashError(const QString& filePath, const QString& error) {
        qWarning() << "Hash error for" << filePath << ":" << error;
        m_errorCount++;
    }
    
    void onAllComplete() {
        qDebug() << "All operations completed";
        m_allComplete = true;
    }

private:
    void createTestFiles() {
        qDebug() << "\n--- Creating Test Files ---";
        
        m_testDir = QDir::temp();
        QString testDirName = "cloneclean_io_test";
        
        if (m_testDir.exists(testDirName)) {
            QDir(m_testDir.absoluteFilePath(testDirName)).removeRecursively();
        }
        
        m_testDir.mkdir(testDirName);
        m_testDir.cd(testDirName);
        
        qDebug() << "Test directory:" << m_testDir.absolutePath();
        
        // Create files of different sizes for I/O testing
        createTestFile("small_file.txt", 1024);           // 1KB
        createTestFile("medium_file.txt", 100 * 1024);    // 100KB
        createTestFile("large_file.txt", 1024 * 1024);    // 1MB
        createTestFile("huge_file.txt", 5 * 1024 * 1024); // 5MB
        
        // Create some binary files for variety
        createBinaryFile("binary_small.bin", 4096);         // 4KB
        createBinaryFile("binary_medium.bin", 256 * 1024);  // 256KB
        createBinaryFile("binary_large.bin", 2 * 1024 * 1024); // 2MB
        
        m_testFiles = {
            m_testDir.absoluteFilePath("small_file.txt"),
            m_testDir.absoluteFilePath("medium_file.txt"),
            m_testDir.absoluteFilePath("large_file.txt"),
            m_testDir.absoluteFilePath("huge_file.txt"),
            m_testDir.absoluteFilePath("binary_small.bin"),
            m_testDir.absoluteFilePath("binary_medium.bin"),
            m_testDir.absoluteFilePath("binary_large.bin")
        };
        
        qDebug() << "Created" << m_testFiles.size() << "test files";
    }
    
    void createTestFile(const QString& fileName, qint64 size) {
        QFile file(m_testDir.absoluteFilePath(fileName));
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            
            const QString pattern = "This is test data for I/O optimization benchmarking. ";
            qint64 written = 0;
            while (written < size) {
                out << pattern;
                written += pattern.length();
                
                // Add some variation
                if (written % 1000 == 0) {
                    out << QString::number(written) << " ";
                }
            }
            
            qDebug() << "Created" << fileName << "(" << (size / 1024) << "KB)";
        }
    }
    
    void createBinaryFile(const QString& fileName, qint64 size) {
        QFile file(m_testDir.absoluteFilePath(fileName));
        if (file.open(QIODevice::WriteOnly)) {
            QByteArray data(1024, 0);
            
            // Fill with random-ish data
            for (int i = 0; i < data.size(); ++i) {
                data[i] = static_cast<char>(QRandomGenerator::global()->bounded(256));
            }
            
            qint64 written = 0;
            while (written < size) {
                qint64 toWrite = qMin(static_cast<qint64>(data.size()), size - written);
                qint64 actuallyWritten = file.write(data.constData(), toWrite);
                if (actuallyWritten == -1) break;
                written += actuallyWritten;
            }
            
            qDebug() << "Created" << fileName << "(" << (size / 1024) << "KB)";
        }
    }
    
    void testBaseline() {
        qDebug() << "\n--- Test 1: Baseline Performance (No I/O Optimizations) ---";
        
        // Configure for baseline
        HashCalculator::HashOptions options = m_calculator->getOptions();
        options.enableIOOptimizations = false;
        options.enableCaching = false;
        m_calculator->setOptions(options);
        
        runTestSuite("Baseline");
    }
    
    void testMemoryMapping() {
        qDebug() << "\n--- Test 2: Memory Mapping for Large Files ---";
        
        // Configure I/O optimization with memory mapping
        HashCalculator::HashOptions options = m_calculator->getOptions();
        options.enableIOOptimizations = true;
        options.enableMemoryMapping = true;
        options.memoryMapThreshold = 50 * 1024;  // Files > 50KB use memory mapping
        options.enableReadAhead = false;
        options.enableAsyncIO = false;
        options.enableCaching = false;
        m_calculator->setOptions(options);
        
        m_calculator->setIOOptimizations(true);
        m_calculator->setMemoryMappingEnabled(true);
        
        runTestSuite("Memory Mapping");
    }
    
    void testReadAhead() {
        qDebug() << "\n--- Test 3: Read-Ahead Caching ---";
        
        // Configure with read-ahead
        HashCalculator::HashOptions options = m_calculator->getOptions();
        options.enableIOOptimizations = true;
        options.enableMemoryMapping = false;
        options.enableReadAhead = true;
        options.readAheadSize = 16384;  // 16KB buffer
        options.enableAsyncIO = false;
        options.enableCaching = false;
        m_calculator->setOptions(options);
        
        m_calculator->setIOOptimizations(true);
        m_calculator->setReadAheadEnabled(true);
        
        runTestSuite("Read-Ahead");
    }
    
    void testAsyncIO() {
        qDebug() << "\n--- Test 4: Asynchronous I/O ---";
        
        // Configure with async I/O
        HashCalculator::HashOptions options = m_calculator->getOptions();
        options.enableIOOptimizations = true;
        options.enableMemoryMapping = false;
        options.enableReadAhead = false;
        options.enableAsyncIO = true;
        options.maxConcurrentReads = 4;  // Use 4 concurrent reads for async I/O
        options.enableCaching = false;
        m_calculator->setOptions(options);
        
        m_calculator->setIOOptimizations(true);
        m_calculator->setAsyncIOEnabled(true);
        
        runTestSuite("Async I/O");
    }
    
    void testCombinedOptimizations() {
        qDebug() << "\n--- Test 5: Combined I/O Optimizations ---";
        
        // Configure with all optimizations enabled
        HashCalculator::HashOptions options = m_calculator->getOptions();
        options.enableIOOptimizations = true;
        options.enableMemoryMapping = true;
        options.memoryMapThreshold = 100 * 1024;  // Files > 100KB use memory mapping
        options.enableReadAhead = true;
        options.readAheadSize = 32768;  // 32KB buffer
        options.enableAsyncIO = true;
        options.maxConcurrentReads = 6;  // Use 6 concurrent reads for async I/O
        options.enableBufferPooling = true;
        options.enableIOOptimizations = true;  // Enable all I/O optimizations
        options.enableCaching = false;
        m_calculator->setOptions(options);
        
        m_calculator->setIOOptimizations(true);
        m_calculator->setMemoryMappingEnabled(true);
        m_calculator->setReadAheadEnabled(true);
        m_calculator->setAsyncIOEnabled(true);
        
        runTestSuite("Combined Optimizations");
    }
    
    void runTestSuite(const QString& testName) {
        qDebug() << "Starting" << testName << "test with" << m_testFiles.size() << "files";
        
        // Reset counters
        m_completedHashes = 0;
        m_errorCount = 0;
        m_allComplete = false;
        
        QElapsedTimer timer;
        timer.start();
        
        // Clear cache to ensure fair comparison
        m_calculator->clearCache();
        m_calculator->resetStatistics();
        
        // Start processing
        for (const QString& filePath : m_testFiles) {
            m_calculator->calculateFileHash(filePath);
        }
        
        // Wait for completion
        while (!m_allComplete && timer.elapsed() < 30000) {  // 30 second timeout
            QCoreApplication::processEvents();
            QThread::msleep(10);
        }
        
        qint64 elapsedTime = timer.elapsed();
        
        // Get final statistics
        HashCalculator::Statistics stats = m_calculator->getStatistics();
        
        qDebug() << "\n" << testName << "Results:";
        qDebug() << "  Elapsed Time:" << elapsedTime << "ms";
        qDebug() << "  Completed Hashes:" << m_completedHashes;
        qDebug() << "  Errors:" << m_errorCount;
        qDebug() << "  Total Bytes Processed:" << (stats.totalBytesProcessed / 1024) << "KB";
        
        if (elapsedTime > 0) {
            double throughput = (stats.totalBytesProcessed / 1024.0 / 1024.0) / (elapsedTime / 1000.0);
            qDebug() << "  Throughput:" << QString::number(throughput, 'f', 2) << "MB/s";
        }
        
        // I/O specific statistics
        qDebug() << "  I/O Statistics:";
        qDebug() << "    I/O Errors:" << stats.ioErrors;
        qDebug() << "    Memory Mapped Reads:" << stats.memoryMappedReads;
        qDebug() << "    Memory Mapped Bytes:" << (stats.memoryMappedBytes / 1024) << "KB";
        qDebug() << "    Memory Mapping Fallbacks:" << stats.memoryMappingFallbacks;
        qDebug() << "    Read-Ahead Operations:" << stats.readAheadOperations;
        qDebug() << "    Read-Ahead Bytes:" << (stats.readAheadBytes / 1024) << "KB";
        qDebug() << "    Async I/O Operations:" << stats.asyncIOOperations;
        qDebug() << "    Async I/O Bytes:" << (stats.asyncIOBytes / 1024) << "KB";
        qDebug() << "    Buffer Pool Hits:" << stats.bufferHits;
        qDebug() << "    Buffer Pool Misses:" << stats.bufferMisses;
        qDebug() << "    Total I/O Time:" << stats.totalIOTime << "ms";
        
        if (stats.totalIOTime > 0) {
            double ioSpeed = (stats.memoryMappedBytes + stats.readAheadBytes + stats.asyncIOBytes) / 1024.0 / 1024.0 / (stats.totalIOTime / 1000.0);
            qDebug() << "    Average I/O Speed:" << QString::number(ioSpeed, 'f', 2) << "MB/s";
        }
        
        // Analyze I/O configuration effectiveness
        HashCalculator::IOOptimizationConfig ioConfig = m_calculator->getIOConfig();
        qDebug() << "  I/O Configuration:";
        qDebug() << "    Optimizations Enabled:" << ioConfig.enabled;
        qDebug() << "    Memory Mapping:" << ioConfig.memoryMappingEnabled;
        qDebug() << "    Read-Ahead:" << ioConfig.readAheadEnabled;
        qDebug() << "    Async I/O:" << ioConfig.asyncIOEnabled;
        qDebug() << "    Buffer Pool:" << ioConfig.bufferPoolEnabled;
        
        QThread::msleep(500);  // Brief pause between tests
    }
    
    void cleanupTestFiles() {
        qDebug() << "\n--- Cleaning Up Test Files ---";
        
        for (const QString& filePath : m_testFiles) {
            QFile::remove(filePath);
        }
        
        m_testDir.cdUp();
        m_testDir.rmdir("cloneclean_io_test");
        
        qDebug() << "Test files cleaned up";
    }
    
    HashCalculator* m_calculator;
    QDir m_testDir;
    QStringList m_testFiles;
    int m_completedHashes = 0;
    int m_errorCount = 0;
    bool m_allComplete = false;
};

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    
    qDebug() << "HashCalculator I/O Optimization Test Suite";
    qDebug() << "==========================================";
    
    IOOptimizationTester tester;
    tester.runTests();
    
    return 0;
}

#include "test_hc002c_io_optimization.moc"