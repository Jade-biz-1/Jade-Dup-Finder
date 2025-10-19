#include <QtTest>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QFile>
#include "file_scanner.h"

class ScanProgressTrackingTest : public QObject {
    Q_OBJECT
    
private:
    FileScanner* m_scanner = nullptr;
    QTemporaryDir* m_tempDir = nullptr;
    
    void createTestFile(const QString& relativePath, qint64 size = 1024) {
        QString fullPath = m_tempDir->path() + "/" + relativePath;
        QFileInfo fileInfo(fullPath);
        QDir().mkpath(fileInfo.absolutePath());
        
        QFile file(fullPath);
        QVERIFY(file.open(QIODevice::WriteOnly));
        QByteArray data(static_cast<int>(size), 'X');
        file.write(data);
        file.close();
    }
    
    void createMultipleFiles(int count, qint64 sizePerFile = 1024) {
        for (int i = 0; i < count; ++i) {
            createTestFile(QString("file_%1.txt").arg(i), sizePerFile);
        }
    }

private slots:
    void initTestCase() {
        qRegisterMetaType<FileScanner::ScanProgress>("FileScanner::ScanProgress");
    }
    
    void init() {
        m_scanner = new FileScanner();
        m_tempDir = new QTemporaryDir();
        QVERIFY(m_tempDir->isValid());
    }
    
    void cleanup() {
        delete m_scanner;
        delete m_tempDir;
        m_scanner = nullptr;
        m_tempDir = nullptr;
    }

    void testDetailedProgressSignalEmitted() {
        createMultipleFiles(150);
        QSignalSpy progressSpy(m_scanner, &FileScanner::detailedProgress);
        
        FileScanner::ScanOptions options;
        options.targetPaths = QStringList() << m_tempDir->path();
        options.minimumFileSize = 0;
        options.progressBatchSize = 100;
        
        m_scanner->startScan(options);
        QSignalSpy completedSpy(m_scanner, &FileScanner::scanCompleted);
        QVERIFY(completedSpy.wait(5000));
        
        QVERIFY(progressSpy.count() > 0);
        
        if (progressSpy.count() > 0) {
            QList<QVariant> arguments = progressSpy.last();
            QCOMPARE(arguments.count(), 1);
            
            FileScanner::ScanProgress progress = 
                qvariant_cast<FileScanner::ScanProgress>(arguments.at(0));
            
            QVERIFY(progress.filesScanned > 0);
            QVERIFY(progress.bytesScanned > 0);
            QVERIFY(!progress.currentFolder.isEmpty());
            QVERIFY(progress.elapsedTimeMs >= 0);
            QVERIFY(progress.filesPerSecond >= 0.0);
        }
    }

    void testFilesPerSecondCalculation() {
        createMultipleFiles(200);
        QSignalSpy progressSpy(m_scanner, &FileScanner::detailedProgress);
        
        FileScanner::ScanOptions options;
        options.targetPaths = QStringList() << m_tempDir->path();
        options.minimumFileSize = 0;
        options.progressBatchSize = 100;
        
        m_scanner->startScan(options);
        QSignalSpy completedSpy(m_scanner, &FileScanner::scanCompleted);
        QVERIFY(completedSpy.wait(5000));
        
        QVERIFY(progressSpy.count() > 0);
        
        QList<QVariant> arguments = progressSpy.last();
        FileScanner::ScanProgress progress = 
            qvariant_cast<FileScanner::ScanProgress>(arguments.at(0));
        
        QVERIFY(progress.filesPerSecond > 0.0);
        
        if (progress.elapsedTimeMs > 0) {
            double expectedRate = progress.filesScanned / (progress.elapsedTimeMs / 1000.0);
            QVERIFY(qAbs(progress.filesPerSecond - expectedRate) < 0.1);
        }
    }

    void testElapsedTimeTracking() {
        createMultipleFiles(150);
        QSignalSpy progressSpy(m_scanner, &FileScanner::detailedProgress);
        
        FileScanner::ScanOptions options;
        options.targetPaths = QStringList() << m_tempDir->path();
        options.minimumFileSize = 0;
        options.progressBatchSize = 100;
        
        qint64 startTime = QDateTime::currentMSecsSinceEpoch();
        m_scanner->startScan(options);
        
        QSignalSpy completedSpy(m_scanner, &FileScanner::scanCompleted);
        QVERIFY(completedSpy.wait(5000));
        
        qint64 endTime = QDateTime::currentMSecsSinceEpoch();
        qint64 actualElapsed = endTime - startTime;
        
        QVERIFY(progressSpy.count() > 0);
        
        QList<QVariant> arguments = progressSpy.last();
        FileScanner::ScanProgress progress = 
            qvariant_cast<FileScanner::ScanProgress>(arguments.at(0));
        
        // Note: On very fast systems, elapsed time might be 0ms at file 100
        // So we check >= 0 instead of > 0
        QVERIFY(progress.elapsedTimeMs >= 0);
        QVERIFY(progress.elapsedTimeMs <= actualElapsed + 100);
    }

    void testCurrentFolderTracking() {
        createTestFile("subdir1/file1.txt");
        createTestFile("subdir1/file2.txt");
        createTestFile("subdir2/file3.txt");
        
        QSignalSpy progressSpy(m_scanner, &FileScanner::detailedProgress);
        
        FileScanner::ScanOptions options;
        options.targetPaths = QStringList() << m_tempDir->path();
        options.minimumFileSize = 0;
        options.progressBatchSize = 1;
        
        m_scanner->startScan(options);
        QSignalSpy completedSpy(m_scanner, &FileScanner::scanCompleted);
        QVERIFY(completedSpy.wait(5000));
        
        QVERIFY(progressSpy.count() > 0);
        
        bool foundValidFolder = false;
        for (int i = 0; i < progressSpy.count(); ++i) {
            QList<QVariant> arguments = progressSpy.at(i);
            FileScanner::ScanProgress progress = 
                qvariant_cast<FileScanner::ScanProgress>(arguments.at(0));
            
            if (!progress.currentFolder.isEmpty()) {
                foundValidFolder = true;
                QVERIFY(progress.currentFolder.startsWith(m_tempDir->path()));
                break;
            }
        }
        
        QVERIFY(foundValidFolder);
    }

    void testCurrentFileTracking() {
        createTestFile("file1.txt");
        createTestFile("file2.txt");
        createTestFile("file3.txt");
        
        QSignalSpy progressSpy(m_scanner, &FileScanner::detailedProgress);
        
        FileScanner::ScanOptions options;
        options.targetPaths = QStringList() << m_tempDir->path();
        options.minimumFileSize = 0;
        options.progressBatchSize = 1;
        
        m_scanner->startScan(options);
        QSignalSpy completedSpy(m_scanner, &FileScanner::scanCompleted);
        QVERIFY(completedSpy.wait(5000));
        
        QVERIFY(progressSpy.count() > 0);
        
        bool foundValidFile = false;
        for (int i = 0; i < progressSpy.count(); ++i) {
            QList<QVariant> arguments = progressSpy.at(i);
            FileScanner::ScanProgress progress = 
                qvariant_cast<FileScanner::ScanProgress>(arguments.at(0));
            
            if (!progress.currentFile.isEmpty()) {
                foundValidFile = true;
                QVERIFY(progress.currentFile.startsWith(m_tempDir->path()));
                QVERIFY(progress.currentFile.endsWith(".txt"));
                break;
            }
        }
        
        QVERIFY(foundValidFile);
    }

    void testBytesScannedTracking() {
        qint64 fileSize = 2048;
        int fileCount = 10;
        createMultipleFiles(fileCount, fileSize);
        
        QSignalSpy progressSpy(m_scanner, &FileScanner::detailedProgress);
        
        FileScanner::ScanOptions options;
        options.targetPaths = QStringList() << m_tempDir->path();
        options.minimumFileSize = 0;
        options.progressBatchSize = 5;
        
        m_scanner->startScan(options);
        QSignalSpy completedSpy(m_scanner, &FileScanner::scanCompleted);
        QVERIFY(completedSpy.wait(5000));
        
        QVERIFY(progressSpy.count() > 0);
        
        QList<QVariant> arguments = progressSpy.last();
        FileScanner::ScanProgress progress = 
            qvariant_cast<FileScanner::ScanProgress>(arguments.at(0));
        
        qint64 expectedBytes = fileSize * fileCount;
        QCOMPARE(progress.bytesScanned, expectedBytes);
    }
};

QTEST_MAIN(ScanProgressTrackingTest)
#include "test_scan_progress_tracking.moc"
