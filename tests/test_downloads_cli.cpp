#include <QCoreApplication>
#include <QDir>
#include <QDebug>
#include <QElapsedTimer>
#include <QThread>
#include "file_scanner.h"
#include "duplicate_detector.h"
#include "logger.h"

/**
 * Command-line test tool for testing duplicate detection on Downloads folder
 * This runs without UI to isolate backend performance issues
 */

class DownloadsTest : public QObject {
    Q_OBJECT

public:
    DownloadsTest() : m_completed(false) {
        // Initialize logger
        Logger::instance()->setLogLevel(Logger::Info);
    }

    void run(const QString& downloadsPath) {
        qDebug() << "===========================================";
        qDebug() << "Downloads Folder Duplicate Detection Test";
        qDebug() << "===========================================";
        qDebug() << "Path:" << downloadsPath;
        qDebug() << "PID:" << QCoreApplication::applicationPid();
        qDebug() << "";

        // Print memory usage
        printMemoryUsage("Initial");

        // Phase 1: File Scanning
        qDebug() << "Phase 1: Scanning files...";
        QElapsedTimer scanTimer;
        scanTimer.start();

        FileScanner scanner;

        // Configure scan options
        FileScanner::ScanOptions options;
        options.targetPaths = QStringList() << downloadsPath;
        options.includeSubdirectories = true;
        options.includeHiddenFiles = false;
        options.minimumFileSize = 0;
        options.progressBatchSize = 10000; // Less frequent updates

        // Connect signals
        connect(&scanner, &FileScanner::scanStarted, this, [this]() {
            qDebug() << "[SCAN] Started";
        });

        connect(&scanner, &FileScanner::scanProgress, this, [this](int filesProcessed, int total, const QString& path) {
            Q_UNUSED(total);
            Q_UNUSED(path);
            if (filesProcessed % 50000 == 0) {
                qDebug() << QString("[SCAN] Progress: %1 files").arg(filesProcessed);
                printMemoryUsage("Scanning");
            }
        });

        connect(&scanner, &FileScanner::scanCompleted, this, [this, &scanner, &scanTimer]() {
            qint64 scanTime = scanTimer.elapsed();
            QVector<FileScanner::FileInfo> files = scanner.getScannedFiles();
            qDebug() << QString("[SCAN] Completed: %1 files in %2 ms (%3 files/sec)")
                .arg(files.size())
                .arg(scanTime)
                .arg(files.size() * 1000.0 / scanTime, 0, 'f', 1);

            printMemoryUsage("After Scan");

            // Phase 2: Duplicate Detection
            runDuplicateDetection(files);
        });

        connect(&scanner, &FileScanner::scanCancelled, this, [this]() {
            qDebug() << "[ERROR] Scan was cancelled";
            m_completed = true;
        });

        scanner.startScan(options);
    }

private slots:
    void runDuplicateDetection(const QVector<FileScanner::FileInfo>& files) {
        qDebug() << "";
        qDebug() << "Phase 2: Detecting duplicates...";
        QElapsedTimer detectionTimer;
        detectionTimer.start();

        DuplicateDetector detector;

        // Configure detection options
        DuplicateDetector::DetectionOptions options;
        options.level = DuplicateDetector::DetectionLevel::Standard;
        options.algorithmType = DetectionAlgorithmFactory::AlgorithmType::ExactHash;
        options.groupBySize = true;
        detector.setOptions(options);

        // Connect signals
        connect(&detector, &DuplicateDetector::detectionStarted, this, [this](int total) {
            qDebug() << QString("[DETECT] Started with %1 files").arg(total);
        });

        connect(&detector, &DuplicateDetector::detectionProgress, this, [this](const DuplicateDetector::DetectionProgress& progress) {
            if (progress.filesProcessed % 10000 == 0) {
                qDebug() << QString("[DETECT] Progress: %1/%2 files (%3%)")
                    .arg(progress.filesProcessed)
                    .arg(progress.totalFiles)
                    .arg(progress.percentComplete, 0, 'f', 1);
                printMemoryUsage("Detection");
            }
        });

        connect(&detector, &DuplicateDetector::detectionCompleted, this, [this, &detectionTimer](int groupsFound) {
            qint64 detectionTime = detectionTimer.elapsed();
            qDebug() << QString("[DETECT] Completed: %1 duplicate groups found in %2 ms")
                .arg(groupsFound)
                .arg(detectionTime);

            printMemoryUsage("After Detection");

            qDebug() << "";
            qDebug() << "===========================================";
            qDebug() << "Test Completed Successfully";
            qDebug() << "===========================================";

            m_completed = true;
            QCoreApplication::quit();
        });

        connect(&detector, &DuplicateDetector::detectionError, this, [this](const QString& error) {
            qDebug() << "[ERROR]" << error;
            m_completed = true;
            QCoreApplication::quit();
        });

        // Convert to DuplicateDetector format
        QList<DuplicateDetector::FileInfo> detectorFiles;
        for (const auto& file : files) {
            detectorFiles.append(DuplicateDetector::FileInfo::fromScannerInfo(file));
        }

        qDebug() << QString("[DETECT] Converted %1 files").arg(detectorFiles.size());
        printMemoryUsage("Before Detection");

        // Start detection
        detector.findDuplicates(detectorFiles);
    }

private:
    void printMemoryUsage(const QString& stage) {
        // Get process memory usage on macOS
        #ifdef Q_OS_MAC
        QProcess proc;
        proc.start("ps", QStringList() << "-o" << "rss=" << "-p" << QString::number(QCoreApplication::applicationPid()));
        proc.waitForFinished();
        QString output = proc.readAllStandardOutput().trimmed();
        bool ok;
        qint64 rssKB = output.toLongLong(&ok);
        if (ok) {
            double rssMB = rssKB / 1024.0;
            qDebug() << QString("[MEMORY] %1: %2 MB").arg(stage, -20).arg(rssMB, 0, 'f', 1);
        }
        #endif
    }

    bool m_completed;
};

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    QString downloadsPath;
    if (argc > 1) {
        downloadsPath = QString::fromLocal8Bit(argv[1]);
    } else {
        // Default to current user's Downloads folder
        downloadsPath = QDir::homePath() + "/Downloads";
    }

    if (!QDir(downloadsPath).exists()) {
        qDebug() << "ERROR: Path does not exist:" << downloadsPath;
        qDebug() << "Usage:" << argv[0] << "[path_to_scan]";
        return 1;
    }

    DownloadsTest test;
    test.run(downloadsPath);

    return app.exec();
}

#include "test_downloads_cli.moc"
