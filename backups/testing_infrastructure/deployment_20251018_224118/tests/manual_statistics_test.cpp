/**
 * Manual test to demonstrate FileScanner statistics functionality
 * 
 * This is a simple program to manually verify the statistics feature works correctly.
 * Compile and run with: 
 *   g++ -o manual_stats_test manual_statistics_test.cpp -I../include -L../build -ldupfinder $(pkg-config --cflags --libs Qt6Core)
 */

#include <QCoreApplication>
#include <QDebug>
#include "file_scanner.h"

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    
    qDebug() << "=== FileScanner Statistics Test ===\n";
    
    FileScanner scanner;
    
    // Connect to statistics signal
    QObject::connect(&scanner, &FileScanner::scanStatistics, [](const FileScanner::ScanStatistics& stats) {
        qDebug() << "\n=== Scan Statistics ===";
        qDebug() << "Total Files Scanned:" << stats.totalFilesScanned;
        qDebug() << "Total Directories Scanned:" << stats.totalDirectoriesScanned;
        qDebug() << "Total Bytes Scanned:" << stats.totalBytesScanned;
        qDebug() << "Files Filtered by Size:" << stats.filesFilteredBySize;
        qDebug() << "Files Filtered by Pattern:" << stats.filesFilteredByPattern;
        qDebug() << "Files Filtered by Hidden:" << stats.filesFilteredByHidden;
        qDebug() << "Directories Skipped:" << stats.directoriesSkipped;
        qDebug() << "Errors Encountered:" << stats.errorsEncountered;
        qDebug() << "Scan Duration:" << stats.scanDurationMs << "ms";
        qDebug() << "Files Per Second:" << stats.filesPerSecond;
        qDebug() << "======================\n";
    });
    
    // Configure scan options
    FileScanner::ScanOptions options;
    options.targetPaths << ".";  // Scan current directory
    options.minimumFileSize = 1024;  // 1KB minimum
    options.includePatterns << "*.cpp" << "*.h";  // Only C++ files
    options.includeHiddenFiles = false;
    
    qDebug() << "Starting scan of current directory...";
    qDebug() << "Filters: *.cpp, *.h files, minimum 1KB\n";
    
    scanner.startScan(options);
    
    // Wait for completion
    while (scanner.isScanning()) {
        QCoreApplication::processEvents();
    }
    
    qDebug() << "Scan completed!";
    
    return 0;
}
