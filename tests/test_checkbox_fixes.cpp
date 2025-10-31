#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <QtWidgets/QPushButton>
#include <QtCore/QTimer>
#include <QtCore/QDebug>

#include "src/gui/results_window.h"
#include "src/core/theme_manager.h"
#include "src/core/logger.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    // Initialize logger
    Logger::setLogLevel(LogLevel::DEBUG);
    
    // Initialize theme manager
    ThemeManager::instance()->initialize();
    
    // Create main window
    QMainWindow mainWindow;
    mainWindow.setWindowTitle("Checkbox Fix Test");
    mainWindow.resize(1200, 800);
    
    // Create results window
    ResultsWindow* resultsWindow = new ResultsWindow(&mainWindow);
    mainWindow.setCentralWidget(resultsWindow);
    
    // Create test data
    ResultsWindow::ScanResults testResults;
    testResults.scanPath = "/test/path";
    testResults.scanTime = QDateTime::currentDateTime();
    testResults.totalFilesScanned = 100;
    
    // Create test group 1
    ResultsWindow::DuplicateGroup group1;
    group1.groupId = "test_group_1";
    group1.isExpanded = false;
    group1.hasSelection = false;
    
    // Add test files to group 1
    ResultsWindow::DuplicateFile file1;
    file1.filePath = "/test/image1.jpg";
    file1.fileName = "image1.jpg";
    file1.directory = "/test";
    file1.fileSize = 1024000;
    file1.lastModified = QDateTime::currentDateTime();
    file1.created = QDateTime::currentDateTime();
    file1.hash = "hash1";
    file1.isSelected = false;
    file1.fileType = "JPEG";
    
    ResultsWindow::DuplicateFile file2;
    file2.filePath = "/test/image1_copy.jpg";
    file2.fileName = "image1_copy.jpg";
    file2.directory = "/test";
    file2.fileSize = 1024000;
    file2.lastModified = QDateTime::currentDateTime();
    file2.created = QDateTime::currentDateTime();
    file2.hash = "hash1";
    file2.isSelected = false;
    file2.fileType = "JPEG";
    
    group1.files << file1 << file2;
    group1.fileCount = 2;
    group1.totalSize = file1.fileSize + file2.fileSize;
    
    // Create test group 2
    ResultsWindow::DuplicateGroup group2;
    group2.groupId = "test_group_2";
    group2.isExpanded = false;
    group2.hasSelection = false;
    
    // Add test files to group 2
    ResultsWindow::DuplicateFile file3;
    file3.filePath = "/test/document.pdf";
    file3.fileName = "document.pdf";
    file3.directory = "/test";
    file3.fileSize = 512000;
    file3.lastModified = QDateTime::currentDateTime();
    file3.created = QDateTime::currentDateTime();
    file3.hash = "hash2";
    file3.isSelected = false;
    file3.fileType = "PDF";
    
    ResultsWindow::DuplicateFile file4;
    file4.filePath = "/test/document_backup.pdf";
    file4.fileName = "document_backup.pdf";
    file4.directory = "/test";
    file4.fileSize = 512000;
    file4.lastModified = QDateTime::currentDateTime();
    file4.created = QDateTime::currentDateTime();
    file4.hash = "hash2";
    file4.isSelected = false;
    file4.fileType = "PDF";
    
    group2.files << file3 << file4;
    group2.fileCount = 2;
    group2.totalSize = file3.fileSize + file4.fileSize;
    
    testResults.duplicateGroups << group1 << group2;
    testResults.calculateTotals();
    
    // Display test results
    resultsWindow->displayResults(testResults);
    
    // Show the window
    mainWindow.show();
    
    // Add a timer to print status after 2 seconds
    QTimer::singleShot(2000, [resultsWindow]() {
        qDebug() << "=== Checkbox Fix Test Status ===";
        qDebug() << "Selected files count:" << resultsWindow->getSelectedFilesCount();
        qDebug() << "Selected files size:" << resultsWindow->getSelectedFilesSize();
        qDebug() << "Test completed - please manually test checkboxes and thumbnails";
    });
    
    return app.exec();
}