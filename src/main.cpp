#include "main_window.h"
#include "file_scanner.h"
#include "hash_calculator.h"
#include "duplicate_detector.h"
#include "safety_manager.h"
#include "file_manager.h"
#include "theme_manager.h"
#include "logger.h"
#include <QtWidgets/QApplication>
#include <QtCore/QTranslator>
#include <QtCore/QLibraryInfo>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>
#include <QtCore/QThread>
#include <QtCore/QLocale>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    // Set application properties
    app.setApplicationName("CloneClean");
    app.setApplicationVersion("1.0.0");
    app.setApplicationDisplayName("CloneClean - One File. One Place.");
    app.setOrganizationName("CloneClean");
    app.setOrganizationDomain("cloneclean.com");

    // Set application icon
    app.setWindowIcon(QIcon(":/icons/icons/cloneclean.svg"));

    // High DPI support is automatically enabled in Qt6
    // No need to set deprecated attributes
    
    // Load system translations
    QTranslator qtTranslator;
    if (qtTranslator.load(QLocale::system(), "qt", "_", QLibraryInfo::path(QLibraryInfo::TranslationsPath))) {
        app.installTranslator(&qtTranslator);
    }
    
    // Load application translations (future enhancement)
    QTranslator appTranslator;
    QString translationsPath = QStandardPaths::locate(QStandardPaths::AppDataLocation, "translations", QStandardPaths::LocateDirectory);
    if (appTranslator.load(QLocale::system(), "cloneclean", "_", translationsPath)) {
        app.installTranslator(&appTranslator);
    }
    
    // Initialize logging system first (before any logging calls)
    Logger* logger = Logger::instance();
    logger->setLogLevel(Logger::Info);
    
    logger->info(LogCategories::SYSTEM, "Starting CloneClean application...");
    logger->debug(LogCategories::SYSTEM, "Logger initialized and configured");
    
    // Initialize theme system - theme preferences are loaded automatically in constructor
    ThemeManager::instance()->applyToApplication();
    
    // Create core components
    FileScanner fileScanner;
    HashCalculator hashCalculator;
    DuplicateDetector duplicateDetector;
    SafetyManager safetyManager;
    FileManager fileManager;
    
    // CRITICAL FIX: Move FileScanner to background thread to prevent UI freezing
    QThread* scannerThread = new QThread(&app);
    scannerThread->setObjectName("FileScannerThread");
    fileScanner.moveToThread(scannerThread);
    scannerThread->start();
    
    logger->info(LogCategories::SYSTEM, "Core components initialized:");
    logger->info(LogCategories::SYSTEM, "  - FileScanner (on background thread)");
    logger->info(LogCategories::SYSTEM, "  - HashCalculator");
    logger->info(LogCategories::SYSTEM, "  - DuplicateDetector");
    logger->info(LogCategories::SYSTEM, "  - SafetyManager");
    logger->info(LogCategories::SYSTEM, "  - FileManager");
    
    // Create and show main window
    MainWindow mainWindow;
    
    // Connect SafetyManager to FileManager
    fileManager.setSafetyManager(&safetyManager);
    
    // Connect core components to main window
    mainWindow.setFileScanner(&fileScanner);
    mainWindow.setHashCalculator(&hashCalculator);
    mainWindow.setDuplicateDetector(&duplicateDetector);
    mainWindow.setSafetyManager(&safetyManager);
    mainWindow.setFileManager(&fileManager);
    
    logger->info(LogCategories::SYSTEM, "Core components connected to MainWindow");
    
    // Connect application exit signal
    QObject::connect(&mainWindow, &MainWindow::applicationExit, &app, &QApplication::quit);
    
    // Show the main window
    mainWindow.show();
    
    logger->info(LogCategories::UI, "Main window displayed");
    logger->info(LogCategories::SYSTEM, "Application ready for user interaction");
    
    logger->info(LogCategories::SYSTEM, "CloneClean started successfully");
    logger->info(LogCategories::SYSTEM, QString("Qt Version: %1").arg(QT_VERSION_STR));
    logger->info(LogCategories::SYSTEM, QString("Application Directory: %1").arg(app.applicationDirPath()));
    
    int result = app.exec();
    
    logger->info(LogCategories::SYSTEM, QString("Application exiting with code: %1").arg(result));
    
    // CRITICAL FIX: Properly stop and cleanup the scanner thread
    if (scannerThread && scannerThread->isRunning()) {
        logger->info(LogCategories::SYSTEM, "Stopping FileScanner thread...");
        scannerThread->quit();
        if (!scannerThread->wait(3000)) {
            logger->warning(LogCategories::SYSTEM, "FileScanner thread did not stop gracefully, terminating...");
            scannerThread->terminate();
            scannerThread->wait();
        }
        logger->info(LogCategories::SYSTEM, "FileScanner thread stopped");
    }
    
    return result;
}
