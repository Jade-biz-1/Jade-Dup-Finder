#include "main_window.h"
#include "file_scanner.h"
#include "hash_calculator.h"
#include "duplicate_detector.h"
#include "safety_manager.h"
#include "file_manager.h"
#include "core/logger.h"
#include <QtWidgets/QApplication>
#include <QtCore/QTranslator>
#include <QtCore/QLibraryInfo>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>
#include <QtCore/QDebug>
#include <QtCore/QLocale>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    // Set application properties
    app.setApplicationName("DupFinder");
    app.setApplicationVersion("1.0.0");
    app.setApplicationDisplayName("DupFinder - Duplicate File Finder");
    app.setOrganizationName("DupFinder Team");
    app.setOrganizationDomain("dupfinder.org");
    
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
    if (appTranslator.load(QLocale::system(), "dupfinder", "_", translationsPath)) {
        app.installTranslator(&appTranslator);
    }
    
    qDebug() << "About to initialize logger...";
    
    // Initialize logging system
    Logger* logger = Logger::instance();
    
    qDebug() << "Logger instance created";
    
    logger->setLogLevel(Logger::Info);
    
    qDebug() << "Log level set";
    
    logger->info(LogCategories::SYSTEM, "Starting DupFinder application...");
    
    qDebug() << "First log message sent";
    
    // Create core components
    FileScanner fileScanner;
    HashCalculator hashCalculator;
    DuplicateDetector duplicateDetector;
    SafetyManager safetyManager;
    FileManager fileManager;
    
    logger->info(LogCategories::SYSTEM, "Core components initialized:");
    logger->info(LogCategories::SYSTEM, "  - FileScanner");
    logger->info(LogCategories::SYSTEM, "  - HashCalculator");
    logger->info(LogCategories::SYSTEM, "  - DuplicateDetector");
    logger->info(LogCategories::SYSTEM, "  - SafetyManager");
    logger->info(LogCategories::SYSTEM, "  - FileManager");
    
    qDebug() << "Core components initialized:";
    qDebug() << "  - FileScanner";
    qDebug() << "  - HashCalculator";
    qDebug() << "  - DuplicateDetector";
    qDebug() << "  - SafetyManager";
    qDebug() << "  - FileManager";
    
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
    qDebug() << "Core components connected to MainWindow";
    
    // Connect application exit signal
    QObject::connect(&mainWindow, &MainWindow::applicationExit, &app, &QApplication::quit);
    
    // Show the main window
    mainWindow.show();
    
    logger->info(LogCategories::UI, "Main window displayed");
    logger->info(LogCategories::SYSTEM, "Application ready for user interaction");
    
    qDebug() << "DupFinder started successfully";
    qDebug() << "Qt Version:" << QT_VERSION_STR;
    qDebug() << "Application Directory:" << app.applicationDirPath();
    
    int result = app.exec();
    
    logger->info(LogCategories::SYSTEM, QString("Application exiting with code: %1").arg(result));
    return result;
}
