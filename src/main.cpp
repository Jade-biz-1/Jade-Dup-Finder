#include "main_window.h"
#include "file_scanner.h"
// #include "hash_calculator.h"
// #include "duplicate_detector.h"
// #include "safety_manager.h"
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
    
    // Create core components
    FileScanner fileScanner;
    // HashCalculator hashCalculator;
    // DuplicateDetector duplicateDetector;
    // SafetyManager safetyManager;
    
    qDebug() << "Core components initialized:";
    qDebug() << "  - FileScanner";
    // qDebug() << "  - HashCalculator";
    // qDebug() << "  - DuplicateDetector";
    // qDebug() << "  - SafetyManager";
    
    // Create and show main window
    MainWindow mainWindow;
    
    // Connect core components to main window
    mainWindow.setFileScanner(&fileScanner);
    // mainWindow.setHashCalculator(&hashCalculator);
    // mainWindow.setDuplicateDetector(&duplicateDetector);
    // mainWindow.setSafetyManager(&safetyManager);
    
    qDebug() << "Core components connected to MainWindow";
    
    // Connect application exit signal
    QObject::connect(&mainWindow, &MainWindow::applicationExit, &app, &QApplication::quit);
    
    // Show the main window
    mainWindow.show();
    
    qDebug() << "DupFinder started successfully";
    qDebug() << "Qt Version:" << QT_VERSION_STR;
    qDebug() << "Application Directory:" << app.applicationDirPath();
    
    return app.exec();
}
