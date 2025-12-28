#include <QCoreApplication>
#include <QDebug>
#include <QCommandLineParser>
#include <QDir>
#include "test_suite_validator.h"

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    app.setApplicationName("CloneClean Test Suite Validator");
    app.setApplicationVersion("1.0");
    
    QCommandLineParser parser;
    parser.setApplicationDescription("Comprehensive test suite validation for CloneClean");
    parser.addHelpOption();
    parser.addVersionOption();
    
    // Add command line options
    QCommandLineOption categoryOption(QStringList() << "c" << "category",
                                     "Run specific test category only",
                                     "category");
    parser.addOption(categoryOption);
    
    QCommandLineOption verboseOption(QStringList() << "v" << "verbose",
                                    "Enable verbose output");
    parser.addOption(verboseOption);
    
    QCommandLineOption outputOption(QStringList() << "o" << "output",
                                   "Output directory for reports",
                                   "directory", ".");
    parser.addOption(outputOption);
    
    QCommandLineOption timeoutOption(QStringList() << "t" << "timeout",
                                    "Test timeout in seconds",
                                    "seconds", "300");
    parser.addOption(timeoutOption);
    
    parser.process(app);
    
    // Set up output directory
    QString outputDir = parser.value(outputOption);
    QDir::setCurrent(outputDir);
    
    qDebug() << "CloneClean Test Suite Validator";
    qDebug() << "==============================";
    qDebug() << "Output directory:" << QDir::currentPath();
    
    TestSuiteValidator validator;
    
    bool success = false;
    
    if (parser.isSet(categoryOption)) {
        // Run specific category only
        QString category = parser.value(categoryOption);
        qDebug() << "Running category:" << category;
        
        ValidationResults results;
        success = validator.executeTestCategory(category, results);
        validator.generateValidationReport(results);
    } else {
        // Run comprehensive validation
        qDebug() << "Running comprehensive test suite validation...";
        success = validator.validateComprehensiveTestSuite();
    }
    
    qDebug() << "\nValidation completed with result:" << (success ? "SUCCESS" : "FAILURE");
    
    return success ? 0 : 1;
}