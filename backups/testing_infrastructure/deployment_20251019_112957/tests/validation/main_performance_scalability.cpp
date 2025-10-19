#include <QCoreApplication>
#include <QDebug>
#include <QCommandLineParser>
#include <QDir>
#include "performance_scalability_validator.h"

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    app.setApplicationName("DupFinder Performance Scalability Validator");
    app.setApplicationVersion("1.0");
    
    QCommandLineParser parser;
    parser.setApplicationDescription("Performance and scalability validation for DupFinder test suite");
    parser.addHelpOption();
    parser.addVersionOption();
    
    // Add command line options
    QCommandLineOption verboseOption(QStringList() << "v" << "verbose",
                                    "Enable verbose output");
    parser.addOption(verboseOption);
    
    QCommandLineOption outputOption(QStringList() << "o" << "output",
                                   "Output directory for reports",
                                   "directory", ".");
    parser.addOption(outputOption);
    
    QCommandLineOption parallelOption(QStringList() << "p" << "parallel",
                                     "Maximum parallel test level",
                                     "level", "8");
    parser.addOption(parallelOption);
    
    QCommandLineOption datasetSizeOption(QStringList() << "d" << "dataset-size",
                                        "Large dataset size for testing",
                                        "size", "10000");
    parser.addOption(datasetSizeOption);
    
    parser.process(app);
    
    // Set up output directory
    QString outputDir = parser.value(outputOption);
    QDir::setCurrent(outputDir);
    
    qDebug() << "DupFinder Performance Scalability Validator";
    qDebug() << "==========================================";
    qDebug() << "Output directory:" << QDir::currentPath();
    
    if (parser.isSet(parallelOption)) {
        int maxParallel = parser.value(parallelOption).toInt();
        qDebug() << "Maximum parallel level:" << maxParallel;
    }
    
    if (parser.isSet(datasetSizeOption)) {
        int datasetSize = parser.value(datasetSizeOption).toInt();
        qDebug() << "Large dataset size:" << datasetSize;
    }
    
    PerformanceScalabilityValidator validator;
    
    qDebug() << "Running performance and scalability validation...";
    bool success = validator.validatePerformanceAndScalability();
    
    qDebug() << "\nPerformance scalability validation completed with result:" << (success ? "SUCCESS" : "FAILURE");
    
    return success ? 0 : 1;
}