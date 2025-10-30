// Theme Compliance Checker - Static Analysis Tool
// Run this separately to validate theme compliance, not during app runtime
//
// Usage: theme_compliance_checker [source_directory] [options]
//
// This tool should be run as part of:
// 1. Pre-commit hooks
// 2. CI/CD pipeline
// 3. Manual code quality checks
//
// NOT as part of the running application!

#include <QCoreApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QDebug>
#include <iostream>

struct StyleViolation {
    QString filePath;
    int lineNumber;
    QString violationType;
    QString currentValue;
    QString suggestedFix;
    QString severity;
};

class ThemeComplianceChecker {
public:
    ThemeComplianceChecker(const QString& sourceDir) 
        : m_sourceDir(sourceDir)
        , m_totalFiles(0)
        , m_filesScanned(0)
        , m_violationsFound(0)
    {}
    
    int run() {
        std::cout << "Theme Compliance Checker\n";
        std::cout << "========================\n\n";
        std::cout << "Scanning directory: " << m_sourceDir.toStdString() << "\n\n";
        
        QDir dir(m_sourceDir);
        if (!dir.exists()) {
            std::cerr << "Error: Directory does not exist: " << m_sourceDir.toStdString() << "\n";
            return 1;
        }
        
        // Scan C++ files
        scanDirectory(dir);
        
        // Generate report
        generateReport();
        
        return m_violationsFound > 0 ? 1 : 0;
    }
    
private:
    void scanDirectory(const QDir& dir) {
        QStringList filters;
        filters << "*.cpp" << "*.h";
        
        QFileInfoList files = dir.entryInfoList(filters, QDir::Files);
        m_totalFiles += files.size();
        
        for (const QFileInfo& fileInfo : files) {
            scanFile(fileInfo.absoluteFilePath());
        }
        
        // Recursively scan subdirectories
        QFileInfoList subdirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
        for (const QFileInfo& subdirInfo : subdirs) {
            scanDirectory(QDir(subdirInfo.absoluteFilePath()));
        }
    }
    
    void scanFile(const QString& filePath) {
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            return;
        }
        
        m_filesScanned++;
        
        QTextStream in(&file);
        int lineNumber = 0;
        
        while (!in.atEnd()) {
            QString line = in.readLine();
            lineNumber++;
            
            checkForViolations(filePath, lineNumber, line);
        }
    }
    
    void checkForViolations(const QString& filePath, int lineNumber, const QString& line) {
        // Check for hardcoded colors
        QRegularExpression colorRegex(R"(#[0-9a-fA-F]{3,6}\b|rgb\(|rgba\()");
        if (colorRegex.match(line).hasMatch() && line.contains("setStyleSheet")) {
            addViolation(filePath, lineNumber, "hardcoded-color", 
                        line.trimmed(), 
                        "Use ThemeManager::getCurrentThemeData().colors instead",
                        "critical");
        }
        
        // Check for hardcoded font families
        QRegularExpression fontRegex(R"(font-family:\s*[\"'])");
        if (fontRegex.match(line).hasMatch()) {
            addViolation(filePath, lineNumber, "hardcoded-font",
                        line.trimmed(),
                        "Use theme.typography.fontFamily from ThemeManager",
                        "warning");
        }
        
        // Check for inline styles
        if (line.contains("setStyleSheet") && line.contains("\"")) {
            addViolation(filePath, lineNumber, "inline-style",
                        line.trimmed(),
                        "Use ThemeManager styling methods instead of inline styles",
                        "warning");
        }
        
        // Check for hardcoded sizes
        QRegularExpression sizeRegex(R"(\d+px)");
        if (sizeRegex.match(line).hasMatch() && line.contains("setStyleSheet")) {
            addViolation(filePath, lineNumber, "hardcoded-size",
                        line.trimmed(),
                        "Use theme.spacing values or relative units from ThemeManager",
                        "info");
        }
    }
    
    void addViolation(const QString& filePath, int lineNumber, 
                     const QString& type, const QString& value,
                     const QString& fix, const QString& severity) {
        StyleViolation violation;
        violation.filePath = filePath;
        violation.lineNumber = lineNumber;
        violation.violationType = type;
        violation.currentValue = value;
        violation.suggestedFix = fix;
        violation.severity = severity;
        
        m_violations.append(violation);
        m_violationsFound++;
    }
    
    void generateReport() {
        std::cout << "Scan Complete\n";
        std::cout << "=============\n";
        std::cout << "Files scanned: " << m_filesScanned << "/" << m_totalFiles << "\n";
        std::cout << "Violations found: " << m_violationsFound << "\n\n";
        
        if (m_violations.isEmpty()) {
            std::cout << "✓ No theme compliance violations found!\n";
            return;
        }
        
        // Count by severity
        int critical = 0, warning = 0, info = 0;
        for (const auto& v : m_violations) {
            if (v.severity == "critical") critical++;
            else if (v.severity == "warning") warning++;
            else info++;
        }
        
        std::cout << "Severity breakdown:\n";
        std::cout << "  Critical: " << critical << "\n";
        std::cout << "  Warning:  " << warning << "\n";
        std::cout << "  Info:     " << info << "\n\n";
        
        // Show first 20 violations
        std::cout << "Sample violations (first 20):\n";
        std::cout << "==============================\n";
        
        int count = 0;
        for (const auto& v : m_violations) {
            if (count++ >= 20) break;
            
            std::cout << "\n[" << v.severity.toUpper().toStdString() << "] "
                     << v.filePath.toStdString() << ":" << v.lineNumber << "\n";
            std::cout << "  Type: " << v.violationType.toStdString() << "\n";
            std::cout << "  Line: " << v.currentValue.toStdString() << "\n";
            std::cout << "  Fix:  " << v.suggestedFix.toStdString() << "\n";
        }
        
        if (m_violations.size() > 20) {
            std::cout << "\n... and " << (m_violations.size() - 20) << " more violations.\n";
        }
        
        std::cout << "\n\nRECOMMENDATIONS:\n";
        std::cout << "- Fix " << critical << " critical violations immediately\n";
        std::cout << "- Address " << warning << " warning violations when refactoring\n";
        std::cout << "- Consider " << info << " info violations for future improvements\n";
    }
    
    QString m_sourceDir;
    int m_totalFiles;
    int m_filesScanned;
    int m_violationsFound;
    QList<StyleViolation> m_violations;
};

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("Theme Compliance Checker");
    QCoreApplication::setApplicationVersion("1.0");
    
    QCommandLineParser parser;
    parser.setApplicationDescription("Static analysis tool for theme compliance checking");
    parser.addHelpOption();
    parser.addVersionOption();
    
    parser.addPositionalArgument("source", "Source directory to scan");
    
    QCommandLineOption outputOption(QStringList() << "o" << "output",
                                   "Output report to file",
                                   "file");
    parser.addOption(outputOption);
    
    QCommandLineOption verboseOption(QStringList() << "v" << "verbose",
                                    "Verbose output");
    parser.addOption(verboseOption);
    
    parser.process(app);
    
    const QStringList args = parser.positionalArguments();
    if (args.isEmpty()) {
        std::cerr << "Error: No source directory specified\n\n";
        parser.showHelp(1);
    }
    
    QString sourceDir = args.first();
    
    ThemeComplianceChecker checker(sourceDir);
    return checker.run();
}
