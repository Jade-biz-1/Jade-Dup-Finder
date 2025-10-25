#ifndef FINAL_THEME_VALIDATOR_H
#define FINAL_THEME_VALIDATOR_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDateTime>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>
#include "theme_manager.h"

class FinalThemeValidator : public QObject
{
    Q_OBJECT
    
public:
    explicit FinalThemeValidator(QObject* parent = nullptr);
    ~FinalThemeValidator();
    
    // Final validation methods
    bool performFinalValidation();
    bool validateNoHardcodedStyling();
    bool validateCompleteThemeCompliance();
    bool validateAllRequirementsMet();
    
    // Source code scanning
    QStringList scanSourceCodeForHardcodedStyles(const QString& sourceDirectory = ".");
    QStringList scanFileForHardcodedStyles(const QString& filePath);
    bool isFileExemptFromScanning(const QString& filePath) const;
    
    // Runtime validation
    bool validateRuntimeCompliance();
    QStringList scanAllWidgetsForViolations();
    bool validateThemeSystemIntegrity();
    
    // Documentation generation
    bool generateComprehensiveDocumentation(const QString& outputDirectory = "docs/theme_validation");
    bool generateValidationReport(const QString& outputPath = "theme_validation_report.json");
    bool generateComplianceMatrix(const QString& outputPath = "theme_compliance_matrix.html");
    bool generatePerformanceReport(const QString& outputPath = "theme_performance_report.html");
    
    // Test execution and reporting
    bool runAllValidationTests();
    bool generateTestReport(const QString& outputPath = "validation_test_results.html");
    
    // Final compliance certification
    struct ComplianceCertification {
        bool isFullyCompliant;
        QDateTime certificationDate;
        QString certificationVersion;
        QStringList remainingIssues;
        QStringList completedRequirements;
        double complianceScore;
        QString certificationSummary;
    };
    
    ComplianceCertification generateComplianceCertification();
    bool saveComplianceCertification(const ComplianceCertification& cert, const QString& outputPath = "theme_compliance_certification.json");
    
    // Configuration
    void setSourceDirectory(const QString& directory);
    void addExemptFile(const QString& filePath);
    void addExemptPattern(const QString& pattern);
    void setStrictMode(bool enabled);
    
signals:
    void validationProgress(int percentage, const QString& currentTask);
    void validationCompleted(bool success, const QString& summary);
    void issueFound(const QString& severity, const QString& description, const QString& location);
    
private:
    // Validation helper methods
    bool scanCppFile(const QString& filePath, QStringList& violations);
    bool scanHeaderFile(const QString& filePath, QStringList& violations);
    bool scanQrcFile(const QString& filePath, QStringList& violations);
    bool scanUiFile(const QString& filePath, QStringList& violations);
    
    // Pattern matching
    QList<QRegularExpression> getHardcodedColorPatterns() const;
    QList<QRegularExpression> getHardcodedStylePatterns() const;
    bool containsHardcodedStyling(const QString& content, QStringList& violations) const;
    
    // Documentation helpers
    QString generateHtmlHeader(const QString& title) const;
    QString generateHtmlFooter() const;
    QString formatViolationForHtml(const QString& violation) const;
    QString generateComplianceScoreHtml(double score) const;
    
    // Test execution helpers
    bool runComprehensiveThemeValidation();
    bool runPerformanceTests();
    bool runAccessibilityTests();
    bool runCrossThemeTests();
    
    // Requirements validation
    struct RequirementStatus {
        QString requirementId;
        QString description;
        bool isCompleted;
        QString evidence;
        QDateTime completionDate;
    };
    
    QList<RequirementStatus> validateAllRequirements();
    RequirementStatus validateRequirement(const QString& requirementId, const QString& description);
    void initializeRequirements();
    void markRequirementCompleted(const QString& requirementId, const QString& evidence);
    double calculateOverallComplianceScore() const;
    
    // Member variables
    QString m_sourceDirectory;
    QStringList m_exemptFiles;
    QStringList m_exemptPatterns;
    bool m_strictMode;
    
    // Validation results
    QStringList m_foundViolations;
    QStringList m_scannedFiles;
    QDateTime m_lastValidation;
    
    // Requirements tracking
    QMap<QString, RequirementStatus> m_requirementStatus;
    
    // Performance tracking
    qint64 m_validationStartTime;
    int m_totalFilesToScan;
    int m_filesScanned;
};

#endif // FINAL_THEME_VALIDATOR_H