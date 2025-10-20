#ifndef STYLE_VALIDATOR_H
#define STYLE_VALIDATOR_H

#include <QObject>
#include <QWidget>
#include <QList>
#include <QString>
#include <QRegularExpression>
#include <QColor>
#include <QTimer>
#include <QDateTime>
#include <QMutex>
#include "theme_manager.h"

class StyleValidator : public QObject
{
    Q_OBJECT
    
public:
    explicit StyleValidator(QObject* parent = nullptr);
    
    static ValidationResult validateComponent(QWidget* component);
    static QList<StyleViolation> scanForHardcodedStyles(QWidget* component);
    static bool validateAccessibility(const ThemeData& theme);
    static ComplianceReport generateReport(const QList<QWidget*>& components);
    
    // Enhanced runtime scanning methods
    void enableRuntimeScanning(bool enabled = true);
    void setRuntimeScanInterval(int milliseconds = 5000);
    ComplianceReport performComprehensiveApplicationScan();
    QList<StyleViolation> scanAllApplicationComponents();
    
    // File-based scanning for source code analysis
    static QList<StyleViolation> scanSourceFiles(const QString& sourceDirectory);
    static QList<StyleViolation> scanSourceFile(const QString& filePath);
    
    // Enhanced reporting
    void generateDetailedComplianceReport(const QString& outputPath = QString());
    QStringList getViolationSummary() const;
    void logViolationDetails(const QList<StyleViolation>& violations);
    
    // Detailed validation methods
    static QList<StyleViolation> detectHardcodedColors(const QString& styleSheet, const QString& componentName);
    static QList<StyleViolation> detectInlineStyles(QWidget* component);
    static QList<StyleViolation> validateAccessibilityCompliance(const ThemeData& theme);
    
    // Enhanced detection methods
    static QList<StyleViolation> detectHardcodedFonts(QWidget* component);
    static QList<StyleViolation> detectHardcodedSizes(QWidget* component);
    static QList<StyleViolation> detectDeprecatedStyles(QWidget* component);
    static QList<StyleViolation> validateThemeConsistency(QWidget* component, const ThemeData& expectedTheme);
    
    // Enhanced accessibility helpers
    static double calculateContrastRatio(const QColor& fg, const QColor& bg);
    static bool meetsWCAGStandards(double contrastRatio, const QString& level = "AA");
    static bool meetsWCAGAAA(double contrastRatio);
    static QString getAccessibilityRecommendation(double contrastRatio);
    static QColor suggestAccessibleColor(const QColor& baseColor, const QColor& background, double targetRatio = 4.5);
    static bool isHighContrastModeEnabled();
    static QList<StyleViolation> validateFocusIndicators(QWidget* component);
    static QList<StyleViolation> validateColorOnlyInformation(QWidget* component);
    
    // Style analysis
    static bool hasHardcodedColors(const QString& styleSheet);
    static QStringList extractColors(const QString& styleSheet);
    static QString suggestThemeAlternative(const QString& hardcodedValue);
    
    // Statistics and tracking
    int getTotalScansPerformed() const { return m_totalScansPerformed; }
    QDateTime getLastScanTime() const { return m_lastScanTime; }
    QList<StyleViolation> getRecentViolations() const { return m_recentViolations; }
    
public slots:
    void performRuntimeScan();
    void clearViolationHistory();
    
signals:
    void validationCompleted(const ValidationResult& result);
    void complianceReportGenerated(const ComplianceReport& report);
    void runtimeViolationDetected(const StyleViolation& violation);
    void scanCompleted(int violationsFound);
    
private:
    static QRegularExpression getHexColorPattern();
    static QRegularExpression getRgbColorPattern();
    static QRegularExpression getRgbaColorPattern();
    static QRegularExpression getHardcodedFontPattern();
    static QRegularExpression getHardcodedSizePattern();
    static QString getComponentIdentifier(QWidget* component);
    static QString getSeverityLevel(const QString& violationType);
    static double getLuminance(const QColor& color);
    
    // Source file analysis helpers
    static QList<StyleViolation> analyzeSourceFileContent(const QString& content, const QString& fileName);
    static QStringList findStyleSheetCalls(const QString& content);
    static int findLineNumber(const QString& content, const QString& searchText);
    
    // Runtime scanning members
    QTimer* m_scanTimer;
    QDateTime m_lastScanTime;
    int m_totalScansPerformed;
    QList<StyleViolation> m_recentViolations;
    QMutex m_violationMutex;
};

#endif // STYLE_VALIDATOR_H