#include "style_validator.h"
#include "core/logger.h"
#include <QApplication>
#include <QRegularExpression>
#include <QRegularExpressionMatch>
#include <QRegularExpressionMatchIterator>
#include <QDir>
#include <QFileInfo>
#include <QTextStream>
#include <QDebug>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QCheckBox>
#include <QRadioButton>
#include <cmath>
#include <windows.h>

StyleValidator::StyleValidator(QObject* parent)
    : QObject(parent)
{
    // Initialize runtime scanning timer
    m_scanTimer = new QTimer(this);
    m_scanTimer->setSingleShot(false);
    m_scanTimer->setInterval(5000); // Scan every 5 seconds
    
    connect(m_scanTimer, &QTimer::timeout, this, &StyleValidator::performRuntimeScan);
    
    // Initialize violation tracking
    m_lastScanTime = QDateTime::currentDateTime();
    m_totalScansPerformed = 0;
}

ValidationResult StyleValidator::validateComponent(QWidget* component)
{
    ValidationResult result;
    result.isCompliant = true;
    result.accessibilityScore = 100.0;
    result.summary = "Component validation completed";
    
    if (!component) {
        result.isCompliant = false;
        result.accessibilityScore = 0.0;
        result.summary = "Invalid component (null pointer)";
        return result;
    }
    
    // Scan for hardcoded styles
    result.violations = scanForHardcodedStyles(component);
    
    // Check if component is compliant
    result.isCompliant = result.violations.isEmpty();
    
    // Calculate accessibility score based on violations
    int criticalViolations = 0;
    int warningViolations = 0;
    
    for (const StyleViolation& violation : result.violations) {
        if (violation.severity == "critical") {
            criticalViolations++;
        } else if (violation.severity == "warning") {
            warningViolations++;
        }
    }
    
    // Calculate score (100 - penalties)
    result.accessibilityScore = 100.0 - (criticalViolations * 20.0) - (warningViolations * 5.0);
    result.accessibilityScore = qMax(0.0, result.accessibilityScore);
    
    // Generate summary
    if (result.isCompliant) {
        result.summary = "Component is fully theme-compliant";
    } else {
        result.summary = QString("Component has %1 violations (%2 critical, %3 warnings)")
                        .arg(result.violations.size())
                        .arg(criticalViolations)
                        .arg(warningViolations);
    }
    
    return result;
}

QList<StyleViolation> StyleValidator::scanForHardcodedStyles(QWidget* component)
{
    QList<StyleViolation> violations;
    
    if (!component) {
        return violations;
    }
    
    QString componentName = getComponentIdentifier(component);
    QString styleSheet = component->styleSheet();
    
    // Detect hardcoded colors
    violations.append(detectHardcodedColors(styleSheet, componentName));
    
    // Detect inline styles (additional checks)
    violations.append(detectInlineStyles(component));
    
    // Detect hardcoded fonts
    violations.append(detectHardcodedFonts(component));
    
    // Detect hardcoded sizes
    violations.append(detectHardcodedSizes(component));
    
    // Detect deprecated styles
    violations.append(detectDeprecatedStyles(component));
    
    // Validate accessibility features
    violations.append(validateFocusIndicators(component));
    violations.append(validateColorOnlyInformation(component));
    
    return violations;
}

bool StyleValidator::validateAccessibility(const ThemeData& theme)
{
    QList<StyleViolation> violations = validateAccessibilityCompliance(theme);
    return violations.isEmpty();
}

ComplianceReport StyleValidator::generateReport(const QList<QWidget*>& components)
{
    ComplianceReport report;
    report.generated = QDateTime::currentDateTime();
    report.totalComponents = components.size();
    report.compliantComponents = 0;
    report.violationCount = 0;
    report.overallScore = 0.0;
    
    double totalScore = 0.0;
    
    for (QWidget* component : components) {
        if (!component) continue;
        
        ValidationResult result = validateComponent(component);
        
        if (result.isCompliant) {
            report.compliantComponents++;
        }
        
        report.violationCount += result.violations.size();
        totalScore += result.accessibilityScore;
        
        // Collect critical violations and warnings
        for (const StyleViolation& violation : result.violations) {
            if (violation.severity == "critical") {
                report.criticalViolations.append(violation);
            } else if (violation.severity == "warning") {
                report.warnings.append(violation);
            }
        }
    }
    
    // Calculate overall score
    if (report.totalComponents > 0) {
        report.overallScore = totalScore / report.totalComponents;
    }
    
    // Generate recommendations
    QStringList recommendations;
    
    if (report.criticalViolations.size() > 0) {
        recommendations.append(QString("Fix %1 critical violations immediately").arg(report.criticalViolations.size()));
    }
    
    if (report.warnings.size() > 0) {
        recommendations.append(QString("Address %1 warning violations").arg(report.warnings.size()));
    }
    
    if (report.overallScore < 80.0) {
        recommendations.append("Overall theme compliance is below acceptable threshold (80%)");
    }
    
    if (recommendations.isEmpty()) {
        recommendations.append("All components are theme-compliant");
    }
    
    report.recommendations = recommendations.join("; ");
    
    return report;
}

QList<StyleViolation> StyleValidator::detectHardcodedColors(const QString& styleSheet, const QString& componentName)
{
    QList<StyleViolation> violations;
    
    if (styleSheet.isEmpty()) {
        return violations;
    }
    
    // Check for hex colors
    QRegularExpression hexPattern = getHexColorPattern();
    QRegularExpressionMatchIterator hexMatches = hexPattern.globalMatch(styleSheet);
    
    while (hexMatches.hasNext()) {
        QRegularExpressionMatch match = hexMatches.next();
        StyleViolation violation;
        violation.componentName = componentName;
        violation.violationType = "hardcoded-color";
        violation.currentValue = match.captured(0);
        violation.suggestedFix = suggestThemeAlternative(violation.currentValue);
        violation.severity = getSeverityLevel(violation.violationType);
        violation.fileName = "runtime";
        violation.lineNumber = -1;
        
        violations.append(violation);
    }
    
    // Check for RGB colors
    QRegularExpression rgbPattern = getRgbColorPattern();
    QRegularExpressionMatchIterator rgbMatches = rgbPattern.globalMatch(styleSheet);
    
    while (rgbMatches.hasNext()) {
        QRegularExpressionMatch match = rgbMatches.next();
        StyleViolation violation;
        violation.componentName = componentName;
        violation.violationType = "hardcoded-color";
        violation.currentValue = match.captured(0);
        violation.suggestedFix = suggestThemeAlternative(violation.currentValue);
        violation.severity = getSeverityLevel(violation.violationType);
        violation.fileName = "runtime";
        violation.lineNumber = -1;
        
        violations.append(violation);
    }
    
    // Check for RGBA colors
    QRegularExpression rgbaPattern = getRgbaColorPattern();
    QRegularExpressionMatchIterator rgbaMatches = rgbaPattern.globalMatch(styleSheet);
    
    while (rgbaMatches.hasNext()) {
        QRegularExpressionMatch match = rgbaMatches.next();
        StyleViolation violation;
        violation.componentName = componentName;
        violation.violationType = "hardcoded-color";
        violation.currentValue = match.captured(0);
        violation.suggestedFix = suggestThemeAlternative(violation.currentValue);
        violation.severity = getSeverityLevel(violation.violationType);
        violation.fileName = "runtime";
        violation.lineNumber = -1;
        
        violations.append(violation);
    }
    
    return violations;
}

QList<StyleViolation> StyleValidator::detectInlineStyles(QWidget* component)
{
    QList<StyleViolation> violations;
    
    if (!component) {
        return violations;
    }
    
    QString componentName = getComponentIdentifier(component);
    QString styleSheet = component->styleSheet();
    
    // Check for common inline style patterns that should use theme system
    QStringList problematicPatterns = {
        "background-color\\s*:\\s*[^;]+",
        "color\\s*:\\s*[^;]+",
        "border\\s*:\\s*[^;]+",
        "font-size\\s*:\\s*[^;]+",
        "font-family\\s*:\\s*[^;]+"
    };
    
    for (const QString& pattern : problematicPatterns) {
        QRegularExpression regex(pattern, QRegularExpression::CaseInsensitiveOption);
        QRegularExpressionMatchIterator matches = regex.globalMatch(styleSheet);
        
        while (matches.hasNext()) {
            QRegularExpressionMatch match = matches.next();
            
            // Skip if it's already using theme variables or palette colors
            QString matchedText = match.captured(0);
            if (matchedText.contains("palette(") || matchedText.contains("theme.")) {
                continue;
            }
            
            StyleViolation violation;
            violation.componentName = componentName;
            violation.violationType = "inline-style";
            violation.currentValue = matchedText;
            violation.suggestedFix = "Use ThemeManager styling methods instead of inline styles";
            violation.severity = "warning";
            violation.fileName = "runtime";
            violation.lineNumber = -1;
            
            violations.append(violation);
        }
    }
    
    return violations;
}

QList<StyleViolation> StyleValidator::validateAccessibilityCompliance(const ThemeData& theme)
{
    QList<StyleViolation> violations;
    
    // Enhanced WCAG 2.1 AA compliance checking
    struct ColorPair {
        QColor foreground;
        QColor background;
        QString description;
        double requiredRatioAA;
        double requiredRatioAAA;
        bool isLargeText;
    };
    
    QList<ColorPair> criticalPairs = {
        // Normal text (WCAG AA: 4.5:1, AAA: 7:1)
        {theme.colors.foreground, theme.colors.background, "Normal text on background", 4.5, 7.0, false},
        {theme.colors.accent, theme.colors.background, "Accent text on background", 4.5, 7.0, false},
        {theme.colors.error, theme.colors.background, "Error text on background", 4.5, 7.0, false},
        {theme.colors.warning, theme.colors.background, "Warning text on background", 4.5, 7.0, false},
        {theme.colors.success, theme.colors.background, "Success text on background", 4.5, 7.0, false},
        {theme.colors.info, theme.colors.background, "Info text on background", 4.5, 7.0, false},
        
        // Large text (WCAG AA: 3:1, AAA: 4.5:1)
        {theme.colors.foreground, theme.colors.background, "Large text on background", 3.0, 4.5, true},
        
        // Interactive elements (WCAG AA: 3:1 for non-text)
        {theme.colors.accent, theme.colors.background, "Interactive elements", 3.0, 4.5, false},
        {theme.colors.border, theme.colors.background, "Borders and focus indicators", 3.0, 4.5, false},
        {theme.colors.hover, theme.colors.background, "Hover state indicators", 3.0, 4.5, false},
        
        // High contrast mode requirements (enhanced ratios)
        {theme.colors.foreground, theme.colors.background, "High contrast text", 7.0, 10.0, false},
        {theme.colors.accent, theme.colors.background, "High contrast interactive", 4.5, 7.0, false},
        
        // Disabled state accessibility
        {theme.colors.disabled, theme.colors.background, "Disabled text visibility", 3.0, 4.5, false}
    };
    
    for (const ColorPair& pair : criticalPairs) {
        double ratio = calculateContrastRatio(pair.foreground, pair.background);
        
        // Check WCAG AA compliance
        if (ratio < pair.requiredRatioAA) {
            StyleViolation violation;
            violation.componentName = "Theme";
            violation.violationType = "accessibility-wcag-aa";
            violation.currentValue = QString("%1 contrast ratio: %2:1 (required: %3:1)")
                                   .arg(pair.description)
                                   .arg(ratio, 0, 'f', 2)
                                   .arg(pair.requiredRatioAA);
            violation.suggestedFix = QString("Increase contrast ratio to at least %1:1 for WCAG AA compliance. %2")
                                   .arg(pair.requiredRatioAA)
                                   .arg(getAccessibilityRecommendation(ratio));
            violation.severity = ratio < (pair.requiredRatioAA * 0.7) ? "critical" : "warning";
            violation.fileName = "theme";
            violation.lineNumber = -1;
            
            violations.append(violation);
        }
        
        // Check WCAG AAA compliance (informational)
        if (ratio < pair.requiredRatioAAA && ratio >= pair.requiredRatioAA) {
            StyleViolation violation;
            violation.componentName = "Theme";
            violation.violationType = "accessibility-wcag-aaa";
            violation.currentValue = QString("%1 contrast ratio: %2:1 (AAA requires: %3:1)")
                                   .arg(pair.description)
                                   .arg(ratio, 0, 'f', 2)
                                   .arg(pair.requiredRatioAAA);
            violation.suggestedFix = QString("Consider increasing contrast ratio to %1:1 for WCAG AAA compliance")
                                   .arg(pair.requiredRatioAAA);
            violation.severity = "info";
            violation.fileName = "theme";
            violation.lineNumber = -1;
            
            violations.append(violation);
        }
    }
    
    // Additional accessibility checks
    
    // Check for sufficient color differentiation between interactive states
    double hoverDifference = calculateContrastRatio(theme.colors.hover, theme.colors.background);
    double accentDifference = calculateContrastRatio(theme.colors.accent, theme.colors.background);
    
    if (std::abs(hoverDifference - accentDifference) < 1.2) {
        StyleViolation violation;
        violation.componentName = "Theme";
        violation.violationType = "accessibility-state-differentiation";
        violation.currentValue = QString("Insufficient contrast between normal and hover states");
        violation.suggestedFix = "Ensure hover states have at least 1.2:1 contrast difference from normal states";
        violation.severity = "warning";
        violation.fileName = "theme";
        violation.lineNumber = -1;
        
        violations.append(violation);
    }
    
    // Check for color-only information conveyance
    QList<QColor> statusColors = {theme.colors.success, theme.colors.warning, theme.colors.error, theme.colors.info};
    for (int i = 0; i < statusColors.size(); ++i) {
        for (int j = i + 1; j < statusColors.size(); ++j) {
            double colorDifference = calculateContrastRatio(statusColors[i], statusColors[j]);
            if (colorDifference < 3.0) {
                StyleViolation violation;
                violation.componentName = "Theme";
                violation.violationType = "accessibility-color-differentiation";
                violation.currentValue = QString("Insufficient contrast between status colors: %1:1")
                                       .arg(colorDifference, 0, 'f', 2);
                violation.suggestedFix = "Ensure status colors have at least 3:1 contrast ratio between each other, or provide alternative indicators (icons, text)";
                violation.severity = "warning";
                violation.fileName = "theme";
                violation.lineNumber = -1;
                
                violations.append(violation);
            }
        }
    }
    
    // Check font size accessibility
    if (theme.typography.baseFontSize < 9) {
        StyleViolation violation;
        violation.componentName = "Theme";
        violation.violationType = "accessibility-font-size";
        violation.currentValue = QString("Base font size too small: %1px").arg(theme.typography.baseFontSize);
        violation.suggestedFix = "Use minimum 9px font size for accessibility compliance";
        violation.severity = "warning";
        violation.fileName = "theme";
        violation.lineNumber = -1;
        
        violations.append(violation);
    }
    
    return violations;
}

double StyleValidator::calculateContrastRatio(const QColor& fg, const QColor& bg)
{
    double l1 = getLuminance(fg);
    double l2 = getLuminance(bg);
    
    if (l1 < l2) {
        std::swap(l1, l2);
    }
    
    return (l1 + 0.05) / (l2 + 0.05);
}

bool StyleValidator::meetsWCAGStandards(double contrastRatio, const QString& level)
{
    if (level == "AAA") {
        return contrastRatio >= 7.0;
    } else { // AA (default)
        return contrastRatio >= 4.5;
    }
}

bool StyleValidator::meetsWCAGAAA(double contrastRatio)
{
    return contrastRatio >= 7.0;
}

QString StyleValidator::getAccessibilityRecommendation(double contrastRatio)
{
    if (contrastRatio < 3.0) {
        return "Consider using completely different colors";
    } else if (contrastRatio < 4.5) {
        return "Adjust brightness or saturation";
    } else if (contrastRatio < 7.0) {
        return "Good for AA compliance, consider improving for AAA";
    } else {
        return "Excellent contrast ratio";
    }
}

bool StyleValidator::hasHardcodedColors(const QString& styleSheet)
{
    return getHexColorPattern().match(styleSheet).hasMatch() ||
           getRgbColorPattern().match(styleSheet).hasMatch() ||
           getRgbaColorPattern().match(styleSheet).hasMatch();
}

QStringList StyleValidator::extractColors(const QString& styleSheet)
{
    QStringList colors;
    
    // Extract hex colors
    QRegularExpressionMatchIterator hexMatches = getHexColorPattern().globalMatch(styleSheet);
    while (hexMatches.hasNext()) {
        colors.append(hexMatches.next().captured(0));
    }
    
    // Extract RGB colors
    QRegularExpressionMatchIterator rgbMatches = getRgbColorPattern().globalMatch(styleSheet);
    while (rgbMatches.hasNext()) {
        colors.append(rgbMatches.next().captured(0));
    }
    
    // Extract RGBA colors
    QRegularExpressionMatchIterator rgbaMatches = getRgbaColorPattern().globalMatch(styleSheet);
    while (rgbaMatches.hasNext()) {
        colors.append(rgbaMatches.next().captured(0));
    }
    
    return colors;
}

QString StyleValidator::suggestThemeAlternative(const QString& hardcodedValue)
{
    // Simple mapping of common hardcoded values to theme alternatives
    static QMap<QString, QString> alternatives = {
        {"#ffffff", "theme.colors.background (light theme)"},
        {"#000000", "theme.colors.foreground (dark theme)"},
        {"#f8f9fa", "theme.colors.background"},
        {"#212529", "theme.colors.foreground"},
        {"#007bff", "theme.colors.accent"},
        {"#28a745", "theme.colors.success"},
        {"#dc3545", "theme.colors.error"},
        {"#ffc107", "theme.colors.warning"},
        {"#17a2b8", "theme.colors.info"}
    };
    
    QString lowerValue = hardcodedValue.toLower();
    if (alternatives.contains(lowerValue)) {
        return alternatives[lowerValue];
    }
    
    // Generic suggestion
    return "Use appropriate theme color from ThemeManager";
}

QRegularExpression StyleValidator::getHexColorPattern()
{
    return QRegularExpression("#[0-9a-fA-F]{3,6}\\b");
}

QRegularExpression StyleValidator::getRgbColorPattern()
{
    return QRegularExpression("rgb\\s*\\([^)]+\\)");
}

QRegularExpression StyleValidator::getRgbaColorPattern()
{
    return QRegularExpression("rgba\\s*\\([^)]+\\)");
}

QString StyleValidator::getComponentIdentifier(QWidget* component)
{
    if (!component) {
        return "Unknown";
    }
    
    QString className = component->metaObject()->className();
    QString objectName = component->objectName();
    
    if (objectName.isEmpty()) {
        return className;
    } else {
        return QString("%1 (%2)").arg(className).arg(objectName);
    }
}

QString StyleValidator::getSeverityLevel(const QString& violationType)
{
    if (violationType == "hardcoded-color" || violationType == "accessibility") {
        return "critical";
    } else if (violationType == "inline-style") {
        return "warning";
    } else {
        return "info";
    }
}

double StyleValidator::getLuminance(const QColor& color)
{
    auto sRGBtoLin = [](double colorChannel) -> double {
        if (colorChannel <= 0.03928) {
            return colorChannel / 12.92;
        } else {
            return std::pow((colorChannel + 0.055) / 1.055, 2.4);
        }
    };
    
    double r = sRGBtoLin(color.redF());
    double g = sRGBtoLin(color.greenF());
    double b = sRGBtoLin(color.blueF());
    
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// Enhanced runtime scanning methods
void StyleValidator::enableRuntimeScanning(bool enabled)
{
    if (enabled) {
        m_scanTimer->start();
        LOG_INFO(LogCategories::UI, "Runtime style validation scanning enabled");
    } else {
        m_scanTimer->stop();
        LOG_INFO(LogCategories::UI, "Runtime style validation scanning disabled");
    }
}

void StyleValidator::setRuntimeScanInterval(int milliseconds)
{
    m_scanTimer->setInterval(milliseconds);
    LOG_INFO(LogCategories::UI, QString("Runtime scan interval set to %1ms").arg(milliseconds));
}

void StyleValidator::performRuntimeScan()
{
    QMutexLocker locker(&m_violationMutex);
    
    m_totalScansPerformed++;
    m_lastScanTime = QDateTime::currentDateTime();
    
    LOG_DEBUG(LogCategories::UI, QString("Performing runtime style validation scan #%1").arg(m_totalScansPerformed));
    
    QList<StyleViolation> newViolations = scanAllApplicationComponents();
    
    // Track new violations
    for (const StyleViolation& violation : newViolations) {
        // Check if this is a new violation
        bool isNew = true;
        for (const StyleViolation& existing : m_recentViolations) {
            if (existing.componentName == violation.componentName &&
                existing.violationType == violation.violationType &&
                existing.currentValue == violation.currentValue) {
                isNew = false;
                break;
            }
        }
        
        if (isNew) {
            m_recentViolations.append(violation);
            emit runtimeViolationDetected(violation);
            
            LOG_WARNING(LogCategories::UI, QString("New style violation detected: %1 in %2")
                       .arg(violation.violationType)
                       .arg(violation.componentName));
        }
    }
    
    // Limit recent violations to last 100 entries
    if (m_recentViolations.size() > 100) {
        m_recentViolations = m_recentViolations.mid(m_recentViolations.size() - 100);
    }
    
    emit scanCompleted(newViolations.size());
}

ComplianceReport StyleValidator::performComprehensiveApplicationScan()
{
    LOG_INFO(LogCategories::UI, "Starting comprehensive application style compliance scan");
    
    QWidgetList allWidgets = QApplication::allWidgets();
    QList<QWidget*> validWidgets;
    
    // Filter out null widgets
    for (QWidget* widget : allWidgets) {
        if (widget) {
            validWidgets.append(widget);
        }
    }
    
    ComplianceReport report = generateReport(validWidgets);
    
    // Add detailed logging
    logViolationDetails(report.criticalViolations);
    logViolationDetails(report.warnings);
    
    LOG_INFO(LogCategories::UI, QString("Comprehensive scan completed: %1/%2 components compliant (%.1f%% score)")
             .arg(report.compliantComponents)
             .arg(report.totalComponents)
             .arg(report.overallScore));
    
    emit complianceReportGenerated(report);
    
    return report;
}

QList<StyleViolation> StyleValidator::scanAllApplicationComponents()
{
    QList<StyleViolation> allViolations;
    QWidgetList allWidgets = QApplication::allWidgets();
    
    for (QWidget* widget : allWidgets) {
        if (widget) {
            QList<StyleViolation> widgetViolations = scanForHardcodedStyles(widget);
            allViolations.append(widgetViolations);
        }
    }
    
    return allViolations;
}

QList<StyleViolation> StyleValidator::scanSourceFiles(const QString& sourceDirectory)
{
    QList<StyleViolation> violations;
    
    QDir dir(sourceDirectory);
    if (!dir.exists()) {
        LOG_ERROR(LogCategories::UI, QString("Source directory does not exist: %1").arg(sourceDirectory));
        return violations;
    }
    
    // Recursively scan for C++ source files
    QStringList nameFilters;
    nameFilters << "*.cpp" << "*.h" << "*.hpp" << "*.cxx" << "*.cc";
    
    QFileInfoList files = dir.entryInfoList(nameFilters, QDir::Files | QDir::Readable, QDir::Name);
    
    for (const QFileInfo& fileInfo : files) {
        QList<StyleViolation> fileViolations = scanSourceFile(fileInfo.absoluteFilePath());
        violations.append(fileViolations);
    }
    
    // Recursively scan subdirectories
    QFileInfoList subdirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    for (const QFileInfo& subdirInfo : subdirs) {
        QList<StyleViolation> subdirViolations = scanSourceFiles(subdirInfo.absoluteFilePath());
        violations.append(subdirViolations);
    }
    
    LOG_INFO(LogCategories::UI, QString("Source file scan completed: %1 violations found in %2")
             .arg(violations.size())
             .arg(sourceDirectory));
    
    return violations;
}

QList<StyleViolation> StyleValidator::scanSourceFile(const QString& filePath)
{
    QList<StyleViolation> violations;
    
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        LOG_ERROR(LogCategories::UI, QString("Cannot open source file: %1").arg(filePath));
        return violations;
    }
    
    QTextStream stream(&file);
    QString content = stream.readAll();
    file.close();
    
    violations = analyzeSourceFileContent(content, QFileInfo(filePath).fileName());
    
    return violations;
}

void StyleValidator::generateDetailedComplianceReport(const QString& outputPath)
{
    ComplianceReport report = performComprehensiveApplicationScan();
    
    QString reportContent;
    QTextStream stream(&reportContent);
    
    stream << "=== DETAILED THEME COMPLIANCE REPORT ===\n";
    stream << "Generated: " << report.generated.toString() << "\n";
    stream << "Total Components: " << report.totalComponents << "\n";
    stream << "Compliant Components: " << report.compliantComponents << "\n";
    stream << "Overall Score: " << QString::number(report.overallScore, 'f', 2) << "%\n";
    stream << "Total Violations: " << report.violationCount << "\n\n";
    
    if (!report.criticalViolations.isEmpty()) {
        stream << "CRITICAL VIOLATIONS:\n";
        for (const StyleViolation& violation : report.criticalViolations) {
            stream << "  - " << violation.componentName << ": " << violation.violationType 
                   << " (" << violation.currentValue << ")\n";
            stream << "    Suggested Fix: " << violation.suggestedFix << "\n";
        }
        stream << "\n";
    }
    
    if (!report.warnings.isEmpty()) {
        stream << "WARNINGS:\n";
        for (const StyleViolation& violation : report.warnings) {
            stream << "  - " << violation.componentName << ": " << violation.violationType 
                   << " (" << violation.currentValue << ")\n";
            stream << "    Suggested Fix: " << violation.suggestedFix << "\n";
        }
        stream << "\n";
    }
    
    stream << "RECOMMENDATIONS:\n";
    stream << report.recommendations << "\n";
    
    // Output to file if path specified
    if (!outputPath.isEmpty()) {
        QFile outputFile(outputPath);
        if (outputFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream fileStream(&outputFile);
            fileStream << reportContent;
            outputFile.close();
            LOG_INFO(LogCategories::UI, QString("Detailed compliance report saved to: %1").arg(outputPath));
        } else {
            LOG_ERROR(LogCategories::UI, QString("Failed to save compliance report to: %1").arg(outputPath));
        }
    }
    
    // Always log to console
    LOG_INFO(LogCategories::UI, reportContent);
}

QStringList StyleValidator::getViolationSummary() const
{
    QMutexLocker locker(const_cast<QMutex*>(&m_violationMutex));
    
    QStringList summary;
    QMap<QString, int> violationCounts;
    
    for (const StyleViolation& violation : m_recentViolations) {
        QString key = QString("%1:%2").arg(violation.violationType).arg(violation.severity);
        violationCounts[key]++;
    }
    
    for (auto it = violationCounts.begin(); it != violationCounts.end(); ++it) {
        summary.append(QString("%1 (%2 occurrences)").arg(it.key()).arg(it.value()));
    }
    
    return summary;
}

void StyleValidator::logViolationDetails(const QList<StyleViolation>& violations)
{
    for (const StyleViolation& violation : violations) {
        QString logMessage = QString("Style Violation - Component: %1, Type: %2, Value: %3, Severity: %4")
                           .arg(violation.componentName)
                           .arg(violation.violationType)
                           .arg(violation.currentValue)
                           .arg(violation.severity);
        
        if (violation.severity == "critical") {
            LOG_ERROR(LogCategories::UI, logMessage);
        } else if (violation.severity == "warning") {
            LOG_WARNING(LogCategories::UI, logMessage);
        } else {
            LOG_INFO(LogCategories::UI, logMessage);
        }
        
        if (!violation.suggestedFix.isEmpty()) {
            LOG_INFO(LogCategories::UI, QString("  Suggested Fix: %1").arg(violation.suggestedFix));
        }
    }
}

void StyleValidator::clearViolationHistory()
{
    QMutexLocker locker(&m_violationMutex);
    m_recentViolations.clear();
    LOG_INFO(LogCategories::UI, "Style violation history cleared");
}

// Enhanced detection methods
QList<StyleViolation> StyleValidator::detectHardcodedFonts(QWidget* component)
{
    QList<StyleViolation> violations;
    
    if (!component) {
        return violations;
    }
    
    QString componentName = getComponentIdentifier(component);
    QString styleSheet = component->styleSheet();
    
    QRegularExpression fontPattern = getHardcodedFontPattern();
    QRegularExpressionMatchIterator matches = fontPattern.globalMatch(styleSheet);
    
    while (matches.hasNext()) {
        QRegularExpressionMatch match = matches.next();
        
        // Skip if using theme font variables
        QString matchedText = match.captured(0);
        if (matchedText.contains("theme.typography") || matchedText.contains("palette(")) {
            continue;
        }
        
        StyleViolation violation;
        violation.componentName = componentName;
        violation.violationType = "hardcoded-font";
        violation.currentValue = matchedText;
        violation.suggestedFix = "Use theme.typography.fontFamily from ThemeManager";
        violation.severity = "warning";
        violation.fileName = "runtime";
        violation.lineNumber = -1;
        
        violations.append(violation);
    }
    
    return violations;
}

QList<StyleViolation> StyleValidator::detectHardcodedSizes(QWidget* component)
{
    QList<StyleViolation> violations;
    
    if (!component) {
        return violations;
    }
    
    QString componentName = getComponentIdentifier(component);
    QString styleSheet = component->styleSheet();
    
    QRegularExpression sizePattern = getHardcodedSizePattern();
    QRegularExpressionMatchIterator matches = sizePattern.globalMatch(styleSheet);
    
    while (matches.hasNext()) {
        QRegularExpressionMatch match = matches.next();
        
        // Skip if using theme size variables
        QString matchedText = match.captured(0);
        if (matchedText.contains("theme.spacing") || matchedText.contains("em") || matchedText.contains("%")) {
            continue;
        }
        
        StyleViolation violation;
        violation.componentName = componentName;
        violation.violationType = "hardcoded-size";
        violation.currentValue = matchedText;
        violation.suggestedFix = "Use theme.spacing values or relative units from ThemeManager";
        violation.severity = "warning";
        violation.fileName = "runtime";
        violation.lineNumber = -1;
        
        violations.append(violation);
    }
    
    return violations;
}

QList<StyleViolation> StyleValidator::detectDeprecatedStyles(QWidget* component)
{
    QList<StyleViolation> violations;
    
    if (!component) {
        return violations;
    }
    
    QString componentName = getComponentIdentifier(component);
    QString styleSheet = component->styleSheet();
    
    // Check for deprecated style properties
    QStringList deprecatedPatterns = {
        "QWidget\\s*\\{[^}]*background\\s*:[^}]*\\}",  // Direct QWidget background
        "setStyleSheet\\s*\\([^)]*#[0-9a-fA-F]{3,6}",  // Direct hex colors in setStyleSheet calls
        "QPalette::[A-Z][a-zA-Z]*\\s*,\\s*QColor\\s*\\([^)]*\\)"  // Direct QPalette color assignments
    };
    
    for (const QString& pattern : deprecatedPatterns) {
        QRegularExpression regex(pattern);
        QRegularExpressionMatchIterator matches = regex.globalMatch(styleSheet);
        
        while (matches.hasNext()) {
            QRegularExpressionMatch match = matches.next();
            
            StyleViolation violation;
            violation.componentName = componentName;
            violation.violationType = "deprecated-style";
            violation.currentValue = match.captured(0);
            violation.suggestedFix = "Use ThemeManager methods for consistent styling";
            violation.severity = "warning";
            violation.fileName = "runtime";
            violation.lineNumber = -1;
            
            violations.append(violation);
        }
    }
    
    return violations;
}

QList<StyleViolation> StyleValidator::validateThemeConsistency(QWidget* component, const ThemeData& expectedTheme)
{
    QList<StyleViolation> violations;
    
    if (!component) {
        return violations;
    }
    
    QString componentName = getComponentIdentifier(component);
    
    // Check if component's actual colors match expected theme colors
    QPalette palette = component->palette();
    
    // Compare background colors
    QColor actualBg = palette.color(QPalette::Window);
    if (actualBg != expectedTheme.colors.background) {
        StyleViolation violation;
        violation.componentName = componentName;
        violation.violationType = "theme-inconsistency";
        violation.currentValue = QString("Background color mismatch: expected %1, got %2")
                               .arg(expectedTheme.colors.background.name())
                               .arg(actualBg.name());
        violation.suggestedFix = "Ensure component uses ThemeManager.applyThemeToComponent()";
        violation.severity = "warning";
        violation.fileName = "runtime";
        violation.lineNumber = -1;
        
        violations.append(violation);
    }
    
    // Compare text colors
    QColor actualFg = palette.color(QPalette::WindowText);
    if (actualFg != expectedTheme.colors.foreground) {
        StyleViolation violation;
        violation.componentName = componentName;
        violation.violationType = "theme-inconsistency";
        violation.currentValue = QString("Text color mismatch: expected %1, got %2")
                               .arg(expectedTheme.colors.foreground.name())
                               .arg(actualFg.name());
        violation.suggestedFix = "Ensure component uses ThemeManager.applyThemeToComponent()";
        violation.severity = "warning";
        violation.fileName = "runtime";
        violation.lineNumber = -1;
        
        violations.append(violation);
    }
    
    return violations;
}

// Helper methods for pattern matching
QRegularExpression StyleValidator::getHardcodedFontPattern()
{
    return QRegularExpression("font-family\\s*:\\s*[^;]+");
}

QRegularExpression StyleValidator::getHardcodedSizePattern()
{
    return QRegularExpression("(width|height|padding|margin|border-width)\\s*:\\s*\\d+px");
}

// Source file analysis helpers
QList<StyleViolation> StyleValidator::analyzeSourceFileContent(const QString& content, const QString& fileName)
{
    QList<StyleViolation> violations;
    
    // Find setStyleSheet calls with hardcoded colors
    QStringList styleSheetCalls = findStyleSheetCalls(content);
    
    for (const QString& call : styleSheetCalls) {
        // Check for hardcoded colors in the call
        if (hasHardcodedColors(call)) {
            QStringList colors = extractColors(call);
            
            for (const QString& color : colors) {
                StyleViolation violation;
                violation.componentName = fileName;
                violation.violationType = "hardcoded-color";
                violation.currentValue = color;
                violation.suggestedFix = suggestThemeAlternative(color);
                violation.severity = "critical";
                violation.fileName = fileName;
                violation.lineNumber = findLineNumber(content, call);
                
                violations.append(violation);
            }
        }
    }
    
    return violations;
}

QStringList StyleValidator::findStyleSheetCalls(const QString& content)
{
    QStringList calls;
    
    QRegularExpression pattern("setStyleSheet\\s*\\([^)]+\\)");
    QRegularExpressionMatchIterator matches = pattern.globalMatch(content);
    
    while (matches.hasNext()) {
        QRegularExpressionMatch match = matches.next();
        calls.append(match.captured(0));
    }
    
    return calls;
}

int StyleValidator::findLineNumber(const QString& content, const QString& searchText)
{
    QStringList lines = content.split('\n');
    
    for (int i = 0; i < lines.size(); ++i) {
        if (lines[i].contains(searchText)) {
            return i + 1; // Line numbers are 1-based
        }
    }
    
    return -1; // Not found
}

// Enhanced accessibility helper methods

QColor StyleValidator::suggestAccessibleColor(const QColor& baseColor, const QColor& background, double targetRatio)
{
    QColor suggestedColor = baseColor;
    double currentRatio = calculateContrastRatio(baseColor, background);
    
    if (currentRatio >= targetRatio) {
        return baseColor; // Already meets target
    }
    
    // Determine if we need to make the color lighter or darker
    bool makeLighter = background.lightness() > 128;
    
    // Adjust color iteratively to meet target ratio
    int iterations = 0;
    const int maxIterations = 100;
    
    while (currentRatio < targetRatio && iterations < maxIterations) {
        if (makeLighter) {
            suggestedColor = suggestedColor.lighter(105);
        } else {
            suggestedColor = suggestedColor.darker(105);
        }
        
        currentRatio = calculateContrastRatio(suggestedColor, background);
        iterations++;
    }
    
    return suggestedColor;
}

bool StyleValidator::isHighContrastModeEnabled()
{
    // Check if system high contrast mode is enabled
    #ifdef Q_OS_WIN
    return GetSystemMetrics(SM_CXBORDER) > 1;
    #elif defined(Q_OS_LINUX)
    QSettings settings("org.gnome.desktop.interface", QSettings::NativeFormat);
    return settings.value("high-contrast", false).toBool();
    #else
    return false;
    #endif
}

QList<StyleViolation> StyleValidator::validateFocusIndicators(QWidget* component)
{
    QList<StyleViolation> violations;
    
    if (!component) {
        return violations;
    }
    
    QString componentName = getComponentIdentifier(component);
    
    // Check if component can receive focus
    if (component->focusPolicy() == Qt::NoFocus) {
        // Check if this is an interactive element that should have focus
        if (qobject_cast<QPushButton*>(component) ||
            qobject_cast<QLineEdit*>(component) ||
            qobject_cast<QComboBox*>(component) ||
            qobject_cast<QCheckBox*>(component) ||
            qobject_cast<QRadioButton*>(component)) {
            
            StyleViolation violation;
            violation.componentName = componentName;
            violation.violationType = "accessibility-focus-policy";
            violation.currentValue = "Interactive element cannot receive focus";
            violation.suggestedFix = "Set focus policy to Qt::StrongFocus or Qt::TabFocus";
            violation.severity = "warning";
            violation.fileName = "runtime";
            violation.lineNumber = -1;
            
            violations.append(violation);
        }
    }
    
    // Check for visible focus indicators in stylesheet
    QString styleSheet = component->styleSheet();
    if (!styleSheet.contains(":focus") && component->focusPolicy() != Qt::NoFocus) {
        StyleViolation violation;
        violation.componentName = componentName;
        violation.violationType = "accessibility-focus-indicator";
        violation.currentValue = "No visible focus indicator defined";
        violation.suggestedFix = "Add :focus pseudo-state styling with visible outline or border";
        violation.severity = "warning";
        violation.fileName = "runtime";
        violation.lineNumber = -1;
        
        violations.append(violation);
    }
    
    return violations;
}

QList<StyleViolation> StyleValidator::validateColorOnlyInformation(QWidget* component)
{
    QList<StyleViolation> violations;
    
    if (!component) {
        return violations;
    }
    
    QString componentName = getComponentIdentifier(component);
    QString styleSheet = component->styleSheet();
    
    // Check for color-only status indicators
    QStringList colorOnlyPatterns = {
        "color\\s*:\\s*(red|green|#ff0000|#00ff00|rgb\\(255,\\s*0,\\s*0\\)|rgb\\(0,\\s*255,\\s*0\\))",
        "background-color\\s*:\\s*(red|green|yellow|#ff0000|#00ff00|#ffff00)"
    };
    
    for (const QString& pattern : colorOnlyPatterns) {
        QRegularExpression regex(pattern, QRegularExpression::CaseInsensitiveOption);
        if (regex.match(styleSheet).hasMatch()) {
            // Check if there are alternative indicators
            QString tooltip = component->toolTip();
            QString text = component->property("text").toString();
            
            if (tooltip.isEmpty() && text.isEmpty()) {
                StyleViolation violation;
                violation.componentName = componentName;
                violation.violationType = "accessibility-color-only-information";
                violation.currentValue = "Information conveyed through color only";
                violation.suggestedFix = "Add text labels, icons, or tooltips to supplement color information";
                violation.severity = "warning";
                violation.fileName = "runtime";
                violation.lineNumber = -1;
                
                violations.append(violation);
            }
        }
    }
    
    return violations;
}