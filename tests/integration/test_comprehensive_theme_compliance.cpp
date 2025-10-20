#include "test_base.h"
#include "theme_manager.h"
#include "style_validator.h"
#include "theme_accessibility_testing.h"
#include "ui_automation.h"
#include "visual_testing.h"
#include <QApplication>
#include <QWidget>
#include <QDialog>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QCheckBox>
#include <QComboBox>
#include <QProgressBar>
#include <QTextEdit>
#include <QGroupBox>
#include <QTabWidget>
#include <QTest>

/**
 * @brief Comprehensive theme compliance testing that integrates StyleValidator 
 * with existing ThemeAccessibilityTesting framework
 */
DECLARE_TEST_CLASS(ComprehensiveThemeComplianceTest, Integration, High, "theme", "compliance", "validation", "accessibility")

private:
    ThemeManager* m_themeManager;
    StyleValidator* m_styleValidator;
    ThemeAccessibilityTesting* m_themeAccessibilityTesting;
    UIAutomation* m_uiAutomation;
    VisualTesting* m_visualTesting;
    QWidget* m_testWidget;

private slots:
    void initTestCase() {
        TestBase::initTestCase();
        logTestInfo("Setting up comprehensive theme compliance testing");
        
        // Initialize theme manager and validator
        m_themeManager = ThemeManager::instance();
        m_styleValidator = new StyleValidator(this);
        
        // Initialize testing frameworks
        m_themeAccessibilityTesting = new ThemeAccessibilityTesting(this);
        m_uiAutomation = new UIAutomation(this);
        m_visualTesting = new VisualTesting(this);
        
        // Configure accessibility testing
        ThemeAccessibilityTesting::AccessibilityConfig config;
        config.complianceLevel = ThemeAccessibilityTesting::AccessibilityLevel::WCAG_AA;
        config.testColorContrast = true;
        config.testKeyboardNav = true;
        config.testScreenReader = true;
        config.minContrastRatio = 4.5;
        m_themeAccessibilityTesting->setAccessibilityConfig(config);
        
        // Link frameworks
        m_themeAccessibilityTesting->setUIAutomation(m_uiAutomation);
        m_themeAccessibilityTesting->setVisualTesting(m_visualTesting);
        
        // Create comprehensive test widget
        createTestWidget();
        
        logTestInfo("Comprehensive theme compliance testing setup completed");
    }
    
    void cleanupTestCase() {
        if (m_testWidget) {
            m_testWidget->deleteLater();
            m_testWidget = nullptr;
        }
        TestBase::cleanupTestCase();
    }
    
    void testHardcodedStyleDetection() {
        logTestInfo("Testing automated hardcoded style detection");
        
        // Create widget with intentional hardcoded styles for testing
        QWidget* testWidget = new QWidget();
        testWidget->setObjectName("HardcodedStyleTestWidget");
        
        // Add hardcoded styles that should be detected
        testWidget->setStyleSheet(
            "QWidget { background-color: #ff0000; color: rgb(255, 255, 255); }"
            "QPushButton { border: 2px solid #00ff00; font-family: Arial; }"
        );
        
        // Test detection
        QList<StyleViolation> violations = m_styleValidator->scanForHardcodedStyles(testWidget);
        
        // Verify violations were detected
        QVERIFY(!violations.isEmpty());
        
        bool foundHardcodedColor = false;
        bool foundHardcodedFont = false;
        
        for (const StyleViolation& violation : violations) {
            if (violation.violationType == "hardcoded-color") {
                foundHardcodedColor = true;
                QVERIFY(violation.severity == "critical");
                QVERIFY(!violation.suggestedFix.isEmpty());
            }
            if (violation.violationType == "hardcoded-font") {
                foundHardcodedFont = true;
            }
        }
        
        QVERIFY(foundHardcodedColor);
        
        logTestInfo(QString("Detected %1 style violations as expected").arg(violations.size()));
        
        testWidget->deleteLater();
    }
    
    void testRuntimeStyleValidation() {
        logTestInfo("Testing runtime style validation system");
        
        // Enable runtime scanning
        m_styleValidator->enableRuntimeScanning(true);
        m_styleValidator->setRuntimeScanInterval(1000); // 1 second for testing
        
        // Create signal spy to monitor violations
        QSignalSpy violationSpy(m_styleValidator, &StyleValidator::runtimeViolationDetected);
        QSignalSpy scanSpy(m_styleValidator, &StyleValidator::scanCompleted);
        
        // Create widget with violations
        QWidget* violatingWidget = new QWidget();
        violatingWidget->setStyleSheet("background-color: #123456;");
        violatingWidget->show();
        
        // Wait for runtime scan to detect violations
        QVERIFY(scanSpy.wait(2000));
        
        // Verify violations were detected
        QVERIFY(violationSpy.count() > 0);
        
        // Clean up
        violatingWidget->deleteLater();
        m_styleValidator->enableRuntimeScanning(false);
        
        logTestInfo("Runtime validation system working correctly");
    }
    
    void testComprehensiveApplicationScan() {
        logTestInfo("Testing comprehensive application-wide style scanning");
        
        // Perform comprehensive scan
        ComplianceReport report = m_styleValidator->performComprehensiveApplicationScan();
        
        // Verify report structure
        QVERIFY(report.generated.isValid());
        QVERIFY(report.totalComponents >= 0);
        QVERIFY(report.compliantComponents >= 0);
        QVERIFY(report.compliantComponents <= report.totalComponents);
        QVERIFY(report.overallScore >= 0.0 && report.overallScore <= 100.0);
        
        logTestInfo(QString("Application scan completed: %1/%2 components compliant (%.1f%% score)")
                   .arg(report.compliantComponents)
                   .arg(report.totalComponents)
                   .arg(report.overallScore));
        
        // Log violations for analysis
        if (!report.criticalViolations.isEmpty()) {
            logTestInfo(QString("Found %1 critical violations").arg(report.criticalViolations.size()));
            for (const StyleViolation& violation : report.criticalViolations) {
                logTestInfo(QString("  Critical: %1 in %2 (%3)")
                           .arg(violation.violationType)
                           .arg(violation.componentName)
                           .arg(violation.currentValue));
            }
        }
        
        if (!report.warnings.isEmpty()) {
            logTestInfo(QString("Found %1 warnings").arg(report.warnings.size()));
        }
    }
    
    void testThemeConsistencyValidation() {
        logTestInfo("Testing theme consistency validation across all themes");
        
        // Test light theme consistency
        m_themeManager->setTheme(ThemeManager::Light);
        QTest::qWait(100); // Allow theme to apply
        
        ThemeData lightTheme = m_themeManager->getCurrentThemeData();
        QList<StyleViolation> lightViolations = StyleValidator::validateThemeConsistency(m_testWidget, lightTheme);
        
        logTestInfo(QString("Light theme consistency: %1 violations").arg(lightViolations.size()));
        
        // Test dark theme consistency
        m_themeManager->setTheme(ThemeManager::Dark);
        QTest::qWait(100); // Allow theme to apply
        
        ThemeData darkTheme = m_themeManager->getCurrentThemeData();
        QList<StyleViolation> darkViolations = StyleValidator::validateThemeConsistency(m_testWidget, darkTheme);
        
        logTestInfo(QString("Dark theme consistency: %1 violations").arg(darkViolations.size()));
        
        // Verify themes are different but both valid
        QVERIFY(lightTheme.colors.background != darkTheme.colors.background);
        QVERIFY(lightTheme.colors.foreground != darkTheme.colors.foreground);
    }
    
    void testAccessibilityIntegration() {
        logTestInfo("Testing integration with accessibility testing framework");
        
        // Test both themes for accessibility compliance
        QList<ThemeAccessibilityTesting::ThemeType> themes = {
            ThemeAccessibilityTesting::ThemeType::Light,
            ThemeAccessibilityTesting::ThemeType::Dark
        };
        
        for (auto themeType : themes) {
            QString themeName = (themeType == ThemeAccessibilityTesting::ThemeType::Light) ? "Light" : "Dark";
            logTestInfo(QString("Testing accessibility compliance for %1 theme").arg(themeName));
            
            // Switch to theme
            QVERIFY(m_themeAccessibilityTesting->switchToTheme(themeType));
            
            // Test color contrast
            auto contrastResults = m_themeAccessibilityTesting->testAllColorContrasts(m_testWidget);
            
            int passedContrast = 0;
            for (const auto& result : contrastResults) {
                if (result.passes) {
                    passedContrast++;
                }
            }
            
            logTestInfo(QString("  Color contrast: %1/%2 passed").arg(passedContrast).arg(contrastResults.size()));
            
            // Test keyboard navigation
            auto keyboardResults = m_themeAccessibilityTesting->testAllKeyboardNavigation(m_testWidget);
            
            int passedKeyboard = 0;
            for (const auto& result : keyboardResults) {
                if (result.canReceiveFocus && result.tabOrderCorrect) {
                    passedKeyboard++;
                }
            }
            
            logTestInfo(QString("  Keyboard navigation: %1/%2 passed").arg(passedKeyboard).arg(keyboardResults.size()));
            
            // Validate theme using StyleValidator
            ThemeData currentTheme = m_themeManager->getCurrentThemeData();
            bool accessibilityValid = StyleValidator::validateAccessibility(currentTheme);
            
            logTestInfo(QString("  StyleValidator accessibility check: %1").arg(accessibilityValid ? "PASSED" : "FAILED"));
        }
    }
    
    void testIntegratedComplianceReport() {
        logTestInfo("Testing integrated compliance reporting");
        
        // Generate comprehensive report using ThemeManager
        ComplianceReport report = m_themeManager->generateComplianceReport();
        
        // Verify report completeness
        QVERIFY(report.generated.isValid());
        QVERIFY(!report.recommendations.isEmpty());
        
        // Generate detailed report to file
        QString reportPath = "test_compliance_report.txt";
        m_themeManager->generateDetailedValidationReport(reportPath);
        
        // Verify file was created
        QFile reportFile(reportPath);
        QVERIFY(reportFile.exists());
        
        // Clean up
        reportFile.remove();
        
        logTestInfo(QString("Integrated compliance report generated successfully"));
        logTestInfo(QString("  Total components: %1").arg(report.totalComponents));
        logTestInfo(QString("  Compliant components: %1").arg(report.compliantComponents));
        logTestInfo(QString("  Overall score: %.1f%%").arg(report.overallScore));
        logTestInfo(QString("  Total violations: %1").arg(report.violationCount));
        logTestInfo(QString("  Critical violations: %1").arg(report.criticalViolations.size()));
        logTestInfo(QString("  Warnings: %1").arg(report.warnings.size()));
    }
    
    void testSourceCodeValidation() {
        logTestInfo("Testing source code validation capabilities");
        
        // Test source code scanning (if src directory exists)
        QDir srcDir("src");
        if (srcDir.exists()) {
            QList<StyleViolation> sourceViolations = m_themeManager->validateSourceCode("src");
            
            logTestInfo(QString("Source code validation found %1 violations").arg(sourceViolations.size()));
            
            // Categorize violations
            QMap<QString, int> violationTypes;
            QMap<QString, int> severityLevels;
            
            for (const StyleViolation& violation : sourceViolations) {
                violationTypes[violation.violationType]++;
                severityLevels[violation.severity]++;
            }
            
            // Log summary
            for (auto it = violationTypes.begin(); it != violationTypes.end(); ++it) {
                logTestInfo(QString("  %1: %2 occurrences").arg(it.key()).arg(it.value()));
            }
            
            for (auto it = severityLevels.begin(); it != severityLevels.end(); ++it) {
                logTestInfo(QString("  %1 severity: %2 violations").arg(it.key()).arg(it.value()));
            }
        } else {
            logTestInfo("Source directory not found, skipping source code validation");
        }
    }
    
    void testPerformanceOfValidation() {
        logTestInfo("Testing validation system performance");
        
        QElapsedTimer timer;
        
        // Test component scanning performance
        timer.start();
        QList<StyleViolation> violations = m_styleValidator->scanAllApplicationComponents();
        qint64 scanTime = timer.elapsed();
        
        logTestInfo(QString("Component scanning took %1ms for %2 violations").arg(scanTime).arg(violations.size()));
        
        // Test comprehensive validation performance
        timer.restart();
        ComplianceReport report = m_styleValidator->performComprehensiveApplicationScan();
        qint64 reportTime = timer.elapsed();
        
        logTestInfo(QString("Comprehensive validation took %1ms").arg(reportTime));
        
        // Performance should be reasonable (less than 5 seconds for typical applications)
        QVERIFY(scanTime < 5000);
        QVERIFY(reportTime < 5000);
    }
    
    void testValidationStatistics() {
        logTestInfo("Testing validation statistics and tracking");
        
        // Clear previous statistics
        m_styleValidator->clearViolationHistory();
        
        // Perform several scans
        for (int i = 0; i < 3; ++i) {
            m_styleValidator->performRuntimeScan();
            QTest::qWait(100);
        }
        
        // Check statistics
        int totalScans = m_styleValidator->getTotalScansPerformed();
        QDateTime lastScan = m_styleValidator->getLastScanTime();
        QStringList summary = m_styleValidator->getViolationSummary();
        
        QVERIFY(totalScans >= 3);
        QVERIFY(lastScan.isValid());
        
        logTestInfo(QString("Validation statistics: %1 scans performed").arg(totalScans));
        logTestInfo(QString("Last scan: %1").arg(lastScan.toString()));
        logTestInfo(QString("Violation summary: %1 types").arg(summary.size()));
        
        for (const QString& summaryItem : summary) {
            logTestInfo(QString("  %1").arg(summaryItem));
        }
    }

private:
    void createTestWidget() {
        m_testWidget = new QWidget();
        m_testWidget->setObjectName("ComprehensiveTestWidget");
        m_testWidget->setWindowTitle("Theme Compliance Test Widget");
        
        QVBoxLayout* mainLayout = new QVBoxLayout(m_testWidget);
        
        // Create various UI elements for comprehensive testing
        
        // Basic controls
        QGroupBox* basicGroup = new QGroupBox("Basic Controls");
        QVBoxLayout* basicLayout = new QVBoxLayout(basicGroup);
        
        basicLayout->addWidget(new QLabel("Test Label"));
        basicLayout->addWidget(new QLineEdit("Test input"));
        basicLayout->addWidget(new QPushButton("Test Button"));
        basicLayout->addWidget(new QCheckBox("Test Checkbox"));
        
        QComboBox* combo = new QComboBox();
        combo->addItems({"Option 1", "Option 2", "Option 3"});
        basicLayout->addWidget(combo);
        
        mainLayout->addWidget(basicGroup);
        
        // Progress and status
        QGroupBox* progressGroup = new QGroupBox("Progress Controls");
        QVBoxLayout* progressLayout = new QVBoxLayout(progressGroup);
        
        QProgressBar* progress = new QProgressBar();
        progress->setValue(50);
        progressLayout->addWidget(progress);
        
        mainLayout->addWidget(progressGroup);
        
        // Text areas
        QGroupBox* textGroup = new QGroupBox("Text Areas");
        QVBoxLayout* textLayout = new QVBoxLayout(textGroup);
        
        QTextEdit* textEdit = new QTextEdit();
        textEdit->setPlainText("Sample text for testing");
        textEdit->setMaximumHeight(100);
        textLayout->addWidget(textEdit);
        
        mainLayout->addWidget(textGroup);
        
        // Tab widget
        QTabWidget* tabs = new QTabWidget();
        tabs->addTab(new QLabel("Tab 1 Content"), "Tab 1");
        tabs->addTab(new QLabel("Tab 2 Content"), "Tab 2");
        mainLayout->addWidget(tabs);
        
        m_testWidget->resize(400, 500);
        m_testWidget->show();
        
        // Apply current theme to ensure consistency
        m_themeManager->applyToWidget(m_testWidget);
    }
};

#include "test_comprehensive_theme_compliance.moc"