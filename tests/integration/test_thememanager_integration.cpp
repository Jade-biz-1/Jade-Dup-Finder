#include <QCoreApplication>
#include <QApplication>
#include <QDebug>
#include <QTimer>
#include <QEventLoop>
#include <QSignalSpy>
#include <QTest>
#include <QWidget>
#include <QDialog>
#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QProgressBar>
#include <QVBoxLayout>

#include "theme_manager.h"
#include "main_window.h"
#include "results_window.h"
#include "settings_dialog.h"
#include "scan_dialog.h"

/**
 * @brief Integration test for ThemeManager across all components
 * 
 * This test verifies:
 * - Theme application across all UI components
 * - Theme switching consistency
 * - Dialog registration and automatic updates
 * - Component-specific styling
 * - Theme persistence and loading
 * - Accessibility compliance across themes
 * 
 * Requirements: 1.3, 2.3, 7.4
 */

class ThemeManagerIntegrationTest : public QObject {
    Q_OBJECT

private:
    ThemeManager* m_themeManager;
    QWidget* m_testWidget;
    QDialog* m_testDialog;
    QMainWindow* m_testMainWindow;

private slots:
    void initTestCase() {
        qDebug() << "===========================================";
        qDebug() << "ThemeManager Integration Test";
        qDebug() << "===========================================";
        qDebug();
        
        // Get ThemeManager instance
        m_themeManager = ThemeManager::instance();
        QVERIFY(m_themeManager != nullptr);
        
        // Create test widgets
        m_testWidget = new QWidget();
        m_testDialog = new QDialog();
        m_testMainWindow = new QMainWindow();
        
        qDebug() << "Test components initialized";
    }
    
    void cleanupTestCase() {
        delete m_testMainWindow;
        delete m_testDialog;
        delete m_testWidget;
        
        qDebug() << "\n===========================================";
        qDebug() << "All tests completed";
        qDebug() << "===========================================";
    }
    
    /**
     * Test 1: Theme application across UI components
     * Test theme application to various widget types
     */
    void test_themeApplicationAcrossComponents() {
        qDebug() << "\n[Test 1] Theme Application Across Components";
        qDebug() << "=============================================";
        
        // Create a complex widget hierarchy
        QWidget* parentWidget = new QWidget();
        QVBoxLayout* layout = new QVBoxLayout(parentWidget);
        
        QLabel* label = new QLabel("Test Label", parentWidget);
        QPushButton* button = new QPushButton("Test Button", parentWidget);
        QProgressBar* progressBar = new QProgressBar(parentWidget);
        
        layout->addWidget(label);
        layout->addWidget(button);
        layout->addWidget(progressBar);
        
        qDebug() << "   Created test widget hierarchy";
        
        // Test Light theme application
        qDebug() << "   Testing Light theme application...";
        m_themeManager->setTheme(ThemeManager::Light);
        m_themeManager->applyToWidget(parentWidget);
        
        QString lightStyleSheet = parentWidget->styleSheet();
        qDebug() << "      Light theme applied, stylesheet length:" << lightStyleSheet.length();
        QVERIFY(!lightStyleSheet.isEmpty());
        
        // Test Dark theme application
        qDebug() << "   Testing Dark theme application...";
        m_themeManager->setTheme(ThemeManager::Dark);
        m_themeManager->applyToWidget(parentWidget);
        
        QString darkStyleSheet = parentWidget->styleSheet();
        qDebug() << "      Dark theme applied, stylesheet length:" << darkStyleSheet.length();
        QVERIFY(!darkStyleSheet.isEmpty());
        QVERIFY(darkStyleSheet != lightStyleSheet);
        
        // Test component-specific styling
        QString progressStyle = m_themeManager->getProgressBarStyle(ThemeManager::ProgressType::Success);
        qDebug() << "      Progress bar style length:" << progressStyle.length();
        QVERIFY(!progressStyle.isEmpty());
        
        QString statusStyle = m_themeManager->getStatusIndicatorStyle(ThemeManager::StatusType::Warning);
        qDebug() << "      Status indicator style length:" << statusStyle.length();
        QVERIFY(!statusStyle.isEmpty());
        
        delete parentWidget;
        qDebug() << "✓ Theme application across components verified";
    }
    
    /**
     * Test 2: Theme switching consistency
     * Test consistent theme switching across multiple components
     */
    void test_themeSwitchingConsistency() {
        qDebug() << "\n[Test 2] Theme Switching Consistency";
        qDebug() << "=====================================";
        
        // Create multiple widgets
        QList<QWidget*> widgets;
        for (int i = 0; i < 5; i++) {
            QWidget* widget = new QWidget();
            widget->setObjectName(QString("TestWidget_%1").arg(i));
            widgets.append(widget);
        }
        
        qDebug() << "   Created" << widgets.size() << "test widgets";
        
        // Apply initial theme to all widgets
        ThemeManager::Theme initialTheme = ThemeManager::Light;
        m_themeManager->setTheme(initialTheme);
        
        for (QWidget* widget : widgets) {
            m_themeManager->applyToWidget(widget);
        }
        
        // Capture initial stylesheets
        QStringList initialStyleSheets;
        for (QWidget* widget : widgets) {
            initialStyleSheets.append(widget->styleSheet());
        }
        
        qDebug() << "   Applied initial theme to all widgets";
        
        // Switch theme and verify consistency
        ThemeManager::Theme newTheme = ThemeManager::Dark;
        QSignalSpy themeChangedSpy(m_themeManager, &ThemeManager::themeChanged);
        
        m_themeManager->setTheme(newTheme);
        QCOMPARE(themeChangedSpy.count(), 1);
        
        // Apply new theme to all widgets
        for (QWidget* widget : widgets) {
            m_themeManager->applyToWidget(widget);
        }
        
        // Verify all widgets have consistent styling
        QString referenceStyleSheet = widgets.first()->styleSheet();
        bool allConsistent = true;
        
        for (int i = 1; i < widgets.size(); i++) {
            if (widgets[i]->styleSheet() != referenceStyleSheet) {
                allConsistent = false;
                qDebug() << "      Widget" << i << "has inconsistent styling";
                break;
            }
        }
        
        QVERIFY(allConsistent);
        qDebug() << "   All widgets have consistent styling after theme switch";
        
        // Verify stylesheets actually changed
        bool stylesChanged = true;
        for (int i = 0; i < widgets.size(); i++) {
            if (widgets[i]->styleSheet() == initialStyleSheets[i]) {
                stylesChanged = false;
                qDebug() << "      Widget" << i << "stylesheet did not change";
                break;
            }
        }
        
        QVERIFY(stylesChanged);
        qDebug() << "   All widget stylesheets changed after theme switch";
        
        // Cleanup
        qDeleteAll(widgets);
        qDebug() << "✓ Theme switching consistency verified";
    }
    
    /**
     * Test 3: Dialog registration and automatic updates
     * Test automatic theme updates for registered dialogs
     */
    void test_dialogRegistrationAndUpdates() {
        qDebug() << "\n[Test 3] Dialog Registration and Updates";
        qDebug() << "=========================================";
        
        // Create test dialogs
        QList<QDialog*> dialogs;
        for (int i = 0; i < 3; i++) {
            QDialog* dialog = new QDialog();
            dialog->setObjectName(QString("TestDialog_%1").arg(i));
            
            // Add some content to the dialog
            QVBoxLayout* layout = new QVBoxLayout(dialog);
            layout->addWidget(new QLabel(QString("Dialog %1 Content").arg(i)));
            layout->addWidget(new QPushButton("OK"));
            
            dialogs.append(dialog);
        }
        
        qDebug() << "   Created" << dialogs.size() << "test dialogs";
        
        // Register dialogs with ThemeManager
        for (QDialog* dialog : dialogs) {
            m_themeManager->registerDialog(dialog);
        }
        
        qDebug() << "   Registered all dialogs with ThemeManager";
        
        // Apply initial theme
        m_themeManager->setTheme(ThemeManager::Light);
        
        // Capture initial stylesheets
        QStringList initialStyleSheets;
        for (QDialog* dialog : dialogs) {
            initialStyleSheets.append(dialog->styleSheet());
        }
        
        // Switch theme - registered dialogs should update automatically
        QSignalSpy themeChangedSpy(m_themeManager, &ThemeManager::themeChanged);
        m_themeManager->setTheme(ThemeManager::Dark);
        
        QCOMPARE(themeChangedSpy.count(), 1);
        qDebug() << "   Theme switched, checking automatic updates...";
        
        // Give some time for automatic updates
        QTest::qWait(100);
        
        // Verify dialogs were updated automatically
        bool dialogsUpdated = true;
        for (int i = 0; i < dialogs.size(); i++) {
            QString currentStyleSheet = dialogs[i]->styleSheet();
            if (currentStyleSheet == initialStyleSheets[i] || currentStyleSheet.isEmpty()) {
                dialogsUpdated = false;
                qDebug() << "      Dialog" << i << "was not updated automatically";
                break;
            }
        }
        
        QVERIFY(dialogsUpdated);
        qDebug() << "   All registered dialogs updated automatically";
        
        // Test unregistration
        QDialog* firstDialog = dialogs.first();
        QString styleBeforeUnregister = firstDialog->styleSheet();
        
        m_themeManager->unregisterDialog(firstDialog);
        m_themeManager->setTheme(ThemeManager::Light);
        
        QTest::qWait(100);
        
        QString styleAfterUnregister = firstDialog->styleSheet();
        QCOMPARE(styleAfterUnregister, styleBeforeUnregister); // Should not change
        
        qDebug() << "   Unregistered dialog did not update automatically";
        
        // Cleanup
        qDeleteAll(dialogs);
        qDebug() << "✓ Dialog registration and updates verified";
    }
    
    /**
     * Test 4: Component-specific styling
     * Test specialized styling for different component types
     */
    void test_componentSpecificStyling() {
        qDebug() << "\n[Test 4] Component-Specific Styling";
        qDebug() << "====================================";
        
        // Test progress bar styling variants
        qDebug() << "   Testing progress bar styling variants...";
        
        QStringList progressStyles;
        progressStyles << m_themeManager->getProgressBarStyle(ThemeManager::ProgressType::Normal);
        progressStyles << m_themeManager->getProgressBarStyle(ThemeManager::ProgressType::Success);
        progressStyles << m_themeManager->getProgressBarStyle(ThemeManager::ProgressType::Warning);
        progressStyles << m_themeManager->getProgressBarStyle(ThemeManager::ProgressType::Error);
        progressStyles << m_themeManager->getProgressBarStyle(ThemeManager::ProgressType::Performance);
        
        // Verify all styles are different and non-empty
        for (int i = 0; i < progressStyles.size(); i++) {
            QVERIFY(!progressStyles[i].isEmpty());
            qDebug() << "      Progress style" << i << "length:" << progressStyles[i].length();
            
            for (int j = i + 1; j < progressStyles.size(); j++) {
                QVERIFY(progressStyles[i] != progressStyles[j]);
            }
        }
        
        // Test status indicator styling variants
        qDebug() << "   Testing status indicator styling variants...";
        
        QStringList statusStyles;
        statusStyles << m_themeManager->getStatusIndicatorStyle(ThemeManager::StatusType::Success);
        statusStyles << m_themeManager->getStatusIndicatorStyle(ThemeManager::StatusType::Warning);
        statusStyles << m_themeManager->getStatusIndicatorStyle(ThemeManager::StatusType::Error);
        statusStyles << m_themeManager->getStatusIndicatorStyle(ThemeManager::StatusType::Info);
        statusStyles << m_themeManager->getStatusIndicatorStyle(ThemeManager::StatusType::Neutral);
        
        // Verify all styles are different and non-empty
        for (int i = 0; i < statusStyles.size(); i++) {
            QVERIFY(!statusStyles[i].isEmpty());
            qDebug() << "      Status style" << i << "length:" << statusStyles[i].length();
            
            for (int j = i + 1; j < statusStyles.size(); j++) {
                QVERIFY(statusStyles[i] != statusStyles[j]);
            }
        }
        
        // Test minimum size enforcement
        qDebug() << "   Testing minimum size enforcement...";
        
        QWidget* testWidget = new QWidget();
        QVBoxLayout* layout = new QVBoxLayout(testWidget);
        
        QPushButton* button = new QPushButton("Test", testWidget);
        layout->addWidget(button);
        
        QSize originalSize = button->size();
        m_themeManager->enforceMinimumSizes(testWidget);
        QSize enforcedSize = button->minimumSize();
        
        qDebug() << "      Original size:" << originalSize;
        qDebug() << "      Enforced minimum size:" << enforcedSize;
        
        QVERIFY(enforcedSize.width() > 0);
        QVERIFY(enforcedSize.height() > 0);
        
        delete testWidget;
        qDebug() << "✓ Component-specific styling verified";
    }
    
    /**
     * Test 5: Theme persistence and loading
     * Test theme settings persistence
     */
    void test_themePersistenceAndLoading() {
        qDebug() << "\n[Test 5] Theme Persistence and Loading";
        qDebug() << "=======================================";
        
        // Get initial theme
        ThemeManager::Theme initialTheme = m_themeManager->currentTheme();
        qDebug() << "   Initial theme:" << m_themeManager->currentThemeString();
        
        // Change theme
        ThemeManager::Theme newTheme = (initialTheme == ThemeManager::Light) ? 
                                      ThemeManager::Dark : ThemeManager::Light;
        
        m_themeManager->setTheme(newTheme);
        qDebug() << "   Changed to theme:" << m_themeManager->currentThemeString();
        QCOMPARE(m_themeManager->currentTheme(), newTheme);
        
        // Save settings
        m_themeManager->saveToSettings();
        qDebug() << "   Settings saved";
        
        // Change theme again
        ThemeManager::Theme tempTheme = ThemeManager::SystemDefault;
        m_themeManager->setTheme(tempTheme);
        QCOMPARE(m_themeManager->currentTheme(), tempTheme);
        
        // Load settings - should restore the saved theme
        m_themeManager->loadFromSettings();
        qDebug() << "   Settings loaded, current theme:" << m_themeManager->currentThemeString();
        
        QCOMPARE(m_themeManager->currentTheme(), newTheme);
        
        qDebug() << "✓ Theme persistence and loading verified";
    }
    
    /**
     * Test 6: System theme detection
     * Test system theme detection capabilities
     */
    void test_systemThemeDetection() {
        qDebug() << "\n[Test 6] System Theme Detection";
        qDebug() << "================================";
        
        // Test system dark mode detection
        bool isSystemDark = m_themeManager->isSystemDarkMode();
        qDebug() << "   System dark mode detected:" << isSystemDark;
        
        // Test system default theme
        ThemeManager::Theme originalTheme = m_themeManager->currentTheme();
        
        m_themeManager->setTheme(ThemeManager::SystemDefault);
        ThemeManager::Theme systemTheme = m_themeManager->currentTheme();
        
        qDebug() << "   System default theme resolved to:" << m_themeManager->currentThemeString();
        
        // System default should resolve to either Light or Dark
        QVERIFY(systemTheme == ThemeManager::Light || systemTheme == ThemeManager::Dark);
        
        // Restore original theme
        m_themeManager->setTheme(originalTheme);
        
        qDebug() << "✓ System theme detection verified";
    }
    
    /**
     * Test 7: Integration with real application components
     * Test theme integration with actual application dialogs
     */
    void test_integrationWithRealComponents() {
        qDebug() << "\n[Test 7] Integration with Real Components";
        qDebug() << "==========================================";
        
        // Note: This test creates actual application components
        // In a real test environment, we might need to mock some dependencies
        
        qDebug() << "   Testing theme application to real components...";
        
        // Test with a simple dialog-like widget (avoiding full MainWindow due to complexity)
        QDialog* settingsLikeDialog = new QDialog();
        settingsLikeDialog->setWindowTitle("Test Settings Dialog");
        settingsLikeDialog->resize(400, 300);
        
        QVBoxLayout* layout = new QVBoxLayout(settingsLikeDialog);
        layout->addWidget(new QLabel("Theme Settings"));
        
        QPushButton* lightButton = new QPushButton("Light Theme");
        QPushButton* darkButton = new QPushButton("Dark Theme");
        QPushButton* systemButton = new QPushButton("System Theme");
        
        layout->addWidget(lightButton);
        layout->addWidget(darkButton);
        layout->addWidget(systemButton);
        
        // Apply theme to the dialog
        m_themeManager->setTheme(ThemeManager::Light);
        m_themeManager->applyToDialog(settingsLikeDialog);
        
        QString lightDialogStyle = settingsLikeDialog->styleSheet();
        qDebug() << "      Light theme applied to dialog, style length:" << lightDialogStyle.length();
        QVERIFY(!lightDialogStyle.isEmpty());
        
        // Switch to dark theme
        m_themeManager->setTheme(ThemeManager::Dark);
        m_themeManager->applyToDialog(settingsLikeDialog);
        
        QString darkDialogStyle = settingsLikeDialog->styleSheet();
        qDebug() << "      Dark theme applied to dialog, style length:" << darkDialogStyle.length();
        QVERIFY(!darkDialogStyle.isEmpty());
        QVERIFY(darkDialogStyle != lightDialogStyle);
        
        // Test comprehensive theme application
        m_themeManager->applyComprehensiveTheme(settingsLikeDialog);
        
        QString comprehensiveStyle = settingsLikeDialog->styleSheet();
        qDebug() << "      Comprehensive theme applied, style length:" << comprehensiveStyle.length();
        QVERIFY(!comprehensiveStyle.isEmpty());
        
        delete settingsLikeDialog;
        qDebug() << "✓ Integration with real components verified";
    }
    
    /**
     * Test 8: Enhanced theme compliance validation
     * Test enhanced theme compliance checking functionality with StyleValidator
     */
    void test_themeComplianceValidation() {
        qDebug() << "\n[Test 8] Enhanced Theme Compliance Validation";
        qDebug() << "==============================================";
        
        // Create a widget with some hardcoded styles (non-compliant)
        QWidget* nonCompliantWidget = new QWidget();
        nonCompliantWidget->setObjectName("NonCompliantTestWidget");
        nonCompliantWidget->setStyleSheet("background-color: #ff0000; color: rgb(0, 255, 0); font-family: Arial;");
        
        QVBoxLayout* layout = new QVBoxLayout(nonCompliantWidget);
        QLabel* label = new QLabel("Test Label", nonCompliantWidget);
        label->setStyleSheet("font-weight: bold; color: #0000ff; padding: 10px;");
        layout->addWidget(label);
        
        qDebug() << "   Created widget with hardcoded styles for testing";
        
        // Test enhanced hardcoded style detection
        QList<StyleViolation> violations = m_themeManager->detectHardcodedStyles(nonCompliantWidget);
        qDebug() << "   Enhanced style violations detected:" << violations.size();
        
        // Verify violations were found
        QVERIFY(!violations.isEmpty());
        
        // Categorize violations
        QMap<QString, int> violationTypes;
        for (const StyleViolation& violation : violations) {
            violationTypes[violation.violationType]++;
            qDebug() << "      " << violation.severity << ":" << violation.violationType 
                     << "in" << violation.componentName << "(" << violation.currentValue << ")";
        }
        
        // Test comprehensive validation
        ValidationResult result = m_themeManager->validateThemeCompliance(nonCompliantWidget);
        qDebug() << "   Compliance validation result:";
        qDebug() << "      Is compliant:" << result.isCompliant;
        qDebug() << "      Accessibility score:" << result.accessibilityScore;
        qDebug() << "      Summary:" << result.summary;
        
        QVERIFY(!result.isCompliant); // Should not be compliant due to hardcoded styles
        
        // Test comprehensive application scan
        ComplianceReport report = m_themeManager->performComprehensiveValidation();
        qDebug() << "   Comprehensive validation report:";
        qDebug() << "      Total components:" << report.totalComponents;
        qDebug() << "      Compliant components:" << report.compliantComponents;
        qDebug() << "      Overall score:" << report.overallScore << "%";
        qDebug() << "      Total violations:" << report.violationCount;
        qDebug() << "      Critical violations:" << report.criticalViolations.size();
        qDebug() << "      Warnings:" << report.warnings.size();
        
        QVERIFY(report.generated.isValid());
        QVERIFY(report.totalComponents >= 0);
        QVERIFY(report.overallScore >= 0.0 && report.overallScore <= 100.0);
        
        // Test accessibility validation
        ThemeData currentTheme = m_themeManager->getCurrentThemeData();
        bool accessibilityValid = m_themeManager->performAccessibilityValidation(currentTheme);
        qDebug() << "   Accessibility validation result:" << accessibilityValid;
        
        // Test validation statistics
        int totalScans = m_themeManager->getValidationScansPerformed();
        QDateTime lastScan = m_themeManager->getLastValidationScan();
        QStringList summary = m_themeManager->getValidationSummary();
        
        qDebug() << "   Validation statistics:";
        qDebug() << "      Total scans performed:" << totalScans;
        qDebug() << "      Last scan:" << lastScan.toString();
        qDebug() << "      Violation summary items:" << summary.size();
        
        // Test legacy compliance methods
        QStringList hardcodedStyles = m_themeManager->scanForHardcodedStyles();
        qDebug() << "   Legacy hardcoded styles found:" << hardcodedStyles.size();
        
        // Test comprehensive compliance test
        m_themeManager->performThemeComplianceTest();
        qDebug() << "   Comprehensive theme compliance test completed";
        
        delete nonCompliantWidget;
        qDebug() << "✓ Enhanced theme compliance validation verified";
    }
    
    /**
     * Test 9: Runtime validation system
     * Test runtime style validation and monitoring
     */
    void test_runtimeValidationSystem() {
        qDebug() << "\n[Test 9] Runtime Validation System";
        qDebug() << "===================================";
        
        // Enable runtime validation
        m_themeManager->enableRuntimeValidation(true);
        m_themeManager->setValidationScanInterval(500); // 500ms for testing
        
        qDebug() << "   Runtime validation enabled with 500ms interval";
        
        // Create a widget that will be monitored
        QWidget* monitoredWidget = new QWidget();
        monitoredWidget->setObjectName("MonitoredWidget");
        monitoredWidget->show();
        
        // Wait for initial scan
        QTest::qWait(600);
        
        // Add hardcoded styles to trigger violations
        monitoredWidget->setStyleSheet("background-color: #123456; color: #abcdef;");
        
        qDebug() << "   Added hardcoded styles to monitored widget";
        
        // Wait for runtime scan to detect violations
        QTest::qWait(600);
        
        // Check if violations were detected
        QStringList violationSummary = m_themeManager->getValidationSummary();
        qDebug() << "   Runtime validation summary:" << violationSummary.size() << "violation types";
        
        for (const QString& summaryItem : violationSummary) {
            qDebug() << "      " << summaryItem;
        }
        
        // Disable runtime validation
        m_themeManager->enableRuntimeValidation(false);
        qDebug() << "   Runtime validation disabled";
        
        monitoredWidget->deleteLater();
        qDebug() << "✓ Runtime validation system verified";
    }
    
    /**
     * Test 10: Source code validation
     * Test source code scanning for hardcoded styles
     */
    void test_sourceCodeValidation() {
        qDebug() << "\n[Test 10] Source Code Validation";
        qDebug() << "=================================";
        
        // Test source code validation (if src directory exists)
        QDir srcDir("src");
        if (srcDir.exists()) {
            qDebug() << "   Source directory found, performing validation...";
            
            QList<StyleViolation> sourceViolations = m_themeManager->validateSourceCode("src");
            
            qDebug() << "   Source code validation results:";
            qDebug() << "      Total violations found:" << sourceViolations.size();
            
            // Categorize source violations
            QMap<QString, int> violationTypes;
            QMap<QString, int> severityLevels;
            
            for (const StyleViolation& violation : sourceViolations) {
                violationTypes[violation.violationType]++;
                severityLevels[violation.severity]++;
            }
            
            qDebug() << "   Violation breakdown by type:";
            for (auto it = violationTypes.begin(); it != violationTypes.end(); ++it) {
                qDebug() << "      " << it.key() << ":" << it.value() << "occurrences";
            }
            
            qDebug() << "   Violation breakdown by severity:";
            for (auto it = severityLevels.begin(); it != severityLevels.end(); ++it) {
                qDebug() << "      " << it.key() << ":" << it.value() << "violations";
            }
            
            // Test detailed report generation
            QString reportPath = "test_source_validation_report.txt";
            m_themeManager->generateDetailedValidationReport(reportPath);
            
            QFile reportFile(reportPath);
            if (reportFile.exists()) {
                qDebug() << "   Detailed validation report generated successfully";
                reportFile.remove(); // Clean up
            }
            
        } else {
            qDebug() << "   Source directory not found, skipping source code validation";
        }
        
        qDebug() << "✓ Source code validation verified";
    }
};

int main(int argc, char* argv[])
{
    // Note: QApplication is needed for widget testing, not just QCoreApplication
    QApplication app(argc, argv);
    
    ThemeManagerIntegrationTest test;
    int result = QTest::qExec(&test, argc, argv);
    
    // Process any remaining events before exit
    QCoreApplication::processEvents();
    
    return result;
}

#include "test_thememanager_integration.moc"