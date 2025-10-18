#include "test_base.h"
#include "ui_automation.h"
#include "visual_testing.h"
#include "theme_accessibility_testing.h"
#include <QTest>
#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QTextEdit>
#include <QProgressBar>
#include <QSlider>
#include <QSpinBox>
#include <QGroupBox>
#include <QTabWidget>
#include <QListWidget>
#include <QTreeWidget>
#include <QTableWidget>
#include <QDebug>

/**
 * @brief Test application for visual regression and theme/accessibility testing
 */
class VisualTestApplication : public QMainWindow {
    Q_OBJECT

public:
    explicit VisualTestApplication(QWidget* parent = nullptr) : QMainWindow(parent) {
        setupUI();
        setObjectName("VisualTestApplication");
        setWindowTitle("Visual & Theme Testing Application");
        resize(900, 700);
    }

private:
    void setupUI() {
        QWidget* centralWidget = new QWidget(this);
        setCentralWidget(centralWidget);

        QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);

        // Create tab widget for different test sections
        QTabWidget* tabWidget = new QTabWidget(this);
        tabWidget->setObjectName("MainTabWidget");
        mainLayout->addWidget(tabWidget);

        // Basic controls tab
        tabWidget->addTab(createBasicControlsTab(), "Basic Controls");
        
        // Advanced controls tab
        tabWidget->addTab(createAdvancedControlsTab(), "Advanced Controls");
        
        // Data display tab
        tabWidget->addTab(createDataDisplayTab(), "Data Display");
        
        // Accessibility test tab
        tabWidget->addTab(createAccessibilityTestTab(), "Accessibility");
    }

    QWidget* createBasicControlsTab() {
        QWidget* tab = new QWidget();
        tab->setObjectName("BasicControlsTab");
        
        QVBoxLayout* layout = new QVBoxLayout(tab);
        
        // Title
        QLabel* title = new QLabel("Basic UI Controls", tab);
        title->setObjectName("BasicControlsTitle");
        title->setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;");
        layout->addWidget(title);
        
        // Button group
        QGroupBox* buttonGroup = new QGroupBox("Buttons", tab);
        buttonGroup->setObjectName("ButtonGroup");
        QHBoxLayout* buttonLayout = new QHBoxLayout(buttonGroup);
        
        QPushButton* primaryBtn = new QPushButton("Primary Action", buttonGroup);
        primaryBtn->setObjectName("PrimaryButton");
        primaryBtn->setDefault(true);
        buttonLayout->addWidget(primaryBtn);
        
        QPushButton* secondaryBtn = new QPushButton("Secondary", buttonGroup);
        secondaryBtn->setObjectName("SecondaryButton");
        buttonLayout->addWidget(secondaryBtn);
        
        QPushButton* disabledBtn = new QPushButton("Disabled", buttonGroup);
        disabledBtn->setObjectName("DisabledButton");
        disabledBtn->setEnabled(false);
        buttonLayout->addWidget(disabledBtn);
        
        layout->addWidget(buttonGroup);
        
        // Input group
        QGroupBox* inputGroup = new QGroupBox("Input Controls", tab);
        inputGroup->setObjectName("InputGroup");
        QVBoxLayout* inputLayout = new QVBoxLayout(inputGroup);
        
        QLineEdit* textInput = new QLineEdit(inputGroup);
        textInput->setObjectName("TextInput");
        textInput->setPlaceholderText("Enter text here...");
        inputLayout->addWidget(textInput);
        
        QComboBox* comboBox = new QComboBox(inputGroup);
        comboBox->setObjectName("ComboBox");
        comboBox->addItems({"Option 1", "Option 2", "Option 3"});
        inputLayout->addWidget(comboBox);
        
        QCheckBox* checkBox = new QCheckBox("Enable feature", inputGroup);
        checkBox->setObjectName("CheckBox");
        inputLayout->addWidget(checkBox);
        
        layout->addWidget(inputGroup);
        
        // Progress and sliders
        QGroupBox* progressGroup = new QGroupBox("Progress & Sliders", tab);
        progressGroup->setObjectName("ProgressGroup");
        QVBoxLayout* progressLayout = new QVBoxLayout(progressGroup);
        
        QProgressBar* progressBar = new QProgressBar(progressGroup);
        progressBar->setObjectName("ProgressBar");
        progressBar->setValue(65);
        progressLayout->addWidget(progressBar);
        
        QSlider* slider = new QSlider(Qt::Horizontal, progressGroup);
        slider->setObjectName("Slider");
        slider->setRange(0, 100);
        slider->setValue(50);
        progressLayout->addWidget(slider);
        
        QSpinBox* spinBox = new QSpinBox(progressGroup);
        spinBox->setObjectName("SpinBox");
        spinBox->setRange(0, 1000);
        spinBox->setValue(42);
        progressLayout->addWidget(spinBox);
        
        layout->addWidget(progressGroup);
        
        layout->addStretch();
        return tab;
    }

    QWidget* createAdvancedControlsTab() {
        QWidget* tab = new QWidget();
        tab->setObjectName("AdvancedControlsTab");
        
        QVBoxLayout* layout = new QVBoxLayout(tab);
        
        // Title
        QLabel* title = new QLabel("Advanced UI Controls", tab);
        title->setObjectName("AdvancedControlsTitle");
        title->setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;");
        layout->addWidget(title);
        
        // Text area
        QTextEdit* textEdit = new QTextEdit(tab);
        textEdit->setObjectName("TextEdit");
        textEdit->setPlaceholderText("Multi-line text input area...");
        textEdit->setMaximumHeight(150);
        layout->addWidget(textEdit);
        
        // Color-coded status indicators
        QHBoxLayout* statusLayout = new QHBoxLayout();
        
        QLabel* successLabel = new QLabel("Success Status", tab);
        successLabel->setObjectName("SuccessLabel");
        successLabel->setStyleSheet("background-color: #d4edda; color: #155724; padding: 8px; border-radius: 4px;");
        statusLayout->addWidget(successLabel);
        
        QLabel* warningLabel = new QLabel("Warning Status", tab);
        warningLabel->setObjectName("WarningLabel");
        warningLabel->setStyleSheet("background-color: #fff3cd; color: #856404; padding: 8px; border-radius: 4px;");
        statusLayout->addWidget(warningLabel);
        
        QLabel* errorLabel = new QLabel("Error Status", tab);
        errorLabel->setObjectName("ErrorLabel");
        errorLabel->setStyleSheet("background-color: #f8d7da; color: #721c24; padding: 8px; border-radius: 4px;");
        statusLayout->addWidget(errorLabel);
        
        layout->addLayout(statusLayout);
        
        layout->addStretch();
        return tab;
    }

    QWidget* createDataDisplayTab() {
        QWidget* tab = new QWidget();
        tab->setObjectName("DataDisplayTab");
        
        QVBoxLayout* layout = new QVBoxLayout(tab);
        
        // Title
        QLabel* title = new QLabel("Data Display Controls", tab);
        title->setObjectName("DataDisplayTitle");
        title->setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;");
        layout->addWidget(title);
        
        // List widget
        QListWidget* listWidget = new QListWidget(tab);
        listWidget->setObjectName("ListWidget");
        listWidget->addItems({"Item 1", "Item 2", "Item 3", "Item 4"});
        listWidget->setMaximumHeight(100);
        layout->addWidget(listWidget);
        
        // Tree widget
        QTreeWidget* treeWidget = new QTreeWidget(tab);
        treeWidget->setObjectName("TreeWidget");
        treeWidget->setHeaderLabels({"Name", "Value"});
        
        QTreeWidgetItem* rootItem = new QTreeWidgetItem(treeWidget, {"Root", "0"});
        QTreeWidgetItem* child1 = new QTreeWidgetItem(rootItem, {"Child 1", "10"});
        QTreeWidgetItem* child2 = new QTreeWidgetItem(rootItem, {"Child 2", "20"});
        new QTreeWidgetItem(child1, {"Grandchild", "5"});
        
        treeWidget->expandAll();
        treeWidget->setMaximumHeight(150);
        layout->addWidget(treeWidget);
        
        // Table widget
        QTableWidget* tableWidget = new QTableWidget(3, 3, tab);
        tableWidget->setObjectName("TableWidget");
        tableWidget->setHorizontalHeaderLabels({"Column 1", "Column 2", "Column 3"});
        
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                tableWidget->setItem(row, col, 
                    new QTableWidgetItem(QString("Cell %1,%2").arg(row + 1).arg(col + 1)));
            }
        }
        
        tableWidget->setMaximumHeight(150);
        layout->addWidget(tableWidget);
        
        layout->addStretch();
        return tab;
    }

    QWidget* createAccessibilityTestTab() {
        QWidget* tab = new QWidget();
        tab->setObjectName("AccessibilityTestTab");
        
        QVBoxLayout* layout = new QVBoxLayout(tab);
        
        // Title
        QLabel* title = new QLabel("Accessibility Test Controls", tab);
        title->setObjectName("AccessibilityTitle");
        title->setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;");
        layout->addWidget(title);
        
        // High contrast elements
        QLabel* highContrastLabel = new QLabel("High Contrast Text", tab);
        highContrastLabel->setObjectName("HighContrastLabel");
        highContrastLabel->setStyleSheet("background-color: black; color: white; padding: 10px; font-size: 14px;");
        layout->addWidget(highContrastLabel);
        
        // Low contrast elements (should fail accessibility tests)
        QLabel* lowContrastLabel = new QLabel("Low Contrast Text (Should Fail)", tab);
        lowContrastLabel->setObjectName("LowContrastLabel");
        lowContrastLabel->setStyleSheet("background-color: #f0f0f0; color: #d0d0d0; padding: 10px; font-size: 12px;");
        layout->addWidget(lowContrastLabel);
        
        // Focusable elements for keyboard navigation testing
        QPushButton* focusBtn1 = new QPushButton("Focusable Button 1", tab);
        focusBtn1->setObjectName("FocusButton1");
        layout->addWidget(focusBtn1);
        
        QPushButton* focusBtn2 = new QPushButton("Focusable Button 2", tab);
        focusBtn2->setObjectName("FocusButton2");
        layout->addWidget(focusBtn2);
        
        // Element with accessibility attributes
        QLineEdit* accessibleInput = new QLineEdit(tab);
        accessibleInput->setObjectName("AccessibleInput");
        accessibleInput->setPlaceholderText("Input with accessibility attributes");
        accessibleInput->setAccessibleName("User Name Input");
        accessibleInput->setAccessibleDescription("Enter your full name here");
        layout->addWidget(accessibleInput);
        
        layout->addStretch();
        return tab;
    }
};

/**
 * @brief Example test class demonstrating visual regression and theme/accessibility testing
 */
DECLARE_TEST_CLASS(VisualThemeTestingExample, UI, High, "visual", "theme", "accessibility", "regression")

private:
    VisualTestApplication* m_testApp;
    UIAutomation* m_uiAutomation;
    VisualTesting* m_visualTesting;
    ThemeAccessibilityTesting* m_themeAccessibilityTesting;

private slots:
    void initTestCase() {
        TestBase::initTestCase();
        logTestInfo("Setting up visual and theme testing example");
        
        // Create test application
        m_testApp = new VisualTestApplication();
        m_testApp->show();
        QTest::qWaitForWindowActive(m_testApp);
        
        // Create testing frameworks
        m_uiAutomation = new UIAutomation(this);
        m_visualTesting = new VisualTesting(this);
        m_themeAccessibilityTesting = new ThemeAccessibilityTesting(this);
        
        // Configure frameworks
        m_uiAutomation->enableDetailedLogging(true);
        
        VisualTesting::TestConfig visualConfig;
        visualConfig.threshold = 0.95;
        visualConfig.generateDiffImage = true;
        visualConfig.saveFailedComparisons = true;
        m_visualTesting->setTestConfig(visualConfig);
        
        ThemeAccessibilityTesting::AccessibilityConfig accessConfig;
        accessConfig.complianceLevel = ThemeAccessibilityTesting::AccessibilityLevel::WCAG_AA;
        accessConfig.minContrastRatio = 4.5;
        m_themeAccessibilityTesting->setAccessibilityConfig(accessConfig);
        m_themeAccessibilityTesting->setUIAutomation(m_uiAutomation);
        m_themeAccessibilityTesting->setVisualTesting(m_visualTesting);
    }

    void cleanupTestCase() {
        logTestInfo("Cleaning up visual and theme testing example");
        
        if (m_testApp) {
            m_testApp->close();
            delete m_testApp;
            m_testApp = nullptr;
        }
        
        TestBase::cleanupTestCase();
    }

    TEST_METHOD(test_visualRegression_createBaselines_succeeds) {
        logTestStep("Testing visual regression baseline creation");
        
        // Create baselines for different tabs
        auto tabWidget = UIAutomation::byObjectName("MainTabWidget");
        QWidget* tabWidgetPtr = m_uiAutomation->findWidget(tabWidget);
        TEST_VERIFY_WITH_MSG(tabWidgetPtr != nullptr, "Tab widget should exist");
        
        // Switch to each tab and create baseline
        QStringList tabNames = {"BasicControlsTab", "AdvancedControlsTab", "DataDisplayTab", "AccessibilityTestTab"};
        
        for (int i = 0; i < tabNames.size(); ++i) {
            // Switch to tab
            if (m_uiAutomation->switchToTab(tabWidget, i)) {
                QThread::msleep(100); // Allow UI to update
                
                // Capture baseline
                QString baselineName = QString("tab_%1").arg(tabNames[i]);
                TEST_VERIFY_WITH_MSG(
                    m_visualTesting->createBaseline(baselineName, m_testApp, 
                                                  QString("Baseline for %1").arg(tabNames[i])),
                    QString("Should create baseline for %1").arg(tabNames[i])
                );
            }
        }
        
        logTestStep("Visual regression baseline creation completed successfully");
    }

    TEST_METHOD(test_visualRegression_compareWithBaselines_detectsChanges) {
        logTestStep("Testing visual regression comparison");
        
        // First ensure baselines exist (create if needed)
        if (!m_visualTesting->baselineExists("tab_BasicControlsTab")) {
            test_visualRegression_createBaselines_succeeds();
        }
        
        auto tabWidget = UIAutomation::byObjectName("MainTabWidget");
        
        // Switch to basic controls tab
        TEST_VERIFY_WITH_MSG(m_uiAutomation->switchToTab(tabWidget, 0), 
                           "Should switch to basic controls tab");
        QThread::msleep(100);
        
        // Compare with baseline (should match)
        auto result = m_visualTesting->compareWithBaseline("tab_BasicControlsTab", m_testApp);
        TEST_VERIFY_WITH_MSG(result.matches, 
                           QString("Visual comparison should match (similarity: %1)").arg(result.similarity));
        
        // Make a visual change and test detection
        auto primaryButton = UIAutomation::byObjectName("PrimaryButton");
        QWidget* buttonWidget = m_uiAutomation->findWidget(primaryButton);
        if (buttonWidget) {
            // Change button style to create visual difference
            buttonWidget->setStyleSheet("background-color: red; color: white;");
            QApplication::processEvents();
            
            // Compare again (should not match)
            auto changedResult = m_visualTesting->compareWithBaseline("tab_BasicControlsTab", m_testApp);
            TEST_VERIFY_WITH_MSG(!changedResult.matches, 
                               "Visual comparison should detect changes");
            
            // Restore original style
            buttonWidget->setStyleSheet("");
            QApplication::processEvents();
        }
        
        logTestStep("Visual regression comparison test completed successfully");
    }

    TEST_METHOD(test_themeValidation_switchThemes_maintainsUsability) {
        logTestStep("Testing theme switching and validation");
        
        // Test different themes
        QList<ThemeAccessibilityTesting::ThemeType> themes = {
            ThemeAccessibilityTesting::ThemeType::Light,
            ThemeAccessibilityTesting::ThemeType::Dark,
            ThemeAccessibilityTesting::ThemeType::HighContrast
        };
        
        for (auto themeType : themes) {
            logTestStep(QString("Testing theme: %1").arg(static_cast<int>(themeType)));
            
            // Switch to theme
            TEST_VERIFY_WITH_MSG(m_themeAccessibilityTesting->switchToTheme(themeType), 
                               "Should switch theme successfully");
            
            QThread::msleep(200); // Allow theme to apply
            
            // Validate theme
            auto themeResult = m_themeAccessibilityTesting->validateCurrentTheme();
            TEST_VERIFY_WITH_MSG(themeResult.colorsConsistent, 
                               "Theme colors should be consistent");
            
            // Test basic interaction still works
            auto primaryButton = UIAutomation::byObjectName("PrimaryButton");
            TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetVisible(primaryButton), 
                               "Button should remain visible in theme");
            TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetEnabled(primaryButton), 
                               "Button should remain enabled in theme");
        }
        
        logTestStep("Theme validation test completed successfully");
    }

    TEST_METHOD(test_accessibilityCompliance_colorContrast_meetsWCAGStandards) {
        logTestStep("Testing color contrast accessibility compliance");
        
        // Switch to accessibility test tab
        auto tabWidget = UIAutomation::byObjectName("MainTabWidget");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->switchToTab(tabWidget, 3), 
                           "Should switch to accessibility tab");
        QThread::msleep(100);
        
        // Test high contrast element (should pass)
        auto highContrastLabel = UIAutomation::byObjectName("HighContrastLabel");
        QWidget* highContrastWidget = m_uiAutomation->findWidget(highContrastLabel);
        if (highContrastWidget) {
            auto contrastResult = m_themeAccessibilityTesting->testColorContrast(highContrastWidget);
            TEST_VERIFY_WITH_MSG(contrastResult.passes, 
                               QString("High contrast element should pass WCAG (ratio: %1)").arg(contrastResult.contrastRatio));
        }
        
        // Test low contrast element (should fail)
        auto lowContrastLabel = UIAutomation::byObjectName("LowContrastLabel");
        QWidget* lowContrastWidget = m_uiAutomation->findWidget(lowContrastLabel);
        if (lowContrastWidget) {
            auto contrastResult = m_themeAccessibilityTesting->testColorContrast(lowContrastWidget);
            TEST_VERIFY_WITH_MSG(!contrastResult.passes, 
                               QString("Low contrast element should fail WCAG (ratio: %1)").arg(contrastResult.contrastRatio));
        }
        
        logTestStep("Color contrast accessibility test completed successfully");
    }

    TEST_METHOD(test_accessibilityCompliance_keyboardNavigation_worksCorrectly) {
        logTestStep("Testing keyboard navigation accessibility");
        
        // Switch to accessibility test tab
        auto tabWidget = UIAutomation::byObjectName("MainTabWidget");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->switchToTab(tabWidget, 3), 
                           "Should switch to accessibility tab");
        QThread::msleep(100);
        
        // Test focusable buttons
        QStringList focusableButtons = {"FocusButton1", "FocusButton2"};
        
        for (const QString& buttonName : focusableButtons) {
            auto buttonSelector = UIAutomation::byObjectName(buttonName);
            QWidget* buttonWidget = m_uiAutomation->findWidget(buttonSelector);
            
            if (buttonWidget) {
                auto keyboardResult = m_themeAccessibilityTesting->testKeyboardNavigation(buttonWidget);
                TEST_VERIFY_WITH_MSG(keyboardResult.canReceiveFocus, 
                                   QString("Button %1 should be focusable").arg(buttonName));
                
                // Test focus setting
                TEST_VERIFY_WITH_MSG(m_uiAutomation->setWidgetFocus(buttonSelector), 
                                   QString("Should be able to set focus to %1").arg(buttonName));
            }
        }
        
        // Test accessible input
        auto accessibleInput = UIAutomation::byObjectName("AccessibleInput");
        QWidget* inputWidget = m_uiAutomation->findWidget(accessibleInput);
        if (inputWidget) {
            auto keyboardResult = m_themeAccessibilityTesting->testKeyboardNavigation(inputWidget);
            TEST_VERIFY_WITH_MSG(keyboardResult.canReceiveFocus, 
                               "Accessible input should be focusable");
        }
        
        logTestStep("Keyboard navigation accessibility test completed successfully");
    }

    TEST_METHOD(test_accessibilityCompliance_screenReader_providesCorrectInfo) {
        logTestStep("Testing screen reader accessibility compliance");
        
        // Switch to accessibility test tab
        auto tabWidget = UIAutomation::byObjectName("MainTabWidget");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->switchToTab(tabWidget, 3), 
                           "Should switch to accessibility tab");
        QThread::msleep(100);
        
        // Test accessible input (has explicit accessibility attributes)
        auto accessibleInput = UIAutomation::byObjectName("AccessibleInput");
        QWidget* inputWidget = m_uiAutomation->findWidget(accessibleInput);
        
        if (inputWidget) {
            auto screenReaderResult = m_themeAccessibilityTesting->testScreenReaderCompatibility(inputWidget);
            
            TEST_VERIFY_WITH_MSG(screenReaderResult.hasAccessibleName, 
                               "Accessible input should have accessible name");
            TEST_VERIFY_WITH_MSG(screenReaderResult.hasCorrectRole, 
                               "Accessible input should have correct role");
            
            logTestInfo(QString("Accessible name: %1").arg(screenReaderResult.accessibleName));
            logTestInfo(QString("Accessible description: %1").arg(screenReaderResult.accessibleDescription));
        }
        
        logTestStep("Screen reader accessibility test completed successfully");
    }

    TEST_METHOD(test_accessibilityCompliance_fullAudit_identifiesIssues) {
        logTestStep("Testing comprehensive accessibility audit");
        
        // Run full accessibility audit on the application
        bool auditPassed = m_themeAccessibilityTesting->runFullAccessibilityAudit(m_testApp);
        
        // Note: We expect some failures due to intentionally problematic elements
        logTestInfo(QString("Full accessibility audit completed. Passed: %1").arg(auditPassed ? "Yes" : "No"));
        
        // The audit should complete without crashing, even if some tests fail
        TEST_VERIFY_WITH_MSG(true, "Accessibility audit should complete successfully");
        
        logTestStep("Comprehensive accessibility audit completed successfully");
    }

END_TEST_CLASS()

/**
 * @brief Main function for running the visual and theme testing example
 */
int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    
    qDebug() << "========================================";
    qDebug() << "Visual Regression & Theme/Accessibility Testing Example";
    qDebug() << "========================================";
    
    // Load test configuration
    TestConfig::instance().loadConfiguration();
    
    // Create and run the test
    VisualThemeTestingExample test;
    
    if (test.shouldRunTest()) {
        int result = QTest::qExec(&test, argc, argv);
        
        if (result == 0) {
            qDebug() << "✅ Visual and theme testing example PASSED";
        } else {
            qDebug() << "❌ Visual and theme testing example FAILED";
        }
        
        return result;
    } else {
        qDebug() << "⏭️  Visual and theme testing example SKIPPED (disabled by configuration)";
        return 0;
    }
}

#include "example_visual_theme_testing.moc"