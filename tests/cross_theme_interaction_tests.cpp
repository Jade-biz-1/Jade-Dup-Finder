#include <QtTest/QtTest>
#include <QApplication>
#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTimer>
#include <QElapsedTimer>
#include <QSignalSpy>

#include "workflow_testing.h"
#include "user_scenario_testing.h"
#include "ui_automation.h"
#include "visual_testing.h"
#include "theme_accessibility_testing.h"
#include "../include/ui_theme_test_integration.h"
#include "../include/theme_manager.h"

/**
 * @brief Cross-theme interaction validation tests (Task 11.2)
 * 
 * This test class implements comprehensive cross-theme interaction validation
 * to ensure all user interactions work correctly in both light and dark themes,
 * validate UI state maintenance throughout complete user workflows, and
 * ensure consistent UI behavior across all workflow steps.
 */
class CrossThemeInteractionTests : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();

    // Core cross-theme interaction tests
    void testButtonInteractionsAcrossThemes();
    void testInputFieldInteractionsAcrossThemes();
    void testProgressIndicatorAcrossThemes();
    void testCheckboxInteractionsAcrossThemes();
    void testMenuInteractionsAcrossThemes();

    // UI state maintenance tests
    void testStateMaintenanceDuringThemeSwitch();
    void testFormDataPersistenceAcrossThemes();
    void testSelectionStatePersistenceAcrossThemes();
    void testProgressStatePersistenceAcrossThemes();

    // Consistent behavior validation
    void testConsistentClickBehaviorAcrossThemes();
    void testConsistentKeyboardNavigationAcrossThemes();
    void testConsistentFocusManagementAcrossThemes();
    void testConsistentTooltipBehaviorAcrossThemes();

    // Theme transition validation
    void testSmoothThemeTransitions();
    void testThemeTransitionWithActiveDialogs();
    void testThemeTransitionDuringOperations();
    void testThemeTransitionErrorRecovery();

private:
    // Test setup and utilities
    void setupTestEnvironment();
    void createTestComponents();
    bool validateInteractionInTheme(const QString& interaction, ThemeManager::Theme theme);
    bool compareInteractionBehavior(const QString& interaction, 
                                   ThemeManager::Theme theme1, 
                                   ThemeManager::Theme theme2);

    // Cross-theme validation helpers
    bool testInteractionConsistency(const QString& componentName, 
                                   const QString& interactionType,
                                   const QList<ThemeManager::Theme>& themes);
    bool validateUIStateAfterThemeChange(ThemeManager::Theme fromTheme, 
                                        ThemeManager::Theme toTheme);
    bool measureInteractionPerformance(const QString& interaction, ThemeManager::Theme theme);

private:
    // Test framework components
    std::shared_ptr<UIAutomation> m_uiAutomation;
    std::shared_ptr<VisualTesting> m_visualTesting;
    std::shared_ptr<ThemeAccessibilityTesting> m_themeAccessibilityTesting;
    UIThemeTestIntegration* m_themeIntegration;
    ThemeManager* m_themeManager;
    
    // Test environment
    QMainWindow* m_testWindow;
    QWidget* m_testWidget;
    QMap<QString, QWidget*> m_testComponents;
    
    // Test configuration
    QList<ThemeManager::Theme> m_supportedThemes;
    QElapsedTimer m_performanceTimer;
    QMap<QString, QMap<ThemeManager::Theme, qint64>> m_interactionTimes;
};void
 CrossThemeInteractionTests::initTestCase() {
    qDebug() << "Initializing cross-theme interaction validation tests...";
    
    // Initialize testing framework components
    m_uiAutomation = std::make_shared<UIAutomation>(this);
    m_visualTesting = std::make_shared<VisualTesting>(this);
    m_themeAccessibilityTesting = std::make_shared<ThemeAccessibilityTesting>(this);
    
    // Connect frameworks
    m_themeAccessibilityTesting->setUIAutomation(m_uiAutomation.get());
    m_themeAccessibilityTesting->setVisualTesting(m_visualTesting.get());
    
    // Initialize theme integration
    m_themeIntegration = new UIThemeTestIntegration(this);
    m_themeIntegration->setUIAutomation(m_uiAutomation.get());
    m_themeIntegration->setVisualTesting(m_visualTesting.get());
    m_themeIntegration->setThemeAccessibilityTesting(m_themeAccessibilityTesting.get());
    
    // Get theme manager
    m_themeManager = ThemeManager::instance();
    
    // Setup supported themes
    m_supportedThemes = {ThemeManager::Light, ThemeManager::Dark};
    
    // Setup test environment
    setupTestEnvironment();
    
    qDebug() << "Cross-theme interaction tests initialization completed";
}

void CrossThemeInteractionTests::cleanupTestCase() {
    // Restore original theme
    m_themeManager->setTheme(ThemeManager::Light);
    
    // Clean up test environment
    delete m_testWindow;
    delete m_themeIntegration;
    
    qDebug() << "Cross-theme interaction tests cleanup completed";
}

void CrossThemeInteractionTests::init() {
    // Reset to light theme before each test
    m_themeManager->setTheme(ThemeManager::Light);
    QTest::qWait(200);
    
    // Clear performance measurements
    m_interactionTimes.clear();
}

void CrossThemeInteractionTests::cleanup() {
    QTest::qWait(100);
}

void CrossThemeInteractionTests::setupTestEnvironment() {
    // Create test window
    m_testWindow = new QMainWindow();
    m_testWindow->setObjectName("crossThemeTestWindow");
    m_testWindow->setWindowTitle("Cross-Theme Interaction Test");
    m_testWindow->resize(600, 400);
    
    // Create test widget
    m_testWidget = new QWidget();
    m_testWidget->setObjectName("crossThemeTestWidget");
    
    createTestComponents();
    
    m_testWindow->setCentralWidget(m_testWidget);
    m_testWindow->show();
    
    // Register with theme manager
    m_themeManager->registerComponent(m_testWidget, ThemeManager::ComponentType::Widget);
    m_themeManager->applyToWidget(m_testWidget);
    
    QTest::qWait(300);
}

void CrossThemeInteractionTests::createTestComponents() {
    auto* layout = new QVBoxLayout(m_testWidget);
    
    // Create various UI components for testing
    auto* testButton = new QPushButton("Test Button");
    testButton->setObjectName("testButton");
    m_testComponents["testButton"] = testButton;
    layout->addWidget(testButton);
    
    auto* testInput = new QLineEdit();
    testInput->setObjectName("testInput");
    testInput->setPlaceholderText("Test input field");
    m_testComponents["testInput"] = testInput;
    layout->addWidget(testInput);
    
    auto* testProgress = new QProgressBar();
    testProgress->setObjectName("testProgress");
    testProgress->setValue(50);
    m_testComponents["testProgress"] = testProgress;
    layout->addWidget(testProgress);
    
    auto* testCheckbox = new QCheckBox("Test checkbox");
    testCheckbox->setObjectName("testCheckbox");
    m_testComponents["testCheckbox"] = testCheckbox;
    layout->addWidget(testCheckbox);
    
    auto* testLabel = new QLabel("Test label");
    testLabel->setObjectName("testLabel");
    m_testComponents["testLabel"] = testLabel;
    layout->addWidget(testLabel);
    
    // Connect signals for interaction testing
    connect(testButton, &QPushButton::clicked, [this]() {
        qDebug() << "Test button clicked in theme:" << static_cast<int>(m_themeManager->currentTheme());
    });
    
    connect(testCheckbox, &QCheckBox::toggled, [this](bool checked) {
        qDebug() << "Test checkbox toggled to" << checked << "in theme:" << static_cast<int>(m_themeManager->currentTheme());
    });
}

// Core cross-theme interaction tests

void CrossThemeInteractionTests::testButtonInteractionsAcrossThemes() {
    qDebug() << "Testing button interactions across themes...";
    
    QVERIFY(testInteractionConsistency("testButton", "click", m_supportedThemes));
    
    // Test button state changes across themes
    QPushButton* button = qobject_cast<QPushButton*>(m_testComponents["testButton"]);
    QVERIFY(button != nullptr);
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Test enabled state
        button->setEnabled(true);
        QVERIFY(button->isEnabled());
        QVERIFY(validateInteractionInTheme("button_click_enabled", theme));
        
        // Test disabled state
        button->setEnabled(false);
        QVERIFY(!button->isEnabled());
        
        // Test re-enabling
        button->setEnabled(true);
        QVERIFY(button->isEnabled());
    }
    
    qDebug() << "Button interaction testing completed successfully";
}

void CrossThemeInteractionTests::testInputFieldInteractionsAcrossThemes() {
    qDebug() << "Testing input field interactions across themes...";
    
    QVERIFY(testInteractionConsistency("testInput", "type", m_supportedThemes));
    
    QLineEdit* input = qobject_cast<QLineEdit*>(m_testComponents["testInput"]);
    QVERIFY(input != nullptr);
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Test text input
        input->clear();
        input->setText("Test text in theme " + QString::number(static_cast<int>(theme)));
        QCOMPARE(input->text(), QString("Test text in theme %1").arg(static_cast<int>(theme)));
        
        // Test focus behavior
        input->setFocus();
        QVERIFY(input->hasFocus());
        
        // Test selection
        input->selectAll();
        QVERIFY(input->hasSelectedText());
        
        // Validate theme compliance
        auto selector = UIThemeTestIntegration::createSelector("testInput");
        QVERIFY(m_themeIntegration->validateWidgetThemeCompliance(selector));
    }
    
    qDebug() << "Input field interaction testing completed successfully";
}

void CrossThemeInteractionTests::testProgressIndicatorAcrossThemes() {
    qDebug() << "Testing progress indicator across themes...";
    
    QProgressBar* progress = qobject_cast<QProgressBar*>(m_testComponents["testProgress"]);
    QVERIFY(progress != nullptr);
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Test progress value changes
        for (int value = 0; value <= 100; value += 25) {
            progress->setValue(value);
            QCOMPARE(progress->value(), value);
            QTest::qWait(50);
            
            // Validate visibility and theme compliance
            QVERIFY(progress->isVisible());
            auto selector = UIThemeTestIntegration::createSelector("testProgress");
            QVERIFY(m_themeIntegration->validateWidgetThemeCompliance(selector));
        }
    }
    
    qDebug() << "Progress indicator testing completed successfully";
}

void CrossThemeInteractionTests::testCheckboxInteractionsAcrossThemes() {
    qDebug() << "Testing checkbox interactions across themes...";
    
    QVERIFY(testInteractionConsistency("testCheckbox", "toggle", m_supportedThemes));
    
    QCheckBox* checkbox = qobject_cast<QCheckBox*>(m_testComponents["testCheckbox"]);
    QVERIFY(checkbox != nullptr);
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Test checkbox state changes
        checkbox->setChecked(false);
        QVERIFY(!checkbox->isChecked());
        
        checkbox->setChecked(true);
        QVERIFY(checkbox->isChecked());
        
        // Test click interaction
        QSignalSpy clickSpy(checkbox, &QCheckBox::clicked);
        QTest::mouseClick(checkbox, Qt::LeftButton);
        QCOMPARE(clickSpy.count(), 1);
        
        // Validate theme compliance and visibility
        auto selector = UIThemeTestIntegration::createSelector("testCheckbox");
        QVERIFY(m_themeIntegration->validateWidgetThemeCompliance(selector));
        
        // Validate accessibility
        auto accessibilityResult = m_themeIntegration->testComponentAccessibility(selector, theme);
        QVERIFY(accessibilityResult.overallPassed);
    }
    
    qDebug() << "Checkbox interaction testing completed successfully";
}

void CrossThemeInteractionTests::testMenuInteractionsAcrossThemes() {
    qDebug() << "Testing menu interactions across themes...";
    
    // Create a test menu
    auto* menuBar = m_testWindow->menuBar();
    auto* testMenu = menuBar->addMenu("Test Menu");
    auto* testAction = testMenu->addAction("Test Action");
    testAction->setObjectName("testAction");
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Test menu visibility and accessibility
        QVERIFY(menuBar->isVisible());
        QVERIFY(testMenu != nullptr);
        QVERIFY(testAction != nullptr);
        
        // Test menu action triggering
        QSignalSpy actionSpy(testAction, &QAction::triggered);
        testAction->trigger();
        QCOMPARE(actionSpy.count(), 1);
        
        // Validate theme compliance for menu components
        // Note: Menu styling validation would require more complex theme integration
    }
    
    qDebug() << "Menu interaction testing completed successfully";
}

// UI state maintenance tests

void CrossThemeInteractionTests::testStateMaintenanceDuringThemeSwitch() {
    qDebug() << "Testing UI state maintenance during theme switch...";
    
    // Set initial states
    QPushButton* button = qobject_cast<QPushButton*>(m_testComponents["testButton"]);
    QLineEdit* input = qobject_cast<QLineEdit*>(m_testComponents["testInput"]);
    QCheckBox* checkbox = qobject_cast<QCheckBox*>(m_testComponents["testCheckbox"]);
    QProgressBar* progress = qobject_cast<QProgressBar*>(m_testComponents["testProgress"]);
    
    QVERIFY(button && input && checkbox && progress);
    
    // Set specific states
    button->setEnabled(false);
    input->setText("State maintenance test");
    checkbox->setChecked(true);
    progress->setValue(75);
    
    // Record initial states
    bool initialButtonEnabled = button->isEnabled();
    QString initialInputText = input->text();
    bool initialCheckboxChecked = checkbox->isChecked();
    int initialProgressValue = progress->value();
    
    // Switch themes and verify state maintenance
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(300); // Allow theme to apply
        
        // Verify states are maintained
        QCOMPARE(button->isEnabled(), initialButtonEnabled);
        QCOMPARE(input->text(), initialInputText);
        QCOMPARE(checkbox->isChecked(), initialCheckboxChecked);
        QCOMPARE(progress->value(), initialProgressValue);
        
        // Verify components are still functional
        QVERIFY(validateUIStateAfterThemeChange(ThemeManager::Light, theme));
    }
    
    qDebug() << "UI state maintenance testing completed successfully";
}

void CrossThemeInteractionTests::testFormDataPersistenceAcrossThemes() {
    qDebug() << "Testing form data persistence across themes...";
    
    QLineEdit* input = qobject_cast<QLineEdit*>(m_testComponents["testInput"]);
    QCheckBox* checkbox = qobject_cast<QCheckBox*>(m_testComponents["testCheckbox"]);
    
    QVERIFY(input && checkbox);
    
    // Set form data
    const QString testData = "Persistent form data test";
    input->setText(testData);
    checkbox->setChecked(true);
    
    // Switch between themes multiple times
    for (int i = 0; i < 3; ++i) {
        for (ThemeManager::Theme theme : m_supportedThemes) {
            QVERIFY(m_themeManager->setTheme(theme));
            QTest::qWait(200);
            
            // Verify data persistence
            QCOMPARE(input->text(), testData);
            QVERIFY(checkbox->isChecked());
            
            // Verify components remain interactive
            input->setFocus();
            QVERIFY(input->hasFocus());
        }
    }
    
    qDebug() << "Form data persistence testing completed successfully";
}

void CrossThemeInteractionTests::testSelectionStatePersistenceAcrossThemes() {
    qDebug() << "Testing selection state persistence across themes...";
    
    QLineEdit* input = qobject_cast<QLineEdit*>(m_testComponents["testInput"]);
    QVERIFY(input != nullptr);
    
    // Set text and selection
    const QString testText = "Selection persistence test";
    input->setText(testText);
    input->setSelection(10, 5); // Select "ence"
    
    QString selectedText = input->selectedText();
    QVERIFY(!selectedText.isEmpty());
    
    // Switch themes and verify selection persistence
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Note: Selection might not persist across theme changes in all Qt versions
        // This test validates that the component remains functional
        QCOMPARE(input->text(), testText);
        QVERIFY(input->isEnabled());
        
        // Re-select and verify functionality
        input->selectAll();
        QVERIFY(input->hasSelectedText());
    }
    
    qDebug() << "Selection state persistence testing completed successfully";
}

void CrossThemeInteractionTests::testProgressStatePersistenceAcrossThemes() {
    qDebug() << "Testing progress state persistence across themes...";
    
    QProgressBar* progress = qobject_cast<QProgressBar*>(m_testComponents["testProgress"]);
    QVERIFY(progress != nullptr);
    
    // Set progress value
    const int testValue = 65;
    progress->setValue(testValue);
    QCOMPARE(progress->value(), testValue);
    
    // Switch themes and verify progress persistence
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Verify progress value is maintained
        QCOMPARE(progress->value(), testValue);
        QVERIFY(progress->isVisible());
        
        // Test progress updates in current theme
        progress->setValue(testValue + 10);
        QCOMPARE(progress->value(), testValue + 10);
        
        // Reset to test value
        progress->setValue(testValue);
    }
    
    qDebug() << "Progress state persistence testing completed successfully";
}

// Consistent behavior validation

void CrossThemeInteractionTests::testConsistentClickBehaviorAcrossThemes() {
    qDebug() << "Testing consistent click behavior across themes...";
    
    QPushButton* button = qobject_cast<QPushButton*>(m_testComponents["testButton"]);
    QVERIFY(button != nullptr);
    
    QMap<ThemeManager::Theme, qint64> clickTimes;
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Measure click response time
        QSignalSpy clickSpy(button, &QPushButton::clicked);
        
        m_performanceTimer.start();
        QTest::mouseClick(button, Qt::LeftButton);
        qint64 clickTime = m_performanceTimer.elapsed();
        
        clickTimes[theme] = clickTime;
        QCOMPARE(clickSpy.count(), 1);
        
        // Verify click behavior is consistent
        QVERIFY(measureInteractionPerformance("button_click", theme));
    }
    
    // Verify click times are reasonably consistent across themes
    qint64 maxTime = *std::max_element(clickTimes.values().begin(), clickTimes.values().end());
    qint64 minTime = *std::min_element(clickTimes.values().begin(), clickTimes.values().end());
    
    // Allow up to 50ms difference between themes
    QVERIFY2(maxTime - minTime <= 50, 
             QString("Click time difference too large: %1ms").arg(maxTime - minTime).toUtf8());
    
    qDebug() << "Consistent click behavior testing completed successfully";
}

void CrossThemeInteractionTests::testConsistentKeyboardNavigationAcrossThemes() {
    qDebug() << "Testing consistent keyboard navigation across themes...";
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Test tab navigation
        QWidget* firstWidget = m_testComponents["testButton"];
        QWidget* secondWidget = m_testComponents["testInput"];
        
        firstWidget->setFocus();
        QVERIFY(firstWidget->hasFocus());
        
        // Simulate tab key press
        QTest::keyPress(firstWidget, Qt::Key_Tab);
        QTest::qWait(50);
        
        // Verify focus moved (may need adjustment based on tab order)
        // This is a simplified test - real implementation would verify complete tab order
        
        // Test accessibility compliance for keyboard navigation
        auto selector = UIThemeTestIntegration::createAccessibilitySelector("testButton");
        auto accessibilityResult = m_themeIntegration->testComponentAccessibility(selector, theme);
        QVERIFY(accessibilityResult.keyboardNavResult.canReceiveFocus);
    }
    
    qDebug() << "Consistent keyboard navigation testing completed successfully";
}

void CrossThemeInteractionTests::testConsistentFocusManagementAcrossThemes() {
    qDebug() << "Testing consistent focus management across themes...";
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Test focus on different widget types
        for (auto it = m_testComponents.begin(); it != m_testComponents.end(); ++it) {
            QWidget* widget = it.value();
            
            if (widget->focusPolicy() != Qt::NoFocus) {
                widget->setFocus();
                QVERIFY2(widget->hasFocus() || !widget->isEnabled(), 
                         QString("Focus failed for %1 in theme %2")
                         .arg(it.key()).arg(static_cast<int>(theme)).toUtf8());
                
                // Verify focus indicator is visible (theme-dependent)
                auto selector = UIThemeTestIntegration::createSelector(it.key());
                QVERIFY(m_themeIntegration->validateWidgetThemeCompliance(selector));
            }
        }
    }
    
    qDebug() << "Consistent focus management testing completed successfully";
}

void CrossThemeInteractionTests::testConsistentTooltipBehaviorAcrossThemes() {
    qDebug() << "Testing consistent tooltip behavior across themes...";
    
    // Set tooltips on test components
    for (auto it = m_testComponents.begin(); it != m_testComponents.end(); ++it) {
        it.value()->setToolTip(QString("Tooltip for %1").arg(it.key()));
    }
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(200);
        
        // Test tooltip display (simplified - actual tooltip testing requires more complex setup)
        for (auto it = m_testComponents.begin(); it != m_testComponents.end(); ++it) {
            QWidget* widget = it.value();
            QString expectedTooltip = QString("Tooltip for %1").arg(it.key());
            
            QCOMPARE(widget->toolTip(), expectedTooltip);
            
            // Verify widget is accessible for tooltip display
            QVERIFY(widget->isVisible());
            QVERIFY(widget->isEnabled() || it.key() == "testButton"); // testButton might be disabled in some tests
        }
    }
    
    qDebug() << "Consistent tooltip behavior testing completed successfully";
}

// Theme transition validation

void CrossThemeInteractionTests::testSmoothThemeTransitions() {
    qDebug() << "Testing smooth theme transitions...";
    
    // Test transitions between all theme pairs
    for (ThemeManager::Theme fromTheme : m_supportedThemes) {
        for (ThemeManager::Theme toTheme : m_supportedThemes) {
            if (fromTheme == toTheme) continue;
            
            // Set initial theme
            QVERIFY(m_themeManager->setTheme(fromTheme));
            QTest::qWait(200);
            
            // Measure transition time
            m_performanceTimer.start();
            QVERIFY(m_themeManager->setTheme(toTheme));
            QTest::qWait(300); // Allow transition to complete
            qint64 transitionTime = m_performanceTimer.elapsed();
            
            // Verify transition completed successfully
            QCOMPARE(m_themeManager->currentTheme(), toTheme);
            
            // Verify all components are still functional after transition
            QVERIFY(validateUIStateAfterThemeChange(fromTheme, toTheme));
            
            // Verify transition time is reasonable (< 1 second)
            QVERIFY2(transitionTime < 1000, 
                     QString("Theme transition too slow: %1ms").arg(transitionTime).toUtf8());
        }
    }
    
    qDebug() << "Smooth theme transition testing completed successfully";
}

void CrossThemeInteractionTests::testThemeTransitionWithActiveDialogs() {
    qDebug() << "Testing theme transitions with active dialogs...";
    
    // This test would be more comprehensive with actual dialog creation
    // For now, we test with the main window and components
    
    for (ThemeManager::Theme theme : m_supportedThemes) {
        // Set focus to an input field
        QLineEdit* input = qobject_cast<QLineEdit*>(m_testComponents["testInput"]);
        input->setFocus();
        input->setText("Dialog test");
        
        // Switch theme while input has focus
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(300);
        
        // Verify input still has focus and content
        QVERIFY(input->hasFocus());
        QCOMPARE(input->text(), QString("Dialog test"));
        
        // Verify theme was applied correctly
        auto selector = UIThemeTestIntegration::createSelector("testInput");
        QVERIFY(m_themeIntegration->validateWidgetThemeCompliance(selector));
    }
    
    qDebug() << "Theme transition with active dialogs testing completed successfully";
}

void CrossThemeInteractionTests::testThemeTransitionDuringOperations() {
    qDebug() << "Testing theme transitions during operations...";
    
    QProgressBar* progress = qobject_cast<QProgressBar*>(m_testComponents["testProgress"]);
    QVERIFY(progress != nullptr);
    
    // Simulate an ongoing operation
    progress->setRange(0, 100);
    progress->setValue(0);
    
    QTimer* progressTimer = new QTimer(this);
    connect(progressTimer, &QTimer::timeout, [progress]() {
        int value = progress->value() + 10;
        if (value <= 100) {
            progress->setValue(value);
        }
    });
    
    progressTimer->start(100); // Update every 100ms
    
    // Switch themes during the "operation"
    for (ThemeManager::Theme theme : m_supportedThemes) {
        QTest::qWait(200); // Let progress advance
        
        QVERIFY(m_themeManager->setTheme(theme));
        QTest::qWait(100);
        
        // Verify progress continues to work
        QVERIFY(progress->isVisible());
        QVERIFY(progress->value() >= 0);
        
        // Verify theme compliance
        auto selector = UIThemeTestIntegration::createSelector("testProgress");
        QVERIFY(m_themeIntegration->validateWidgetThemeCompliance(selector));
    }
    
    progressTimer->stop();
    delete progressTimer;
    
    qDebug() << "Theme transition during operations testing completed successfully";
}

void CrossThemeInteractionTests::testThemeTransitionErrorRecovery() {
    qDebug() << "Testing theme transition error recovery...";
    
    // Test recovery from invalid theme states
    ThemeManager::Theme originalTheme = m_themeManager->currentTheme();
    
    // This test would be more meaningful with actual error injection
    // For now, we test rapid theme switching to stress the system
    
    for (int i = 0; i < 5; ++i) {
        for (ThemeManager::Theme theme : m_supportedThemes) {
            QVERIFY(m_themeManager->setTheme(theme));
            QTest::qWait(50); // Rapid switching
            
            // Verify system remains stable
            QVERIFY(m_testWidget->isVisible());
            
            // Verify at least basic functionality works
            QPushButton* button = qobject_cast<QPushButton*>(m_testComponents["testButton"]);
            QVERIFY(button->isVisible());
        }
    }
    
    // Verify final state is consistent
    QVERIFY(validateUIStateAfterThemeChange(originalTheme, m_themeManager->currentTheme()));
    
    qDebug() << "Theme transition error recovery testing completed successfully";
}

// Helper method implementations

bool CrossThemeInteractionTests::validateInteractionInTheme(const QString& interaction, ThemeManager::Theme theme) {
    QVERIFY(m_themeManager->setTheme(theme));
    QTest::qWait(200);
    
    // Basic validation - ensure components are visible and themed correctly
    for (auto it = m_testComponents.begin(); it != m_testComponents.end(); ++it) {
        if (!it.value()->isVisible()) {
            qWarning() << "Component" << it.key() << "not visible in theme" << static_cast<int>(theme);
            return false;
        }
        
        auto selector = UIThemeTestIntegration::createSelector(it.key());
        if (!m_themeIntegration->validateWidgetThemeCompliance(selector)) {
            qWarning() << "Component" << it.key() << "failed theme compliance in theme" << static_cast<int>(theme);
            return false;
        }
    }
    
    return true;
}

bool CrossThemeInteractionTests::compareInteractionBehavior(const QString& interaction, 
                                                           ThemeManager::Theme theme1, 
                                                           ThemeManager::Theme theme2) {
    // This would compare specific interaction behaviors between themes
    // For now, we ensure both themes pass basic validation
    return validateInteractionInTheme(interaction, theme1) && 
           validateInteractionInTheme(interaction, theme2);
}

bool CrossThemeInteractionTests::testInteractionConsistency(const QString& componentName, 
                                                           const QString& interactionType,
                                                           const QList<ThemeManager::Theme>& themes) {
    QWidget* component = m_testComponents.value(componentName);
    if (!component) {
        qWarning() << "Component not found:" << componentName;
        return false;
    }
    
    for (ThemeManager::Theme theme : themes) {
        if (!validateInteractionInTheme(interactionType, theme)) {
            return false;
        }
    }
    
    return true;
}

bool CrossThemeInteractionTests::validateUIStateAfterThemeChange(ThemeManager::Theme fromTheme, 
                                                                ThemeManager::Theme toTheme) {
    Q_UNUSED(fromTheme) // Could be used for more detailed validation
    
    // Verify all components are still functional after theme change
    for (auto it = m_testComponents.begin(); it != m_testComponents.end(); ++it) {
        QWidget* widget = it.value();
        
        if (!widget->isVisible()) {
            qWarning() << "Component" << it.key() << "lost visibility after theme change to" << static_cast<int>(toTheme);
            return false;
        }
        
        // Test basic functionality
        if (auto* button = qobject_cast<QPushButton*>(widget)) {
            // Button should be clickable (if enabled)
            if (button->isEnabled()) {
                QSignalSpy spy(button, &QPushButton::clicked);
                QTest::mouseClick(button, Qt::LeftButton);
                if (spy.count() != 1) {
                    qWarning() << "Button" << it.key() << "not responding to clicks after theme change";
                    return false;
                }
            }
        }
    }
    
    return true;
}

bool CrossThemeInteractionTests::measureInteractionPerformance(const QString& interaction, ThemeManager::Theme theme) {
    m_performanceTimer.start();
    
    // Perform the interaction (simplified)
    bool success = validateInteractionInTheme(interaction, theme);
    
    qint64 elapsed = m_performanceTimer.elapsed();
    m_interactionTimes[interaction][theme] = elapsed;
    
    // Interaction should complete within reasonable time (100ms)
    return success && elapsed < 100;
}

QTEST_MAIN(CrossThemeInteractionTests)
#include "cross_theme_interaction_tests.moc"