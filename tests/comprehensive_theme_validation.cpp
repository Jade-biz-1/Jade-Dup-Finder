#include <QtTest/QtTest>
#include <QApplication>
#include <QWidget>
#include <QDialog>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QCheckBox>
#include <QProgressBar>
#include <QLabel>
#include <QGroupBox>
#include <QTabWidget>
#include <QTreeWidget>
#include <QTableWidget>
#include <QScrollArea>
#include <QSplitter>
#include <QScreen>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QStyleFactory>
#include <QElapsedTimer>
#include "theme_manager.h"
#include "component_registry.h"
#include "style_validator.h"
#include "theme_performance_optimizer.h"
#include "core/logger.h"

class ComprehensiveThemeValidation : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();
    
    // Component visibility tests
    void testComponentVisibilityInLightTheme();
    void testComponentVisibilityInDarkTheme();
    void testComponentVisibilityInHighContrastTheme();
    void testCheckboxVisibilityAcrossThemes();
    void testProgressBarVisibilityAcrossThemes();
    void testDialogVisibilityAcrossThemes();
    
    // Screen size and scaling tests
    void testThemeAcrossDifferentScreenSizes();
    void testThemeWithDifferentScalingFactors();
    void testMinimumSizeConstraints();
    void testResponsiveLayout();
    
    // Accessibility compliance tests
    void testContrastRatiosInAllThemes();
    void testFocusIndicatorsVisibility();
    void testKeyboardNavigation();
    void testScreenReaderCompatibility();
    void testHighContrastModeCompliance();
    
    // Performance tests
    void testThemeSwitchingPerformance();
    void testCacheEfficiency();
    void testBatchUpdatePerformance();
    void testMemoryUsageDuringThemeSwitch();
    
    // Comprehensive workflow tests
    void testCompleteUserWorkflow();
    void testMultipleDialogsThemeConsistency();
    void testThemeStateAfterErrors();
    void testThemePersistenceAcrossRestarts();
    
    // Edge case tests
    void testThemeWithNullWidgets();
    void testThemeWithDestroyedWidgets();
    void testConcurrentThemeChanges();
    void testThemeWithLargeNumberOfComponents();

private:
    // Test helper methods
    QWidget* createTestWidget(const QString& widgetType);
    QDialog* createTestDialog();
    void validateWidgetVisibility(QWidget* widget, const QString& themeName);
    void validateContrastRatio(const QColor& foreground, const QColor& background, double minRatio = 4.5);
    void simulateUserInteraction(QWidget* widget);
    void measurePerformance(std::function<void()> operation, const QString& operationName, int maxTimeMs = 100);
    bool isWidgetProperlyVisible(QWidget* widget);
    void validateAccessibilityFeatures(QWidget* widget);
    void testWidgetAcrossScreenSizes(QWidget* widget);
    
    // Test data
    ThemeManager* m_themeManager;
    ComponentRegistry* m_componentRegistry;
    StyleValidator* m_styleValidator;
    ThemePerformanceOptimizer* m_performanceOptimizer;
    QList<QWidget*> m_testWidgets;
    QList<QDialog*> m_testDialogs;
    
    // Performance tracking
    struct PerformanceResults {
        qint64 lightThemeSwitchTime;
        qint64 darkThemeSwitchTime;
        qint64 highContrastSwitchTime;
        int cacheHitRate;
        int memoryUsageMB;
        bool performanceTargetMet;
    } m_performanceResults;
};

void ComprehensiveThemeValidation::initTestCase()
{
    // Initialize theme system
    m_themeManager = ThemeManager::instance();
    QVERIFY(m_themeManager != nullptr);
    
    // Enable performance optimization for testing
    m_themeManager->enablePerformanceOptimization(true);
    m_themeManager->setPerformanceTarget(100); // 100ms target
    
    // Reset performance metrics
    m_themeManager->resetPerformanceMetrics();
    
    LOG_INFO(LogCategories::UI, "=== Starting Comprehensive Theme Validation Tests ===");
}

void ComprehensiveThemeValidation::cleanupTestCase()
{
    // Clean up test widgets
    for (QWidget* widget : m_testWidgets) {
        delete widget;
    }
    m_testWidgets.clear();
    
    for (QDialog* dialog : m_testDialogs) {
        delete dialog;
    }
    m_testDialogs.clear();
    
    // Generate final performance report
    QString performanceReport = m_themeManager->generatePerformanceReport();
    LOG_INFO(LogCategories::UI, QString("Final Performance Report:\n%1").arg(performanceReport));
    
    LOG_INFO(LogCategories::UI, "=== Comprehensive Theme Validation Tests Completed ===");
}

void ComprehensiveThemeValidation::init()
{
    // Reset to system default theme before each test
    m_themeManager->setTheme(ThemeManager::SystemDefault);
}

void ComprehensiveThemeValidation::cleanup()
{
    // Clean up any widgets created during the test
    // (Individual tests should clean up their own widgets)
}

void ComprehensiveThemeValidation::testComponentVisibilityInLightTheme()
{
    LOG_INFO(LogCategories::UI, "Testing component visibility in light theme");
    
    // Switch to light theme
    QElapsedTimer timer;
    timer.start();
    m_themeManager->setTheme(ThemeManager::Light);
    qint64 switchTime = timer.elapsed();
    
    QVERIFY(switchTime < 100); // Should complete within 100ms
    QCOMPARE(m_themeManager->currentTheme(), ThemeManager::Light);
    
    // Test various widget types
    QStringList widgetTypes = {"QPushButton", "QLineEdit", "QComboBox", "QCheckBox", 
                              "QProgressBar", "QLabel", "QGroupBox", "QTabWidget"};
    
    for (const QString& widgetType : widgetTypes) {
        QWidget* widget = createTestWidget(widgetType);
        QVERIFY(widget != nullptr);
        
        m_themeManager->applyToWidget(widget);
        validateWidgetVisibility(widget, "Light");
        
        m_testWidgets.append(widget);
    }
    
    LOG_INFO(LogCategories::UI, QString("Light theme visibility test completed in %1ms").arg(switchTime));
}

void ComprehensiveThemeValidation::testComponentVisibilityInDarkTheme()
{
    LOG_INFO(LogCategories::UI, "Testing component visibility in dark theme");
    
    // Switch to dark theme
    QElapsedTimer timer;
    timer.start();
    m_themeManager->setTheme(ThemeManager::Dark);
    qint64 switchTime = timer.elapsed();
    
    QVERIFY(switchTime < 100); // Should complete within 100ms
    QCOMPARE(m_themeManager->currentTheme(), ThemeManager::Dark);
    
    // Test various widget types
    QStringList widgetTypes = {"QPushButton", "QLineEdit", "QComboBox", "QCheckBox", 
                              "QProgressBar", "QLabel", "QGroupBox", "QTabWidget"};
    
    for (const QString& widgetType : widgetTypes) {
        QWidget* widget = createTestWidget(widgetType);
        QVERIFY(widget != nullptr);
        
        m_themeManager->applyToWidget(widget);
        validateWidgetVisibility(widget, "Dark");
        
        m_testWidgets.append(widget);
    }
    
    LOG_INFO(LogCategories::UI, QString("Dark theme visibility test completed in %1ms").arg(switchTime));
}

void ComprehensiveThemeValidation::testComponentVisibilityInHighContrastTheme()
{
    LOG_INFO(LogCategories::UI, "Testing component visibility in high contrast theme");
    
    // Switch to high contrast theme
    QElapsedTimer timer;
    timer.start();
    m_themeManager->setTheme(ThemeManager::HighContrast);
    qint64 switchTime = timer.elapsed();
    
    QVERIFY(switchTime < 100); // Should complete within 100ms
    QCOMPARE(m_themeManager->currentTheme(), ThemeManager::HighContrast);
    
    // Test various widget types with enhanced accessibility requirements
    QStringList widgetTypes = {"QPushButton", "QLineEdit", "QComboBox", "QCheckBox", 
                              "QProgressBar", "QLabel", "QGroupBox", "QTabWidget"};
    
    for (const QString& widgetType : widgetTypes) {
        QWidget* widget = createTestWidget(widgetType);
        QVERIFY(widget != nullptr);
        
        m_themeManager->applyToWidget(widget);
        validateWidgetVisibility(widget, "HighContrast");
        validateAccessibilityFeatures(widget);
        
        m_testWidgets.append(widget);
    }
    
    LOG_INFO(LogCategories::UI, QString("High contrast theme visibility test completed in %1ms").arg(switchTime));
}

void ComprehensiveThemeValidation::testCheckboxVisibilityAcrossThemes()
{
    LOG_INFO(LogCategories::UI, "Testing checkbox visibility across all themes");
    
    QCheckBox* checkbox = new QCheckBox("Test Checkbox");
    checkbox->show();
    m_testWidgets.append(checkbox);
    
    // Test in each theme
    QList<ThemeManager::Theme> themes = {ThemeManager::Light, ThemeManager::Dark, ThemeManager::HighContrast};
    
    for (ThemeManager::Theme theme : themes) {
        m_themeManager->setTheme(theme);
        m_themeManager->applyToWidget(checkbox);
        
        // Verify checkbox is visible and functional
        QVERIFY(isWidgetProperlyVisible(checkbox));
        
        // Test both checked and unchecked states
        checkbox->setChecked(false);
        QTest::qWait(10); // Allow UI to update
        QVERIFY(isWidgetProperlyVisible(checkbox));
        
        checkbox->setChecked(true);
        QTest::qWait(10); // Allow UI to update
        QVERIFY(isWidgetProperlyVisible(checkbox));
        
        // Simulate user interaction
        simulateUserInteraction(checkbox);
    }
    
    LOG_INFO(LogCategories::UI, "Checkbox visibility test across themes completed");
}

void ComprehensiveThemeValidation::testProgressBarVisibilityAcrossThemes()
{
    LOG_INFO(LogCategories::UI, "Testing progress bar visibility across all themes");
    
    QProgressBar* progressBar = new QProgressBar();
    progressBar->setRange(0, 100);
    progressBar->setValue(50);
    progressBar->show();
    m_testWidgets.append(progressBar);
    
    // Test in each theme
    QList<ThemeManager::Theme> themes = {ThemeManager::Light, ThemeManager::Dark, ThemeManager::HighContrast};
    
    for (ThemeManager::Theme theme : themes) {
        m_themeManager->setTheme(theme);
        m_themeManager->applyToWidget(progressBar);
        
        // Verify progress bar is visible
        QVERIFY(isWidgetProperlyVisible(progressBar));
        
        // Test different progress values
        for (int value = 0; value <= 100; value += 25) {
            progressBar->setValue(value);
            QTest::qWait(10); // Allow UI to update
            QVERIFY(isWidgetProperlyVisible(progressBar));
        }
    }
    
    LOG_INFO(LogCategories::UI, "Progress bar visibility test across themes completed");
}

void ComprehensiveThemeValidation::testDialogVisibilityAcrossThemes()
{
    LOG_INFO(LogCategories::UI, "Testing dialog visibility across all themes");
    
    QDialog* dialog = createTestDialog();
    QVERIFY(dialog != nullptr);
    
    // Test in each theme
    QList<ThemeManager::Theme> themes = {ThemeManager::Light, ThemeManager::Dark, ThemeManager::HighContrast};
    
    for (ThemeManager::Theme theme : themes) {
        m_themeManager->setTheme(theme);
        m_themeManager->applyToDialog(dialog);
        
        // Verify dialog and all child widgets are visible
        QVERIFY(isWidgetProperlyVisible(dialog));
        
        QList<QWidget*> children = dialog->findChildren<QWidget*>();
        for (QWidget* child : children) {
            if (child && child->isVisible()) {
                QVERIFY(isWidgetProperlyVisible(child));
            }
        }
    }
    
    m_testDialogs.append(dialog);
    LOG_INFO(LogCategories::UI, "Dialog visibility test across themes completed");
}

void ComprehensiveThemeValidation::testThemeAcrossDifferentScreenSizes()
{
    LOG_INFO(LogCategories::UI, "Testing theme across different screen sizes");
    
    QWidget* testWidget = createTestWidget("QWidget");
    QVERIFY(testWidget != nullptr);
    
    // Simulate different screen sizes
    QList<QSize> screenSizes = {
        QSize(1024, 768),   // Standard
        QSize(1920, 1080),  // Full HD
        QSize(2560, 1440),  // QHD
        QSize(3840, 2160),  // 4K
        QSize(800, 600)     // Small screen
    };
    
    for (const QSize& size : screenSizes) {
        testWidget->resize(size);
        testWidgetAcrossScreenSizes(testWidget);
        
        // Test theme application at this size
        m_themeManager->setTheme(ThemeManager::Light);
        m_themeManager->applyToWidget(testWidget);
        QVERIFY(isWidgetProperlyVisible(testWidget));
        
        m_themeManager->setTheme(ThemeManager::Dark);
        m_themeManager->applyToWidget(testWidget);
        QVERIFY(isWidgetProperlyVisible(testWidget));
    }
    
    m_testWidgets.append(testWidget);
    LOG_INFO(LogCategories::UI, "Screen size compatibility test completed");
}

void ComprehensiveThemeValidation::testThemeWithDifferentScalingFactors()
{
    LOG_INFO(LogCategories::UI, "Testing theme with different scaling factors");
    
    QWidget* testWidget = createTestWidget("QWidget");
    QVERIFY(testWidget != nullptr);
    
    // Test different scaling factors (simulated by font size changes)
    QList<int> fontSizes = {8, 9, 10, 12, 14, 16, 18, 20};
    
    for (int fontSize : fontSizes) {
        QFont font = testWidget->font();
        font.setPointSize(fontSize);
        testWidget->setFont(font);
        
        // Apply theme and verify visibility
        m_themeManager->setTheme(ThemeManager::Light);
        m_themeManager->applyToWidget(testWidget);
        QVERIFY(isWidgetProperlyVisible(testWidget));
        
        // Verify minimum size constraints are maintained
        QSize minSize = testWidget->minimumSize();
        QSize actualSize = testWidget->size();
        QVERIFY(actualSize.width() >= minSize.width());
        QVERIFY(actualSize.height() >= minSize.height());
    }
    
    m_testWidgets.append(testWidget);
    LOG_INFO(LogCategories::UI, "Scaling factor compatibility test completed");
}

void ComprehensiveThemeValidation::testMinimumSizeConstraints()
{
    LOG_INFO(LogCategories::UI, "Testing minimum size constraints");
    
    QStringList widgetTypes = {"QPushButton", "QLineEdit", "QComboBox", "QCheckBox"};
    
    for (const QString& widgetType : widgetTypes) {
        QWidget* widget = createTestWidget(widgetType);
        QVERIFY(widget != nullptr);
        
        // Apply theme
        m_themeManager->applyToWidget(widget);
        
        // Verify minimum size is set and respected
        QSize minSize = widget->minimumSize();
        QVERIFY(!minSize.isEmpty());
        QVERIFY(minSize.width() > 0);
        QVERIFY(minSize.height() > 0);
        
        // Try to resize smaller than minimum
        widget->resize(minSize.width() / 2, minSize.height() / 2);
        QTest::qWait(10);
        
        // Verify widget maintains minimum size
        QSize actualSize = widget->size();
        QVERIFY(actualSize.width() >= minSize.width());
        QVERIFY(actualSize.height() >= minSize.height());
        
        m_testWidgets.append(widget);
    }
    
    LOG_INFO(LogCategories::UI, "Minimum size constraints test completed");
}

void ComprehensiveThemeValidation::testResponsiveLayout()
{
    LOG_INFO(LogCategories::UI, "Testing responsive layout behavior");
    
    QDialog* dialog = createTestDialog();
    QVERIFY(dialog != nullptr);
    
    // Test different dialog sizes
    QList<QSize> dialogSizes = {
        QSize(300, 200),   // Small
        QSize(600, 400),   // Medium
        QSize(1000, 700),  // Large
        QSize(1400, 900)   // Extra large
    };
    
    for (const QSize& size : dialogSizes) {
        dialog->resize(size);
        m_themeManager->applyToDialog(dialog);
        
        // Verify all child widgets are properly positioned and visible
        QList<QWidget*> children = dialog->findChildren<QWidget*>();
        for (QWidget* child : children) {
            if (child && child->isVisible()) {
                QVERIFY(isWidgetProperlyVisible(child));
                
                // Verify child is within dialog bounds
                QRect childGeometry = child->geometry();
                QRect dialogGeometry = dialog->rect();
                QVERIFY(dialogGeometry.contains(childGeometry) || 
                       dialogGeometry.intersects(childGeometry));
            }
        }
    }
    
    m_testDialogs.append(dialog);
    LOG_INFO(LogCategories::UI, "Responsive layout test completed");
}

void ComprehensiveThemeValidation::testContrastRatiosInAllThemes()
{
    LOG_INFO(LogCategories::UI, "Testing contrast ratios in all themes");
    
    QList<ThemeManager::Theme> themes = {ThemeManager::Light, ThemeManager::Dark, ThemeManager::HighContrast};
    
    for (ThemeManager::Theme theme : themes) {
        m_themeManager->setTheme(theme);
        ThemeData themeData = m_themeManager->getCurrentThemeData();
        
        // Test critical color combinations
        validateContrastRatio(themeData.colors.foreground, themeData.colors.background, 4.5);
        validateContrastRatio(themeData.colors.accent, themeData.colors.background, 3.0);
        
        // High contrast theme should have higher ratios
        if (theme == ThemeManager::HighContrast) {
            validateContrastRatio(themeData.colors.foreground, themeData.colors.background, 7.0);
        }
        
        // Verify accessibility compliance
        QVERIFY(themeData.meetsAccessibilityStandards());
    }
    
    LOG_INFO(LogCategories::UI, "Contrast ratio validation completed");
}

void ComprehensiveThemeValidation::testFocusIndicatorsVisibility()
{
    LOG_INFO(LogCategories::UI, "Testing focus indicators visibility");
    
    QPushButton* button = new QPushButton("Test Button");
    button->show();
    m_testWidgets.append(button);
    
    // Enable enhanced focus indicators
    m_themeManager->enableEnhancedFocusIndicators(true);
    
    QList<ThemeManager::Theme> themes = {ThemeManager::Light, ThemeManager::Dark, ThemeManager::HighContrast};
    
    for (ThemeManager::Theme theme : themes) {
        m_themeManager->setTheme(theme);
        m_themeManager->applyToWidget(button);
        
        // Set focus and verify focus indicator is visible
        button->setFocus();
        QTest::qWait(10);
        
        QVERIFY(button->hasFocus());
        QVERIFY(isWidgetProperlyVisible(button));
        
        // Focus indicator should be more prominent in high contrast mode
        if (theme == ThemeManager::HighContrast) {
            // Additional validation for high contrast focus indicators
            QString styleSheet = button->styleSheet();
            QVERIFY(styleSheet.contains("focus") || styleSheet.contains("outline"));
        }
    }
    
    LOG_INFO(LogCategories::UI, "Focus indicators visibility test completed");
}

void ComprehensiveThemeValidation::testKeyboardNavigation()
{
    LOG_INFO(LogCategories::UI, "Testing keyboard navigation");
    
    QDialog* dialog = createTestDialog();
    QVERIFY(dialog != nullptr);
    
    // Setup accessible tab order
    m_themeManager->setupAccessibleTabOrder(dialog);
    m_themeManager->setupAccessibleKeyboardShortcuts(dialog);
    
    // Find all focusable widgets
    QList<QWidget*> focusableWidgets;
    QList<QWidget*> children = dialog->findChildren<QWidget*>();
    
    for (QWidget* child : children) {
        if (child && child->focusPolicy() != Qt::NoFocus && child->isVisible() && child->isEnabled()) {
            focusableWidgets.append(child);
        }
    }
    
    QVERIFY(!focusableWidgets.isEmpty());
    
    // Test tab navigation
    if (!focusableWidgets.isEmpty()) {
        focusableWidgets.first()->setFocus();
        
        for (int i = 0; i < focusableWidgets.size() - 1; ++i) {
            QWidget* currentWidget = focusableWidgets[i];
            QVERIFY(currentWidget->hasFocus());
            
            // Simulate Tab key press
            QTest::keyPress(currentWidget, Qt::Key_Tab);
            QTest::qWait(10);
            
            // Verify focus moved to next widget
            QWidget* nextWidget = focusableWidgets[i + 1];
            QVERIFY(nextWidget->hasFocus());
        }
    }
    
    m_testDialogs.append(dialog);
    LOG_INFO(LogCategories::UI, "Keyboard navigation test completed");
}

void ComprehensiveThemeValidation::testScreenReaderCompatibility()
{
    LOG_INFO(LogCategories::UI, "Testing screen reader compatibility");
    
    QWidget* widget = createTestWidget("QPushButton");
    QVERIFY(widget != nullptr);
    
    // Verify accessibility properties are set
    QVERIFY(!widget->accessibleName().isEmpty() || !widget->toolTip().isEmpty());
    
    // Test with alternative indicators enabled
    m_themeManager->enableAlternativeIndicators(true);
    m_themeManager->addTextIndicator(widget, "Button for testing");
    
    // Verify text indicator was added
    QVERIFY(widget->toolTip().contains("Button for testing"));
    
    m_testWidgets.append(widget);
    LOG_INFO(LogCategories::UI, "Screen reader compatibility test completed");
}

void ComprehensiveThemeValidation::testHighContrastModeCompliance()
{
    LOG_INFO(LogCategories::UI, "Testing high contrast mode compliance");
    
    // Enable high contrast mode
    m_themeManager->enableHighContrastMode(true);
    m_themeManager->setTheme(ThemeManager::HighContrast);
    
    QWidget* widget = createTestWidget("QWidget");
    QVERIFY(widget != nullptr);
    
    m_themeManager->applyToWidget(widget);
    
    // Verify high contrast compliance
    QVERIFY(m_themeManager->validateAccessibilityCompliance());
    
    // Test contrast ratios are enhanced
    ThemeData themeData = m_themeManager->getCurrentThemeData();
    double contrastRatio = themeData.getContrastRatio(themeData.colors.foreground, themeData.colors.background);
    QVERIFY(contrastRatio >= 7.0); // WCAG AAA standard
    
    m_testWidgets.append(widget);
    LOG_INFO(LogCategories::UI, "High contrast mode compliance test completed");
}

void ComprehensiveThemeValidation::testThemeSwitchingPerformance()
{
    LOG_INFO(LogCategories::UI, "Testing theme switching performance");
    
    // Create multiple widgets to test performance with
    QList<QWidget*> performanceTestWidgets;
    for (int i = 0; i < 50; ++i) {
        QWidget* widget = createTestWidget("QPushButton");
        performanceTestWidgets.append(widget);
        m_testWidgets.append(widget);
    }
    
    // Test switching between themes and measure performance
    QList<ThemeManager::Theme> themes = {ThemeManager::Light, ThemeManager::Dark, ThemeManager::HighContrast};
    
    for (ThemeManager::Theme theme : themes) {
        measurePerformance([this, theme]() {
            m_themeManager->setTheme(theme);
        }, QString("Switch to %1 theme").arg(static_cast<int>(theme)), 100);
        
        // Verify theme was applied
        QCOMPARE(m_themeManager->currentTheme(), theme);
    }
    
    // Check overall performance metrics
    qint64 averageTime = m_themeManager->getAverageThemeSwitchTime();
    QVERIFY(averageTime < 100); // Should average under 100ms
    
    LOG_INFO(LogCategories::UI, QString("Theme switching performance test completed. Average time: %1ms").arg(averageTime));
}

void ComprehensiveThemeValidation::testCacheEfficiency()
{
    LOG_INFO(LogCategories::UI, "Testing cache efficiency");
    
    // Enable caching
    m_themeManager->enableStyleSheetCaching(true);
    
    // Create widgets and apply themes multiple times
    QWidget* widget = createTestWidget("QPushButton");
    m_testWidgets.append(widget);
    
    // First application should miss cache
    m_themeManager->setTheme(ThemeManager::Light);
    m_themeManager->applyToWidget(widget);
    
    // Second application should hit cache
    m_themeManager->applyToWidget(widget);
    
    // Switch themes and back to test cache
    m_themeManager->setTheme(ThemeManager::Dark);
    m_themeManager->applyToWidget(widget);
    
    m_themeManager->setTheme(ThemeManager::Light);
    m_themeManager->applyToWidget(widget);
    
    // Check cache hit rate
    int hitRate = m_themeManager->getCacheHitRate();
    QVERIFY(hitRate > 0); // Should have some cache hits
    
    LOG_INFO(LogCategories::UI, QString("Cache efficiency test completed. Hit rate: %1%").arg(hitRate));
}

void ComprehensiveThemeValidation::testBatchUpdatePerformance()
{
    LOG_INFO(LogCategories::UI, "Testing batch update performance");
    
    // Enable batch updates
    m_themeManager->enableBatchUpdates(true);
    
    // Create many widgets
    QList<QWidget*> batchTestWidgets;
    for (int i = 0; i < 100; ++i) {
        QWidget* widget = createTestWidget("QLabel");
        batchTestWidgets.append(widget);
        m_testWidgets.append(widget);
    }
    
    // Measure batch update performance
    measurePerformance([this]() {
        m_themeManager->setTheme(ThemeManager::Dark);
    }, "Batch update of 100 widgets", 200); // Allow more time for batch processing
    
    LOG_INFO(LogCategories::UI, "Batch update performance test completed");
}

void ComprehensiveThemeValidation::testMemoryUsageDuringThemeSwitch()
{
    LOG_INFO(LogCategories::UI, "Testing memory usage during theme switch");
    
    // This is a basic test - in a real implementation, you would use
    // platform-specific memory monitoring tools
    
    QWidget* widget = createTestWidget("QWidget");
    m_testWidgets.append(widget);
    
    // Switch themes multiple times
    for (int i = 0; i < 10; ++i) {
        m_themeManager->setTheme(ThemeManager::Light);
        m_themeManager->setTheme(ThemeManager::Dark);
    }
    
    // Verify widget is still valid and functional
    QVERIFY(isWidgetProperlyVisible(widget));
    
    LOG_INFO(LogCategories::UI, "Memory usage test completed");
}

void ComprehensiveThemeValidation::testCompleteUserWorkflow()
{
    LOG_INFO(LogCategories::UI, "Testing complete user workflow");
    
    // Simulate a complete user workflow
    QDialog* mainDialog = createTestDialog();
    QVERIFY(mainDialog != nullptr);
    
    // 1. User opens dialog
    m_themeManager->applyToDialog(mainDialog);
    QVERIFY(isWidgetProperlyVisible(mainDialog));
    
    // 2. User changes theme
    m_themeManager->setTheme(ThemeManager::Dark);
    QVERIFY(isWidgetProperlyVisible(mainDialog));
    
    // 3. User interacts with controls
    QList<QWidget*> children = mainDialog->findChildren<QWidget*>();
    for (QWidget* child : children) {
        if (child && child->isVisible()) {
            simulateUserInteraction(child);
            QVERIFY(isWidgetProperlyVisible(child));
        }
    }
    
    // 4. User changes theme again
    m_themeManager->setTheme(ThemeManager::HighContrast);
    QVERIFY(isWidgetProperlyVisible(mainDialog));
    
    // 5. Verify all interactions still work
    for (QWidget* child : children) {
        if (child && child->isVisible()) {
            QVERIFY(isWidgetProperlyVisible(child));
        }
    }
    
    m_testDialogs.append(mainDialog);
    LOG_INFO(LogCategories::UI, "Complete user workflow test completed");
}

void ComprehensiveThemeValidation::testMultipleDialogsThemeConsistency()
{
    LOG_INFO(LogCategories::UI, "Testing multiple dialogs theme consistency");
    
    // Create multiple dialogs
    QList<QDialog*> dialogs;
    for (int i = 0; i < 5; ++i) {
        QDialog* dialog = createTestDialog();
        dialogs.append(dialog);
        m_testDialogs.append(dialog);
    }
    
    // Apply theme to all dialogs
    m_themeManager->setTheme(ThemeManager::Dark);
    
    for (QDialog* dialog : dialogs) {
        m_themeManager->applyToDialog(dialog);
        QVERIFY(isWidgetProperlyVisible(dialog));
    }
    
    // Change theme and verify all dialogs update consistently
    m_themeManager->setTheme(ThemeManager::Light);
    
    for (QDialog* dialog : dialogs) {
        QVERIFY(isWidgetProperlyVisible(dialog));
        
        // Verify all child widgets are also properly themed
        QList<QWidget*> children = dialog->findChildren<QWidget*>();
        for (QWidget* child : children) {
            if (child && child->isVisible()) {
                QVERIFY(isWidgetProperlyVisible(child));
            }
        }
    }
    
    LOG_INFO(LogCategories::UI, "Multiple dialogs consistency test completed");
}

void ComprehensiveThemeValidation::testThemeStateAfterErrors()
{
    LOG_INFO(LogCategories::UI, "Testing theme state after errors");
    
    QWidget* widget = createTestWidget("QWidget");
    m_testWidgets.append(widget);
    
    // Apply valid theme first
    m_themeManager->setTheme(ThemeManager::Light);
    m_themeManager->applyToWidget(widget);
    QVERIFY(isWidgetProperlyVisible(widget));
    
    // Simulate error condition (try to apply theme to null widget)
    m_themeManager->applyToWidget(nullptr);
    
    // Verify original widget is still properly themed
    QVERIFY(isWidgetProperlyVisible(widget));
    QCOMPARE(m_themeManager->currentTheme(), ThemeManager::Light);
    
    // Verify theme system is still functional
    m_themeManager->setTheme(ThemeManager::Dark);
    m_themeManager->applyToWidget(widget);
    QVERIFY(isWidgetProperlyVisible(widget));
    
    LOG_INFO(LogCategories::UI, "Theme state after errors test completed");
}

void ComprehensiveThemeValidation::testThemePersistenceAcrossRestarts()
{
    LOG_INFO(LogCategories::UI, "Testing theme persistence across restarts");
    
    // Set a specific theme
    m_themeManager->setTheme(ThemeManager::Dark);
    m_themeManager->saveThemePreference();
    
    // Simulate restart by loading preferences
    m_themeManager->loadThemePreference();
    
    // Verify theme was restored
    QCOMPARE(m_themeManager->currentTheme(), ThemeManager::Dark);
    
    LOG_INFO(LogCategories::UI, "Theme persistence test completed");
}

void ComprehensiveThemeValidation::testThemeWithNullWidgets()
{
    LOG_INFO(LogCategories::UI, "Testing theme with null widgets");
    
    // Test applying theme to null widget (should not crash)
    m_themeManager->applyToWidget(nullptr);
    m_themeManager->applyToDialog(nullptr);
    
    // Verify theme system is still functional
    QWidget* validWidget = createTestWidget("QPushButton");
    m_testWidgets.append(validWidget);
    
    m_themeManager->applyToWidget(validWidget);
    QVERIFY(isWidgetProperlyVisible(validWidget));
    
    LOG_INFO(LogCategories::UI, "Null widgets test completed");
}

void ComprehensiveThemeValidation::testThemeWithDestroyedWidgets()
{
    LOG_INFO(LogCategories::UI, "Testing theme with destroyed widgets");
    
    QWidget* widget = createTestWidget("QPushButton");
    
    // Apply theme to widget
    m_themeManager->applyToWidget(widget);
    QVERIFY(isWidgetProperlyVisible(widget));
    
    // Destroy widget
    delete widget;
    
    // Verify theme system handles destroyed widgets gracefully
    m_themeManager->setTheme(ThemeManager::Dark);
    
    // Create new widget to verify system is still functional
    QWidget* newWidget = createTestWidget("QLabel");
    m_testWidgets.append(newWidget);
    
    m_themeManager->applyToWidget(newWidget);
    QVERIFY(isWidgetProperlyVisible(newWidget));
    
    LOG_INFO(LogCategories::UI, "Destroyed widgets test completed");
}

void ComprehensiveThemeValidation::testConcurrentThemeChanges()
{
    LOG_INFO(LogCategories::UI, "Testing concurrent theme changes");
    
    QWidget* widget = createTestWidget("QWidget");
    m_testWidgets.append(widget);
    
    // Rapidly switch themes
    for (int i = 0; i < 10; ++i) {
        m_themeManager->setTheme(ThemeManager::Light);
        m_themeManager->setTheme(ThemeManager::Dark);
        m_themeManager->setTheme(ThemeManager::HighContrast);
    }
    
    // Verify widget is still properly themed
    m_themeManager->applyToWidget(widget);
    QVERIFY(isWidgetProperlyVisible(widget));
    
    LOG_INFO(LogCategories::UI, "Concurrent theme changes test completed");
}

void ComprehensiveThemeValidation::testThemeWithLargeNumberOfComponents()
{
    LOG_INFO(LogCategories::UI, "Testing theme with large number of components");
    
    // Create many widgets
    QList<QWidget*> manyWidgets;
    for (int i = 0; i < 500; ++i) {
        QWidget* widget = createTestWidget(i % 2 == 0 ? "QPushButton" : "QLabel");
        manyWidgets.append(widget);
        m_testWidgets.append(widget);
    }
    
    // Measure performance with large number of widgets
    measurePerformance([this]() {
        m_themeManager->setTheme(ThemeManager::Dark);
    }, "Theme switch with 500 widgets", 500); // Allow more time for large number of widgets
    
    // Verify a sample of widgets are properly themed
    for (int i = 0; i < 10; ++i) {
        QWidget* widget = manyWidgets[i * 50]; // Sample every 50th widget
        QVERIFY(isWidgetProperlyVisible(widget));
    }
    
    LOG_INFO(LogCategories::UI, "Large number of components test completed");
}

// Helper method implementations
QWidget* ComprehensiveThemeValidation::createTestWidget(const QString& widgetType)
{
    QWidget* widget = nullptr;
    
    if (widgetType == "QPushButton") {
        widget = new QPushButton("Test Button");
    } else if (widgetType == "QLineEdit") {
        widget = new QLineEdit("Test Text");
    } else if (widgetType == "QComboBox") {
        QComboBox* combo = new QComboBox();
        combo->addItems({"Option 1", "Option 2", "Option 3"});
        widget = combo;
    } else if (widgetType == "QCheckBox") {
        widget = new QCheckBox("Test Checkbox");
    } else if (widgetType == "QProgressBar") {
        QProgressBar* progress = new QProgressBar();
        progress->setRange(0, 100);
        progress->setValue(50);
        widget = progress;
    } else if (widgetType == "QLabel") {
        widget = new QLabel("Test Label");
    } else if (widgetType == "QGroupBox") {
        QGroupBox* group = new QGroupBox("Test Group");
        QPushButton* button = new QPushButton("Inside Button");
        QVBoxLayout* layout = new QVBoxLayout(group);
        layout->addWidget(button);
        widget = group;
    } else if (widgetType == "QTabWidget") {
        QTabWidget* tabs = new QTabWidget();
        tabs->addTab(new QWidget(), "Tab 1");
        tabs->addTab(new QWidget(), "Tab 2");
        widget = tabs;
    } else {
        widget = new QWidget();
    }
    
    if (widget) {
        widget->show();
    }
    
    return widget;
}

QDialog* ComprehensiveThemeValidation::createTestDialog()
{
    QDialog* dialog = new QDialog();
    dialog->setWindowTitle("Test Dialog");
    dialog->resize(400, 300);
    
    QVBoxLayout* layout = new QVBoxLayout(dialog);
    
    // Add various controls
    layout->addWidget(new QLabel("Test Dialog Label"));
    layout->addWidget(new QLineEdit("Test input"));
    
    QComboBox* combo = new QComboBox();
    combo->addItems({"Option A", "Option B", "Option C"});
    layout->addWidget(combo);
    
    layout->addWidget(new QCheckBox("Test checkbox"));
    
    QProgressBar* progress = new QProgressBar();
    progress->setRange(0, 100);
    progress->setValue(75);
    layout->addWidget(progress);
    
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(new QPushButton("OK"));
    buttonLayout->addWidget(new QPushButton("Cancel"));
    layout->addLayout(buttonLayout);
    
    dialog->show();
    return dialog;
}

void ComprehensiveThemeValidation::validateWidgetVisibility(QWidget* widget, const QString& themeName)
{
    QVERIFY(widget != nullptr);
    QVERIFY(widget->isVisible());
    
    // Check that widget has proper styling applied
    QString styleSheet = widget->styleSheet();
    QVERIFY(!styleSheet.isEmpty());
    
    // Verify no hardcoded colors remain
    QVERIFY(!styleSheet.contains(QRegularExpression("#[0-9a-fA-F]{3,6}")));
    QVERIFY(!styleSheet.contains(QRegularExpression("rgb\\s*\\(")));
}

void ComprehensiveThemeValidation::validateContrastRatio(const QColor& foreground, const QColor& background, double minRatio)
{
    ThemeData themeData;
    double ratio = themeData.getContrastRatio(foreground, background);
    QVERIFY2(ratio >= minRatio, QString("Contrast ratio %1 is below minimum %2").arg(ratio).arg(minRatio).toLocal8Bit());
}

void ComprehensiveThemeValidation::simulateUserInteraction(QWidget* widget)
{
    if (!widget) return;
    
    // Simulate mouse hover
    QEnterEvent enterEvent(QPointF(10, 10), QPointF(10, 10), QPointF(10, 10));
    QApplication::sendEvent(widget, &enterEvent);
    QTest::qWait(10);
    
    // Simulate mouse click for clickable widgets
    if (qobject_cast<QPushButton*>(widget) || qobject_cast<QCheckBox*>(widget)) {
        QTest::mouseClick(widget, Qt::LeftButton);
        QTest::qWait(10);
    }
    
    // Simulate focus
    widget->setFocus();
    QTest::qWait(10);
}

void ComprehensiveThemeValidation::measurePerformance(std::function<void()> operation, const QString& operationName, int maxTimeMs)
{
    QElapsedTimer timer;
    timer.start();
    
    operation();
    
    qint64 elapsed = timer.elapsed();
    
    LOG_INFO(LogCategories::UI, QString("%1 completed in %2ms (target: %3ms)")
             .arg(operationName).arg(elapsed).arg(maxTimeMs));
    
    QVERIFY2(elapsed <= maxTimeMs, QString("%1 took %2ms, exceeding target of %3ms")
             .arg(operationName).arg(elapsed).arg(maxTimeMs).toLocal8Bit());
}

bool ComprehensiveThemeValidation::isWidgetProperlyVisible(QWidget* widget)
{
    if (!widget || !widget->isVisible()) {
        return false;
    }
    
    // Check that widget has reasonable size
    QSize size = widget->size();
    if (size.width() <= 0 || size.height() <= 0) {
        return false;
    }
    
    // Check that widget has styling applied
    QString styleSheet = widget->styleSheet();
    if (styleSheet.isEmpty()) {
        // Widget might be using palette colors, which is acceptable
        return true;
    }
    
    // Check for hardcoded colors (should not have any)
    if (styleSheet.contains(QRegularExpression("#[0-9a-fA-F]{3,6}")) ||
        styleSheet.contains(QRegularExpression("rgb\\s*\\("))) {
        return false;
    }
    
    return true;
}

void ComprehensiveThemeValidation::validateAccessibilityFeatures(QWidget* widget)
{
    QVERIFY(widget != nullptr);
    
    // Check for accessibility properties
    QString accessibleName = widget->accessibleName();
    QString toolTip = widget->toolTip();
    
    // Widget should have either accessible name or tooltip
    QVERIFY(!accessibleName.isEmpty() || !toolTip.isEmpty());
    
    // Check focus policy for interactive widgets
    if (qobject_cast<QPushButton*>(widget) || 
        qobject_cast<QLineEdit*>(widget) || 
        qobject_cast<QComboBox*>(widget) || 
        qobject_cast<QCheckBox*>(widget)) {
        QVERIFY(widget->focusPolicy() != Qt::NoFocus);
    }
}

void ComprehensiveThemeValidation::testWidgetAcrossScreenSizes(QWidget* widget)
{
    QVERIFY(widget != nullptr);
    
    // Verify widget maintains minimum size
    QSize minSize = widget->minimumSize();
    QSize actualSize = widget->size();
    
    QVERIFY(actualSize.width() >= minSize.width());
    QVERIFY(actualSize.height() >= minSize.height());
    
    // Verify widget is still functional
    QVERIFY(isWidgetProperlyVisible(widget));
}

QTEST_MAIN(ComprehensiveThemeValidation)
#include "comprehensive_theme_validation.moc"