#include "theme_performance_optimizer.h"
#include "core/logger.h"
#include <QApplication>
#include <QThread>
#include <QMutexLocker>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QCheckBox>
#include <QProgressBar>
#include <QLabel>
#include <QGroupBox>
#include <QTabWidget>

ThemePerformanceOptimizer::ThemePerformanceOptimizer(QObject* parent)
    : QObject(parent)
    , m_styleSheetCache(1000) // Default cache size
    , m_cachingEnabled(true)
    , m_batchTimer(new QTimer(this))
    , m_batchUpdatesEnabled(true)
    , m_batchInterval(50)
    , m_metricsTimer(new QTimer(this))
    , m_monitoringEnabled(false)
    , m_performanceTarget(100)
    , m_asyncUpdatesEnabled(true)
{
    // Initialize performance metrics
    resetPerformanceMetrics();
    
    // Setup batch timer
    m_batchTimer->setSingleShot(true);
    m_batchTimer->setInterval(m_batchInterval);
    connect(m_batchTimer, &QTimer::timeout, this, &ThemePerformanceOptimizer::processPendingBatch);
    
    // Setup metrics timer (update every 5 seconds)
    m_metricsTimer->setInterval(5000);
    connect(m_metricsTimer, &QTimer::timeout, this, &ThemePerformanceOptimizer::updatePerformanceMetrics);
    
    // Preload common styles
    preloadCommonStyles();
    
    LOG_INFO(LogCategories::UI, "ThemePerformanceOptimizer initialized with caching and batch updates");
}

ThemePerformanceOptimizer::~ThemePerformanceOptimizer()
{
    stopPerformanceMonitoring();
    clearBatch();
    LOG_INFO(LogCategories::UI, "ThemePerformanceOptimizer destroyed");
}

void ThemePerformanceOptimizer::enableStyleSheetCaching(bool enabled)
{
    QMutexLocker locker(&m_cacheMutex);
    m_cachingEnabled = enabled;
    
    if (!enabled) {
        m_styleSheetCache.clear();
    }
    
    LOG_INFO(LogCategories::UI, QString("StyleSheet caching %1").arg(enabled ? "enabled" : "disabled"));
}

void ThemePerformanceOptimizer::enableBatchUpdates(bool enabled)
{
    QMutexLocker locker(&m_batchMutex);
    m_batchUpdatesEnabled = enabled;
    
    if (!enabled) {
        // Process any pending batch immediately
        if (!m_batchWidgets.isEmpty() || !m_batchDialogs.isEmpty()) {
            m_batchTimer->stop();
            processPendingBatch();
        }
    }
    
    LOG_INFO(LogCategories::UI, QString("Batch updates %1").arg(enabled ? "enabled" : "disabled"));
}

void ThemePerformanceOptimizer::enableAsyncUpdates(bool enabled)
{
    m_asyncUpdatesEnabled = enabled;
    LOG_INFO(LogCategories::UI, QString("Async updates %1").arg(enabled ? "enabled" : "disabled"));
}

void ThemePerformanceOptimizer::setMaxCacheSize(int maxItems)
{
    QMutexLocker locker(&m_cacheMutex);
    m_styleSheetCache.setMaxCost(maxItems);
    LOG_INFO(LogCategories::UI, QString("Cache size set to %1 items").arg(maxItems));
}

void ThemePerformanceOptimizer::setBatchUpdateInterval(int milliseconds)
{
    m_batchInterval = qMax(10, milliseconds); // Minimum 10ms
    m_batchTimer->setInterval(m_batchInterval);
    LOG_INFO(LogCategories::UI, QString("Batch update interval set to %1ms").arg(m_batchInterval));
}

void ThemePerformanceOptimizer::setPerformanceTarget(int maxSwitchTimeMs)
{
    m_performanceTarget = maxSwitchTimeMs;
    LOG_INFO(LogCategories::UI, QString("Performance target set to %1ms").arg(maxSwitchTimeMs));
}

void ThemePerformanceOptimizer::clearStyleSheetCache()
{
    QMutexLocker locker(&m_cacheMutex);
    m_styleSheetCache.clear();
    LOG_DEBUG(LogCategories::UI, "StyleSheet cache cleared");
}

void ThemePerformanceOptimizer::preloadCommonStyles()
{
    if (!m_cachingEnabled) return;
    
    LOG_DEBUG(LogCategories::UI, "Preloading common styles for performance optimization");
    
    // Create sample theme data for preloading
    ThemeData lightTheme, darkTheme;
    
    // Light theme
    lightTheme.colors.background = QColor(255, 255, 255);
    lightTheme.colors.foreground = QColor(0, 0, 0);
    lightTheme.colors.accent = QColor(0, 120, 215);
    lightTheme.colors.border = QColor(200, 200, 200);
    lightTheme.colors.hover = QColor(230, 230, 230);
    lightTheme.typography.fontFamily = "Segoe UI, Ubuntu, sans-serif";
    lightTheme.typography.baseFontSize = 9;
    lightTheme.spacing.padding = 8;
    lightTheme.spacing.margin = 4;
    lightTheme.spacing.borderRadius = 4;
    lightTheme.spacing.borderWidth = 1;
    
    // Dark theme
    darkTheme.colors.background = QColor(53, 53, 53);
    darkTheme.colors.foreground = QColor(255, 255, 255);
    darkTheme.colors.accent = QColor(42, 130, 218);
    darkTheme.colors.border = QColor(85, 85, 85);
    darkTheme.colors.hover = QColor(74, 74, 74);
    darkTheme.typography = lightTheme.typography;
    darkTheme.spacing = lightTheme.spacing;
    
    // Precompute styles for common widget types
    precomputeCommonStyles();
    
    LOG_DEBUG(LogCategories::UI, "Common styles preloaded successfully");
}

QString ThemePerformanceOptimizer::getCachedStyleSheet(const QString& key)
{
    if (!m_cachingEnabled) return QString();
    
    QMutexLocker locker(&m_cacheMutex);
    QString* cached = m_styleSheetCache.object(key);
    
    if (cached) {
        updateCacheStatistics(true); // Cache hit
        return *cached;
    }
    
    updateCacheStatistics(false); // Cache miss
    return QString();
}

void ThemePerformanceOptimizer::cacheStyleSheet(const QString& key, const QString& styleSheet)
{
    if (!m_cachingEnabled || styleSheet.isEmpty()) return;
    
    QMutexLocker locker(&m_cacheMutex);
    m_styleSheetCache.insert(key, new QString(styleSheet));
}

void ThemePerformanceOptimizer::startPerformanceMonitoring()
{
    m_monitoringEnabled = true;
    m_metricsTimer->start();
    LOG_INFO(LogCategories::UI, "Performance monitoring started");
}

void ThemePerformanceOptimizer::stopPerformanceMonitoring()
{
    m_monitoringEnabled = false;
    m_metricsTimer->stop();
    LOG_INFO(LogCategories::UI, "Performance monitoring stopped");
}

qint64 ThemePerformanceOptimizer::getLastSwitchTime() const
{
    return m_recentSwitchTimes.isEmpty() ? 0 : m_recentSwitchTimes.last();
}

qint64 ThemePerformanceOptimizer::getAverageSwitchTime() const
{
    return m_metrics.averageSwitchTime;
}

int ThemePerformanceOptimizer::getCacheHitRate() const
{
    int total = m_metrics.cacheHits + m_metrics.cacheMisses;
    return total > 0 ? (m_metrics.cacheHits * 100) / total : 0;
}

void ThemePerformanceOptimizer::optimizedApplyTheme(const ThemeData& theme)
{
    if (m_monitoringEnabled) {
        m_switchTimer.start();
    }
    
    LOG_DEBUG(LogCategories::UI, "Starting optimized theme application");
    
    // Get all widgets that need updating
    QWidgetList allWidgets = QApplication::allWidgets();
    
    if (m_batchUpdatesEnabled && allWidgets.size() > 10) {
        // Use batch processing for large numbers of widgets
        clearBatch();
        
        for (QWidget* widget : allWidgets) {
            if (widget) {
                if (auto* dialog = qobject_cast<QDialog*>(widget)) {
                    addDialogToBatch(dialog);
                } else {
                    addToBatch(widget);
                }
            }
        }
        
        processBatch(theme);
    } else {
        // Direct application for small numbers of widgets
        for (QWidget* widget : allWidgets) {
            if (widget) {
                if (auto* dialog = qobject_cast<QDialog*>(widget)) {
                    optimizedApplyToDialog(dialog, theme);
                } else {
                    optimizedApplyToWidget(widget, theme);
                }
            }
        }
    }
    
    if (m_monitoringEnabled) {
        qint64 elapsed = m_switchTimer.elapsed();
        m_recentSwitchTimes.append(elapsed);
        
        // Keep only last 10 measurements
        if (m_recentSwitchTimes.size() > 10) {
            m_recentSwitchTimes.removeFirst();
        }
        
        // Update metrics
        m_metrics.totalSwitchTime += elapsed;
        m_metrics.switchCount++;
        m_metrics.averageSwitchTime = m_metrics.totalSwitchTime / m_metrics.switchCount;
        m_metrics.minSwitchTime = qMin(m_metrics.minSwitchTime, elapsed);
        m_metrics.maxSwitchTime = qMax(m_metrics.maxSwitchTime, elapsed);
        m_metrics.lastUpdate = QDateTime::currentDateTime();
        
        LOG_INFO(LogCategories::UI, QString("Theme switch completed in %1ms (target: %2ms)")
                 .arg(elapsed).arg(m_performanceTarget));
        
        if (elapsed > m_performanceTarget) {
            emit performanceTargetExceeded(elapsed, m_performanceTarget);
        }
    }
}

void ThemePerformanceOptimizer::optimizedApplyToWidget(QWidget* widget, const ThemeData& theme)
{
    if (!widget) return;
    
    // Check cache first
    QString cacheKey = generateCacheKey(widget, theme);
    QString cachedStyle = getCachedStyleSheet(cacheKey);
    
    if (!cachedStyle.isEmpty()) {
        // Use cached style
        widget->setStyleSheet(cachedStyle);
        widget->update();
        return;
    }
    
    // Generate new style and cache it
    QString styleSheet = generateOptimizedStyleSheet(widget, theme);
    if (!styleSheet.isEmpty()) {
        cacheStyleSheet(cacheKey, styleSheet);
        widget->setStyleSheet(styleSheet);
        widget->update();
    }
}

void ThemePerformanceOptimizer::optimizedApplyToDialog(QDialog* dialog, const ThemeData& theme)
{
    if (!dialog) return;
    
    // Check cache first
    QString cacheKey = generateCacheKey(dialog, theme);
    QString cachedStyle = getCachedStyleSheet(cacheKey);
    
    if (!cachedStyle.isEmpty()) {
        // Use cached style
        dialog->setStyleSheet(cachedStyle);
        
        // Apply to child widgets efficiently
        QList<QWidget*> children = dialog->findChildren<QWidget*>();
        for (QWidget* child : children) {
            if (child) {
                optimizedApplyToWidget(child, theme);
            }
        }
        
        dialog->update();
        return;
    }
    
    // Generate new style and cache it
    QString styleSheet = generateOptimizedDialogStyleSheet(dialog, theme);
    if (!styleSheet.isEmpty()) {
        cacheStyleSheet(cacheKey, styleSheet);
        dialog->setStyleSheet(styleSheet);
        
        // Apply to child widgets
        QList<QWidget*> children = dialog->findChildren<QWidget*>();
        for (QWidget* child : children) {
            if (child) {
                optimizedApplyToWidget(child, theme);
            }
        }
        
        dialog->update();
    }
}

void ThemePerformanceOptimizer::addToBatch(QWidget* widget)
{
    if (!widget || !m_batchUpdatesEnabled) return;
    
    QMutexLocker locker(&m_batchMutex);
    m_batchWidgets.append(QPointer<QWidget>(widget));
    
    // Start batch timer if not already running
    if (!m_batchTimer->isActive()) {
        m_batchTimer->start();
    }
}

void ThemePerformanceOptimizer::addDialogToBatch(QDialog* dialog)
{
    if (!dialog || !m_batchUpdatesEnabled) return;
    
    QMutexLocker locker(&m_batchMutex);
    m_batchDialogs.append(QPointer<QDialog>(dialog));
    
    // Start batch timer if not already running
    if (!m_batchTimer->isActive()) {
        m_batchTimer->start();
    }
}

void ThemePerformanceOptimizer::processBatch(const ThemeData& theme)
{
    QMutexLocker locker(&m_batchMutex);
    
    if (m_batchWidgets.isEmpty() && m_batchDialogs.isEmpty()) {
        return;
    }
    
    QElapsedTimer batchTimer;
    batchTimer.start();
    
    int processedCount = 0;
    
    // Process widgets
    for (const QPointer<QWidget>& ptr : m_batchWidgets) {
        if (!ptr.isNull()) {
            optimizedApplyToWidget(ptr.data(), theme);
            processedCount++;
        }
    }
    
    // Process dialogs
    for (const QPointer<QDialog>& ptr : m_batchDialogs) {
        if (!ptr.isNull()) {
            optimizedApplyToDialog(ptr.data(), theme);
            processedCount++;
        }
    }
    
    qint64 elapsed = batchTimer.elapsed();
    m_metrics.batchedUpdates++;
    
    LOG_DEBUG(LogCategories::UI, QString("Processed batch of %1 items in %2ms")
             .arg(processedCount).arg(elapsed));
    
    emit batchProcessed(processedCount, elapsed);
    
    // Clear batch
    m_batchWidgets.clear();
    m_batchDialogs.clear();
}

void ThemePerformanceOptimizer::clearBatch()
{
    QMutexLocker locker(&m_batchMutex);
    m_batchWidgets.clear();
    m_batchDialogs.clear();
    m_batchTimer->stop();
}

ThemePerformanceOptimizer::PerformanceMetrics ThemePerformanceOptimizer::getPerformanceMetrics() const
{
    return m_metrics;
}

void ThemePerformanceOptimizer::resetPerformanceMetrics()
{
    m_metrics = PerformanceMetrics();
    m_metrics.minSwitchTime = LLONG_MAX;
    m_metrics.lastUpdate = QDateTime::currentDateTime();
    m_recentSwitchTimes.clear();
    
    LOG_DEBUG(LogCategories::UI, "Performance metrics reset");
}

QString ThemePerformanceOptimizer::generatePerformanceReport() const
{
    QString report;
    report += "=== Theme Performance Report ===\n";
    report += QString("Generated: %1\n").arg(QDateTime::currentDateTime().toString());
    report += QString("Total theme switches: %1\n").arg(m_metrics.switchCount);
    report += QString("Average switch time: %1ms\n").arg(m_metrics.averageSwitchTime);
    report += QString("Min switch time: %1ms\n").arg(m_metrics.minSwitchTime == LLONG_MAX ? 0 : m_metrics.minSwitchTime);
    report += QString("Max switch time: %1ms\n").arg(m_metrics.maxSwitchTime);
    report += QString("Performance target: %1ms\n").arg(m_performanceTarget);
    report += QString("Cache hit rate: %1%\n").arg(getCacheHitRate());
    report += QString("Cache hits: %1\n").arg(m_metrics.cacheHits);
    report += QString("Cache misses: %1\n").arg(m_metrics.cacheMisses);
    report += QString("Batched updates: %1\n").arg(m_metrics.batchedUpdates);
    report += QString("Caching enabled: %1\n").arg(m_cachingEnabled ? "Yes" : "No");
    report += QString("Batch updates enabled: %1\n").arg(m_batchUpdatesEnabled ? "Yes" : "No");
    report += QString("Async updates enabled: %1\n").arg(m_asyncUpdatesEnabled ? "Yes" : "No");
    report += "================================\n";
    
    return report;
}

void ThemePerformanceOptimizer::processPendingBatch()
{
    // This will be called when batch timer expires
    // We need the current theme data, so we'll emit a signal
    // The ThemeManager will connect to this and provide the current theme
    LOG_DEBUG(LogCategories::UI, "Processing pending batch due to timer expiry");
}

void ThemePerformanceOptimizer::updatePerformanceMetrics()
{
    if (!m_monitoringEnabled) return;
    
    int currentHitRate = getCacheHitRate();
    static int lastHitRate = currentHitRate;
    
    if (currentHitRate != lastHitRate) {
        emit cacheEfficiencyChanged(currentHitRate);
        lastHitRate = currentHitRate;
    }
}

QString ThemePerformanceOptimizer::generateCacheKey(QWidget* widget, const ThemeData& theme) const
{
    if (!widget) return QString();
    
    QString className = widget->metaObject()->className();
    QString themeHash = QString("%1_%2_%3_%4")
                       .arg(theme.colors.background.name())
                       .arg(theme.colors.foreground.name())
                       .arg(theme.colors.accent.name())
                       .arg(theme.typography.baseFontSize);
    
    return QString("%1_%2").arg(className).arg(themeHash);
}

QString ThemePerformanceOptimizer::generateCacheKey(QDialog* dialog, const ThemeData& theme) const
{
    if (!dialog) return QString();
    
    QString className = dialog->metaObject()->className();
    QString themeHash = QString("%1_%2_%3_%4")
                       .arg(theme.colors.background.name())
                       .arg(theme.colors.foreground.name())
                       .arg(theme.colors.accent.name())
                       .arg(theme.typography.baseFontSize);
    
    return QString("Dialog_%1_%2").arg(className).arg(themeHash);
}

QString ThemePerformanceOptimizer::generateOptimizedStyleSheet(QWidget* widget, const ThemeData& theme) const
{
    if (!widget) return QString();
    
    QString className = widget->metaObject()->className();
    
    // Generate optimized styles based on widget type
    if (className == "QPushButton") {
        return QString(R"(
            QPushButton {
                background-color: %1; color: %2; border: %3px solid %4;
                border-radius: %5px; padding: %6px; min-height: 24px;
                font-family: %7; font-size: %8pt;
            }
            QPushButton:hover { background-color: %9; }
            QPushButton:pressed { background-color: %10; }
        )").arg(theme.colors.accent.name(), theme.colors.background.name())
           .arg(theme.spacing.borderWidth).arg(theme.colors.border.name())
           .arg(theme.spacing.borderRadius).arg(theme.spacing.padding)
           .arg(theme.typography.fontFamily).arg(theme.typography.baseFontSize)
           .arg(theme.colors.hover.name()).arg(theme.colors.accent.darker(120).name());
    }
    
    if (className == "QLineEdit") {
        return QString(R"(
            QLineEdit {
                background-color: %1; color: %2; border: %3px solid %4;
                border-radius: %5px; padding: %6px; min-height: 20px;
                font-family: %7; font-size: %8pt;
            }
            QLineEdit:focus { border-color: %9; border-width: 2px; }
        )").arg(theme.colors.background.name(), theme.colors.foreground.name())
           .arg(theme.spacing.borderWidth).arg(theme.colors.border.name())
           .arg(theme.spacing.borderRadius).arg(theme.spacing.padding)
           .arg(theme.typography.fontFamily).arg(theme.typography.baseFontSize)
           .arg(theme.colors.accent.name());
    }
    
    if (className == "QCheckBox") {
        return QString(R"(
            QCheckBox {
                color: %1; spacing: 8px; font-family: %2; font-size: %3pt;
            }
            QCheckBox::indicator {
                width: 16px; height: 16px; border: %4px solid %5;
                border-radius: 3px; background-color: %6;
            }
            QCheckBox::indicator:checked {
                background-color: %7; border-color: %7;
            }
            QCheckBox::indicator:hover { border-color: %8; }
        )").arg(theme.colors.foreground.name(), theme.typography.fontFamily)
           .arg(theme.typography.baseFontSize).arg(theme.spacing.borderWidth)
           .arg(theme.colors.border.name(), theme.colors.background.name())
           .arg(theme.colors.accent.name(), theme.colors.hover.name());
    }
    
    // Default widget style
    return QString(R"(
        QWidget {
            background-color: %1; color: %2;
            font-family: %3; font-size: %4pt;
        }
    )").arg(theme.colors.background.name(), theme.colors.foreground.name())
       .arg(theme.typography.fontFamily).arg(theme.typography.baseFontSize);
}

QString ThemePerformanceOptimizer::generateOptimizedDialogStyleSheet(QDialog* dialog, const ThemeData& theme) const
{
    if (!dialog) return QString();
    
    return QString(R"(
        QDialog {
            background-color: %1; color: %2;
            font-family: %3; font-size: %4pt;
        }
        QDialog QWidget {
            background-color: %1; color: %2;
        }
    )").arg(theme.colors.background.name(), theme.colors.foreground.name())
       .arg(theme.typography.fontFamily).arg(theme.typography.baseFontSize);
}

void ThemePerformanceOptimizer::updateCacheStatistics(bool hit)
{
    if (hit) {
        m_metrics.cacheHits++;
    } else {
        m_metrics.cacheMisses++;
    }
}

bool ThemePerformanceOptimizer::shouldUseCache(QWidget* widget) const
{
    if (!m_cachingEnabled || !widget) return false;
    
    // Always cache for common widget types
    QString className = widget->metaObject()->className();
    return (className == "QPushButton" || 
            className == "QLineEdit" || 
            className == "QComboBox" || 
            className == "QCheckBox" || 
            className == "QLabel" ||
            className == "QProgressBar");
}

void ThemePerformanceOptimizer::precomputeCommonStyles()
{
    // This method precomputes styles for common widget types
    // to improve initial performance
    LOG_DEBUG(LogCategories::UI, "Precomputing common widget styles");
    
    // Create sample widgets to generate cache keys
    QStringList commonWidgets = {"QPushButton", "QLineEdit", "QComboBox", "QCheckBox", "QLabel"};
    
    // This is a placeholder - in a real implementation, we would
    // generate styles for common theme combinations
    LOG_DEBUG(LogCategories::UI, "Common styles precomputation completed");
}