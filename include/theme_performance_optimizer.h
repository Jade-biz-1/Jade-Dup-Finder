#ifndef THEME_PERFORMANCE_OPTIMIZER_H
#define THEME_PERFORMANCE_OPTIMIZER_H

#include <QObject>
#include <QTimer>
#include <QElapsedTimer>
#include <QCache>
#include <QMutex>
#include <QHash>
#include <QDateTime>
#include <QWidget>
#include <QDialog>
#include "theme_manager.h"

class ThemePerformanceOptimizer : public QObject
{
    Q_OBJECT
    
public:
    explicit ThemePerformanceOptimizer(QObject* parent = nullptr);
    ~ThemePerformanceOptimizer();
    
    // Performance optimization settings
    void enableStyleSheetCaching(bool enabled = true);
    void enableBatchUpdates(bool enabled = true);
    void enableAsyncUpdates(bool enabled = true);
    void setMaxCacheSize(int maxItems = 1000);
    void setBatchUpdateInterval(int milliseconds = 50);
    void setPerformanceTarget(int maxSwitchTimeMs = 100);
    
    // Cache management
    void clearStyleSheetCache();
    void preloadCommonStyles();
    QString getCachedStyleSheet(const QString& key);
    void cacheStyleSheet(const QString& key, const QString& styleSheet);
    
    // Performance monitoring
    void startPerformanceMonitoring();
    void stopPerformanceMonitoring();
    qint64 getLastSwitchTime() const;
    qint64 getAverageSwitchTime() const;
    int getCacheHitRate() const;
    
    // Optimized theme application
    void optimizedApplyTheme(const ThemeData& theme);
    void optimizedApplyToWidget(QWidget* widget, const ThemeData& theme);
    void optimizedApplyToDialog(QDialog* dialog, const ThemeData& theme);
    
    // Batch operations
    void addToBatch(QWidget* widget);
    void addDialogToBatch(QDialog* dialog);
    void processBatch(const ThemeData& theme);
    void clearBatch();
    
    // Performance analysis
    struct PerformanceMetrics {
        qint64 totalSwitchTime;
        qint64 averageSwitchTime;
        qint64 minSwitchTime;
        qint64 maxSwitchTime;
        int switchCount;
        int cacheHits;
        int cacheMisses;
        int batchedUpdates;
        QDateTime lastUpdate;
    };
    
    PerformanceMetrics getPerformanceMetrics() const;
    void resetPerformanceMetrics();
    QString generatePerformanceReport() const;
    
signals:
    void performanceTargetExceeded(qint64 actualTime, int targetTime);
    void cacheEfficiencyChanged(int hitRate);
    void batchProcessed(int itemCount, qint64 processingTime);
    
private slots:
    void processPendingBatch();
    void updatePerformanceMetrics();
    
private:
    // Cache system
    QCache<QString, QString> m_styleSheetCache;
    QMutex m_cacheMutex;
    bool m_cachingEnabled;
    
    // Batch update system
    QList<QPointer<QWidget>> m_batchWidgets;
    QList<QPointer<QDialog>> m_batchDialogs;
    QTimer* m_batchTimer;
    QMutex m_batchMutex;
    bool m_batchUpdatesEnabled;
    int m_batchInterval;
    
    // Performance monitoring
    QElapsedTimer m_switchTimer;
    QTimer* m_metricsTimer;
    PerformanceMetrics m_metrics;
    QList<qint64> m_recentSwitchTimes;
    bool m_monitoringEnabled;
    int m_performanceTarget;
    
    // Async update system
    bool m_asyncUpdatesEnabled;
    
    // Helper methods
    QString generateCacheKey(QWidget* widget, const ThemeData& theme) const;
    QString generateCacheKey(QDialog* dialog, const ThemeData& theme) const;
    QString generateOptimizedStyleSheet(QWidget* widget, const ThemeData& theme) const;
    QString generateOptimizedDialogStyleSheet(QDialog* dialog, const ThemeData& theme) const;
    void updateCacheStatistics(bool hit);
    bool shouldUseCache(QWidget* widget) const;
    void precomputeCommonStyles();
};

#endif // THEME_PERFORMANCE_OPTIMIZER_H