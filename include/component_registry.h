#ifndef COMPONENT_REGISTRY_H
#define COMPONENT_REGISTRY_H

#include <QObject>
#include <QWidget>
#include <QDialog>
#include <QMap>
#include <QList>
#include <QPointer>
#include <QMutex>
#include <QDateTime>
#include <QTimer>
#include "theme_manager.h"

// Forward declaration to avoid circular dependency
struct ThemeData;

class ComponentRegistry : public QObject
{
    Q_OBJECT
    
public:
    explicit ComponentRegistry(QObject* parent = nullptr);
    ~ComponentRegistry();
    
    // Component registration
    void registerComponent(QWidget* component, ThemeManager::ComponentType type);
    void unregisterComponent(QWidget* component);
    void registerDialog(QDialog* dialog);
    void unregisterDialog(QDialog* dialog);
    
    // Component retrieval
    QList<QWidget*> getComponentsByType(ThemeManager::ComponentType type) const;
    QList<QWidget*> getAllComponents() const;
    QList<QDialog*> getAllDialogs() const;
    
    // Theme application
    void applyThemeToAll(const ThemeData& theme);
    void applyThemeToComponent(QWidget* component, const ThemeData& theme);
    void applyThemeToDialog(QDialog* dialog, const ThemeData& theme);
    
    // Real-time theme update notification system
    void enableRealTimeUpdates(bool enabled = true);
    void setUpdateInterval(int milliseconds = 100);
    void forceUpdateAll();
    
    // Component validation and maintenance
    void validateAllComponents();
    void cleanupInvalidComponents();
    int getComponentCount() const;
    int getDialogCount() const;
    int getComponentCountByType(ThemeManager::ComponentType type) const;
    
    // Error recovery
    void attemptRecovery();
    QStringList getFailedComponents() const;
    void retryFailedComponents(const ThemeData& theme);
    
    // Component monitoring
    void startMonitoring();
    void stopMonitoring();
    bool isMonitoring() const;
    
signals:
    void componentRegistered(QWidget* component, ThemeManager::ComponentType type);
    void componentUnregistered(QWidget* component);
    void dialogRegistered(QDialog* dialog);
    void dialogUnregistered(QDialog* dialog);
    void themeAppliedToAll();
    void themeApplicationFailed(QWidget* component, const QString& error);
    void componentValidationCompleted(int validCount, int totalCount);
    void realTimeUpdateTriggered();
    
private slots:
    void onComponentDestroyed(QObject* obj);
    void onDialogDestroyed(QObject* obj);
    void onRealTimeUpdate();
    void onMonitoringUpdate();
    
private:
    struct ComponentInfo {
        QPointer<QWidget> widget;
        ThemeManager::ComponentType type;
        QDateTime registered;
        QDateTime lastUpdated;
        bool isValid;
        bool updateFailed;
        QString objectName;
        QString className;
        QString lastError;
        int failureCount;
    };
    
    struct DialogInfo {
        QPointer<QDialog> dialog;
        QDateTime registered;
        QDateTime lastUpdated;
        bool isValid;
        bool updateFailed;
        QString objectName;
        QString className;
        QString lastError;
        int failureCount;
    };
    
    QString generateComponentStyleSheet(ThemeManager::ComponentType type, const ThemeData& theme) const;
    QString generateDialogStyleSheet(const ThemeData& theme) const;
    ThemeManager::ComponentType determineComponentType(QWidget* widget) const;
    void updateComponentInfo(QWidget* component, bool success, const QString& error = QString());
    void updateDialogInfo(QDialog* dialog, bool success, const QString& error = QString());
    
    QMap<QWidget*, ComponentInfo> m_components;
    QMap<QDialog*, DialogInfo> m_dialogs;
    mutable QMutex m_mutex;
    
    // Real-time update system
    QTimer* m_realTimeUpdateTimer;
    QTimer* m_monitoringTimer;
    bool m_realTimeUpdatesEnabled;
    bool m_monitoringEnabled;
    int m_updateInterval;
    
    // Error tracking
    QStringList m_failedComponents;
    QDateTime m_lastCleanup;
};

#endif // COMPONENT_REGISTRY_H