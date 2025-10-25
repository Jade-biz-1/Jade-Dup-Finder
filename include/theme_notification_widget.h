#ifndef THEME_NOTIFICATION_WIDGET_H
#define THEME_NOTIFICATION_WIDGET_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTimer>
#include <QPropertyAnimation>
#include <QGraphicsOpacityEffect>
#include "theme_error_handler.h"

class ThemeNotificationWidget : public QWidget
{
    Q_OBJECT

public:
    enum class NotificationType {
        Info,
        Warning,
        Error,
        Success
    };

    explicit ThemeNotificationWidget(QWidget* parent = nullptr);
    ~ThemeNotificationWidget() = default;

    void showNotification(const QString& message, 
                         NotificationType type = NotificationType::Info,
                         int duration = 5000);
    
    void showThemeError(ThemeErrorHandler::ErrorType errorType, 
                       const QString& message,
                       int duration = 8000);

public slots:
    void hideNotification();

private slots:
    void onActionClicked();
    void onCloseClicked();

private:
    void setupUI();
    void updateStyleForType(NotificationType type);
    void animateIn();
    void animateOut();
    QString getIconForType(NotificationType type) const;
    QString getActionTextForErrorType(ThemeErrorHandler::ErrorType errorType) const;

    // UI Components
    QHBoxLayout* m_mainLayout;
    QLabel* m_iconLabel;
    QVBoxLayout* m_contentLayout;
    QLabel* m_messageLabel;
    QHBoxLayout* m_buttonLayout;
    QPushButton* m_actionButton;
    QPushButton* m_closeButton;
    
    // Animation
    QPropertyAnimation* m_fadeAnimation;
    QGraphicsOpacityEffect* m_opacityEffect;
    QTimer* m_hideTimer;
    
    // Data
    NotificationType m_currentType;
    ThemeErrorHandler::ErrorType m_currentErrorType;
    QString m_currentMessage;
    bool m_isVisible;
};

#endif // THEME_NOTIFICATION_WIDGET_H