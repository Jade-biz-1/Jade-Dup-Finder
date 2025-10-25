#include "theme_notification_widget.h"
#include "theme_recovery_dialog.h"
#include "theme_manager.h"
#include "core/logger.h"
#include <QApplication>
#include <QStyle>

#include <QScreen>

ThemeNotificationWidget::ThemeNotificationWidget(QWidget* parent)
    : QWidget(parent)
    , m_currentType(NotificationType::Info)
    , m_currentErrorType(ThemeErrorHandler::ErrorType::ValidationFailure)
    , m_isVisible(false)
{
    setupUI();
    
    // Set up animation
    m_opacityEffect = new QGraphicsOpacityEffect(this);
    setGraphicsEffect(m_opacityEffect);
    
    m_fadeAnimation = new QPropertyAnimation(m_opacityEffect, "opacity", this);
    m_fadeAnimation->setDuration(300);
    
    // Set up auto-hide timer
    m_hideTimer = new QTimer(this);
    m_hideTimer->setSingleShot(true);
    connect(m_hideTimer, &QTimer::timeout, this, &ThemeNotificationWidget::hideNotification);
    
    // Initially hidden
    setVisible(false);
    m_opacityEffect->setOpacity(0.0);
    
    // Position at top-right of screen
    setWindowFlags(Qt::Tool | Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint);
    setAttribute(Qt::WA_TranslucentBackground);
    
    LOG_DEBUG(LogCategories::UI, "Theme notification widget initialized");
}

void ThemeNotificationWidget::setupUI()
{
    setFixedSize(400, 80);
    
    m_mainLayout = new QHBoxLayout(this);
    m_mainLayout->setContentsMargins(12, 8, 12, 8);
    m_mainLayout->setSpacing(12);
    
    // Icon
    m_iconLabel = new QLabel();
    m_iconLabel->setFixedSize(32, 32);
    m_iconLabel->setAlignment(Qt::AlignCenter);
    m_mainLayout->addWidget(m_iconLabel);
    
    // Content
    m_contentLayout = new QVBoxLayout();
    m_contentLayout->setSpacing(4);
    
    m_messageLabel = new QLabel();
    m_messageLabel->setWordWrap(true);
    // Apply theme-aware label styling
    ThemeManager::instance()->applyToWidget(m_messageLabel);
    m_contentLayout->addWidget(m_messageLabel);
    
    m_buttonLayout = new QHBoxLayout();
    m_buttonLayout->setSpacing(8);
    
    m_actionButton = new QPushButton();
    // Apply theme-aware button styling
    ThemeManager::instance()->applyToWidget(m_actionButton);
    m_actionButton->setMinimumSize(ThemeManager::instance()->getMinimumControlSize(ThemeManager::ControlType::Button));
    connect(m_actionButton, &QPushButton::clicked, this, &ThemeNotificationWidget::onActionClicked);
    m_buttonLayout->addWidget(m_actionButton);
    
    m_buttonLayout->addStretch();
    m_contentLayout->addLayout(m_buttonLayout);
    
    m_mainLayout->addLayout(m_contentLayout, 1);
    
    // Close button
    m_closeButton = new QPushButton("×");
    m_closeButton->setFixedSize(20, 20);
    // Apply theme-aware close button styling
    ThemeManager::instance()->applyToWidget(m_closeButton);
    m_closeButton->setFlat(true);  // Make it flat/borderless
    connect(m_closeButton, &QPushButton::clicked, this, &ThemeNotificationWidget::onCloseClicked);
    m_mainLayout->addWidget(m_closeButton);
    
    // Apply theme-aware widget styling
    ThemeManager::instance()->applyToWidget(this);
}

void ThemeNotificationWidget::showNotification(const QString& message, NotificationType type, int duration)
{
    m_currentType = type;
    m_currentMessage = message;
    
    // Update content
    m_messageLabel->setText(message);
    m_iconLabel->setPixmap(style()->standardIcon(QStyle::SP_MessageBoxInformation).pixmap(24, 24));
    
    // Update styling
    updateStyleForType(type);
    
    // Hide action button for generic notifications
    m_actionButton->setVisible(false);
    
    // Position and show
    QScreen* screen = QApplication::primaryScreen();
    if (screen) {
        QRect screenGeometry = screen->availableGeometry();
        move(screenGeometry.right() - width() - 20, screenGeometry.top() + 20);
    }
    
    animateIn();
    
    // Set auto-hide timer
    if (duration > 0) {
        m_hideTimer->start(duration);
    }
    
    LOG_INFO(LogCategories::UI, QString("Showing notification: %1").arg(message));
}

void ThemeNotificationWidget::showThemeError(ThemeErrorHandler::ErrorType errorType, const QString& message, int duration)
{
    m_currentErrorType = errorType;
    m_currentMessage = message;
    
    // Determine notification type based on error severity
    NotificationType notificationType;
    switch (errorType) {
        case ThemeErrorHandler::ErrorType::ThemeLoadFailure:
        case ThemeErrorHandler::ErrorType::CustomThemeCorruption:
            notificationType = NotificationType::Error;
            break;
        case ThemeErrorHandler::ErrorType::StyleApplicationFailure:
        case ThemeErrorHandler::ErrorType::ComponentRegistrationFailure:
            notificationType = NotificationType::Warning;
            break;
        default:
            notificationType = NotificationType::Info;
            break;
    }
    
    m_currentType = notificationType;
    
    // Update content
    QString displayMessage = QString("Theme Error: %1").arg(message);
    m_messageLabel->setText(displayMessage);
    
    // Set appropriate icon
    QStyle::StandardPixmap iconType;
    switch (notificationType) {
        case NotificationType::Error:
            iconType = QStyle::SP_MessageBoxCritical;
            break;
        case NotificationType::Warning:
            iconType = QStyle::SP_MessageBoxWarning;
            break;
        default:
            iconType = QStyle::SP_MessageBoxInformation;
            break;
    }
    m_iconLabel->setPixmap(style()->standardIcon(iconType).pixmap(24, 24));
    
    // Update styling
    updateStyleForType(notificationType);
    
    // Show action button for theme errors
    m_actionButton->setVisible(true);
    m_actionButton->setText(getActionTextForErrorType(errorType));
    
    // Position and show
    QScreen* screen = QApplication::primaryScreen();
    if (screen) {
        QRect screenGeometry = screen->availableGeometry();
        move(screenGeometry.right() - width() - 20, screenGeometry.top() + 20);
    }
    
    animateIn();
    
    // Set auto-hide timer (longer for errors)
    if (duration > 0) {
        m_hideTimer->start(duration);
    }
    
    LOG_INFO(LogCategories::UI, QString("Showing theme error notification: %1").arg(message));
}

void ThemeNotificationWidget::hideNotification()
{
    if (m_isVisible) {
        animateOut();
    }
}

void ThemeNotificationWidget::onActionClicked()
{
    LOG_INFO(LogCategories::UI, "User clicked notification action button");
    
    // Hide notification first
    hideNotification();
    
    // Show recovery dialog for theme errors
    ThemeRecoveryDialog dialog(m_currentMessage, m_currentErrorType);
    dialog.exec();
}

void ThemeNotificationWidget::onCloseClicked()
{
    LOG_DEBUG(LogCategories::UI, "User closed notification");
    hideNotification();
}

void ThemeNotificationWidget::updateStyleForType(NotificationType type)
{
    // Get theme data for color scheme
    ThemeData themeData = ThemeManager::instance()->getCurrentThemeData();
    
    QString backgroundColor, borderColor, textColor;
    
    switch (type) {
        case NotificationType::Error:
            backgroundColor = themeData.colors.error.name();
            borderColor = themeData.colors.error.darker(120).name();
            textColor = themeData.colors.error.darker(150).name();
            break;
        case NotificationType::Warning:
            backgroundColor = themeData.colors.warning.name();
            borderColor = themeData.colors.warning.darker(120).name();
            textColor = themeData.colors.warning.darker(150).name();
            break;
        case NotificationType::Success:
            backgroundColor = themeData.colors.success.name();
            borderColor = themeData.colors.success.darker(120).name();
            textColor = themeData.colors.success.darker(150).name();
            break;
        case NotificationType::Info:
        default:
            backgroundColor = themeData.colors.info.name();
            borderColor = themeData.colors.info.darker(120).name();
            textColor = themeData.colors.info.darker(150).name();
            break;
    }
    
    setStyleSheet(QString("QWidget { background-color: %1; border: 2px solid %2; border-radius: 6px; }")
                  .arg(backgroundColor, borderColor));
    
    // Apply theme-aware text color
    m_messageLabel->setStyleSheet(QString("color: %1;").arg(textColor));
}

void ThemeNotificationWidget::animateIn()
{
    if (m_isVisible) return;
    
    m_isVisible = true;
    setVisible(true);
    
    m_fadeAnimation->setStartValue(0.0);
    m_fadeAnimation->setEndValue(1.0);
    m_fadeAnimation->start();
    
    LOG_DEBUG(LogCategories::UI, "Animating notification in");
}

void ThemeNotificationWidget::animateOut()
{
    if (!m_isVisible) return;
    
    m_fadeAnimation->setStartValue(1.0);
    m_fadeAnimation->setEndValue(0.0);
    
    connect(m_fadeAnimation, &QPropertyAnimation::finished, this, [this]() {
        setVisible(false);
        m_isVisible = false;
        m_fadeAnimation->disconnect();
        LOG_DEBUG(LogCategories::UI, "Notification hidden");
    }, Qt::UniqueConnection);
    
    m_fadeAnimation->start();
    
    LOG_DEBUG(LogCategories::UI, "Animating notification out");
}

QString ThemeNotificationWidget::getActionTextForErrorType(ThemeErrorHandler::ErrorType errorType) const
{
    switch (errorType) {
        case ThemeErrorHandler::ErrorType::ThemeLoadFailure:
        case ThemeErrorHandler::ErrorType::CustomThemeCorruption:
            return "Fix Now";
        case ThemeErrorHandler::ErrorType::StyleApplicationFailure:
            return "Retry";
        case ThemeErrorHandler::ErrorType::ComponentRegistrationFailure:
        case ThemeErrorHandler::ErrorType::DialogRegistrationFailure:
            return "Restart";
        case ThemeErrorHandler::ErrorType::ValidationFailure:
            return "Details";
        case ThemeErrorHandler::ErrorType::PersistenceFailure:
            return "Check Settings";
        case ThemeErrorHandler::ErrorType::SystemThemeDetectionFailure:
            return "Manual Setup";
        default:
            return "Fix";
    }
}