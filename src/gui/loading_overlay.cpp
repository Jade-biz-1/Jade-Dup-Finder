#include "loading_overlay.h"
#include "theme_manager.h"
#include <QPainter>
#include <QPainterPath>
#include <QResizeEvent>

LoadingOverlay::LoadingOverlay(QWidget* parent)
    : QWidget(parent)
    , m_messageLabel(nullptr)
    , m_animation(nullptr)
    , m_rotation(0)
    , m_spinnerSize(64)
{
    // Make overlay semi-transparent and on top
    setAttribute(Qt::WA_TransparentForMouseEvents, false);
    setAttribute(Qt::WA_TranslucentBackground, false);
    setAutoFillBackground(true);
    
    // Create layout
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setAlignment(Qt::AlignCenter);
    
    // Add spacing for spinner (drawn in paintEvent)
    layout->addSpacing(m_spinnerSize + 20);
    
    // Message label
    m_messageLabel = new QLabel(tr("Processing..."), this);
    m_messageLabel->setAlignment(Qt::AlignCenter);
    QFont font = m_messageLabel->font();
    font.setPointSize(font.pointSize() + 2);
    font.setBold(true);
    m_messageLabel->setFont(font);
    layout->addWidget(m_messageLabel);
    
    // Create rotation animation
    m_animation = new QPropertyAnimation(this, "rotation", this);
    m_animation->setDuration(1000); // 1 second per rotation
    m_animation->setStartValue(0);
    m_animation->setEndValue(360);
    m_animation->setLoopCount(-1); // Infinite loop
    
    // Apply theme
    ThemeData theme = ThemeManager::instance()->getCurrentThemeData();
    QPalette pal = palette();
    pal.setColor(QPalette::Window, theme.colors.background);
    pal.setColor(QPalette::WindowText, theme.colors.foreground);
    setPalette(pal);
    m_messageLabel->setPalette(pal);
    
    // Start hidden
    QWidget::hide();
}

void LoadingOverlay::show(const QString& message)
{
    if (!message.isEmpty()) {
        m_messageLabel->setText(message);
    }
    
    // Resize to cover parent
    if (parentWidget()) {
        setGeometry(parentWidget()->rect());
        raise(); // Bring to front
    }
    
    // Start animation
    m_animation->start();
    
    QWidget::show();
    
    // Force immediate repaint
    QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

void LoadingOverlay::hide()
{
    m_animation->stop();
    QWidget::hide();
}

void LoadingOverlay::setMessage(const QString& message)
{
    m_messageLabel->setText(message);
    QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

void LoadingOverlay::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Draw semi-transparent background
    ThemeData theme = ThemeManager::instance()->getCurrentThemeData();
    QColor bgColor = theme.colors.background;
    bgColor.setAlpha(220); // Semi-transparent
    painter.fillRect(rect(), bgColor);
    
    // Draw spinner in center
    int centerX = width() / 2;
    int centerY = height() / 2 - 40; // Offset up a bit for message below
    
    painter.save();
    painter.translate(centerX, centerY);
    painter.rotate(m_rotation);
    
    // Draw circular spinner with arc
    QPen pen(theme.colors.accent, 6, Qt::SolidLine, Qt::RoundCap);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    
    int radius = m_spinnerSize / 2;
    QRect spinnerRect(-radius, -radius, m_spinnerSize, m_spinnerSize);
    
    // Draw arc (270 degrees, leaving 90 degree gap)
    painter.drawArc(spinnerRect, 0 * 16, 270 * 16);
    
    painter.restore();
}

void LoadingOverlay::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    
    // Keep overlay covering entire parent
    if (parentWidget() && isVisible()) {
        setGeometry(parentWidget()->rect());
    }
}
