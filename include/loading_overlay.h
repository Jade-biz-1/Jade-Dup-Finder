#ifndef LOADING_OVERLAY_H
#define LOADING_OVERLAY_H

#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QVBoxLayout>
#include <QPainter>
#include <QPropertyAnimation>

/**
 * @brief Loading overlay widget with animated spinner
 * 
 * Shows a semi-transparent overlay with a spinning indicator and message
 * to indicate background processing is occurring.
 */
class LoadingOverlay : public QWidget
{
    Q_OBJECT
    Q_PROPERTY(int rotation READ rotation WRITE setRotation)

public:
    explicit LoadingOverlay(QWidget* parent = nullptr);
    
    void show(const QString& message = QString());
    void hide();
    void setMessage(const QString& message);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    int rotation() const { return m_rotation; }
    void setRotation(int angle) { m_rotation = angle; update(); }
    
    QLabel* m_messageLabel;
    QPropertyAnimation* m_animation;
    int m_rotation;
    int m_spinnerSize;
};

#endif // LOADING_OVERLAY_H
