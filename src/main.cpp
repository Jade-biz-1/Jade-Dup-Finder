#include <QApplication>
#include <QMessageBox>
#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QDebug>

/**
 * @brief DupFinder - Cross-Platform Duplicate File Finder
 * 
 * This is the main entry point for Phase 1 development.
 * Currently shows a simple placeholder UI to verify the build system works.
 * 
 * Next steps:
 * 1. Implement core duplicate detection algorithms
 * 2. Build proper Qt6 GUI interface
 * 3. Add platform-specific file operations
 */

class DupFinderMainWindow : public QWidget {
    Q_OBJECT

public:
    DupFinderMainWindow(QWidget* parent = nullptr) : QWidget(parent) {
        setWindowTitle("DupFinder v1.0 - Development Build");
        setFixedSize(500, 300);
        
        setupUI();
    }

private slots:
    void showAbout() {
        QMessageBox::about(this, "About DupFinder",
            "<h2>DupFinder v1.0</h2>"
            "<p>Cross-platform duplicate file finder</p>"
            "<p><b>Development Phase:</b> Phase 1 - Foundation</p>"
            "<p><b>Platform:</b> Linux (Ubuntu)</p>"
            "<p><b>Framework:</b> Qt " QT_VERSION_STR "</p>"
            "<p><b>Build:</b> " + QString(__DATE__) + " " + QString(__TIME__) + "</p>"
            "<hr>"
            "<p><b>Next Development Steps:</b></p>"
            "<ul>"
            "<li>Implement file scanning engine</li>"
            "<li>Add duplicate detection algorithms</li>"
            "<li>Build comprehensive GUI</li>"
            "<li>Add safety and recovery features</li>"
            "</ul>");
    }

private:
    void setupUI() {
        auto layout = new QVBoxLayout(this);
        
        // Header
        auto titleLabel = new QLabel("<h1>🔍 DupFinder</h1>");
        titleLabel->setAlignment(Qt::AlignCenter);
        titleLabel->setStyleSheet("color: #2c3e50; margin: 20px;");
        
        // Status
        auto statusLabel = new QLabel("✅ <b>Build System Ready!</b>");
        statusLabel->setAlignment(Qt::AlignCenter);
        statusLabel->setStyleSheet("color: #27ae60; font-size: 14px; margin: 10px;");
        
        // Development info
        auto devInfo = new QLabel(
            "<p><b>Development Environment:</b> Ready</p>"
            "<p><b>Qt6 Version:</b> " QT_VERSION_STR "</p>"
            "<p><b>Platform:</b> Linux</p>"
            "<p><b>Phase:</b> 1 - Foundation</p>");
        devInfo->setAlignment(Qt::AlignCenter);
        devInfo->setStyleSheet("background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px;");
        
        // Action button
        auto aboutButton = new QPushButton("📋 View Development Roadmap");
        aboutButton->setStyleSheet(
            "QPushButton {"
            "  background: #3498db; color: white; border: none;"
            "  padding: 10px 20px; border-radius: 5px; font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "  background: #2980b9;"
            "}");
        
        connect(aboutButton, &QPushButton::clicked, this, &DupFinderMainWindow::showAbout);
        
        layout->addWidget(titleLabel);
        layout->addWidget(statusLabel);
        layout->addWidget(devInfo);
        layout->addWidget(aboutButton);
        layout->addStretch();
    }
};

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    // Application metadata
    app.setApplicationName("DupFinder");
    app.setApplicationVersion("1.0.0-dev");
    app.setOrganizationName("DupFinder Team");
    
    qDebug() << "DupFinder v1.0 starting...";
    qDebug() << "Qt version:" << QT_VERSION_STR;
    qDebug() << "Platform: Linux";
    qDebug() << "Phase: 1 - Foundation Development";
    
    DupFinderMainWindow window;
    window.show();
    
    qDebug() << "Application UI loaded successfully!";
    
    return app.exec();
}

// Include the MOC file for Qt's meta-object system
#include "main.moc"