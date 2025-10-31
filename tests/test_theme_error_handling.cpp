#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QMessageBox>
#include "include/theme_error_handler.h"
#include "include/theme_manager.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    // Create a simple test window
    QWidget window;
    window.setWindowTitle("Theme Error Handler Test");
    window.resize(400, 300);
    
    QVBoxLayout* layout = new QVBoxLayout(&window);
    
    // Test buttons for different error scenarios
    QPushButton* testThemeLoadError = new QPushButton("Test Theme Load Error");
    QPushButton* testStyleApplicationError = new QPushButton("Test Style Application Error");
    QPushButton* testCustomThemeCorruption = new QPushButton("Test Custom Theme Corruption");
    QPushButton* testRecoveryDialog = new QPushButton("Test Recovery Dialog");
    
    layout->addWidget(testThemeLoadError);
    layout->addWidget(testStyleApplicationError);
    layout->addWidget(testCustomThemeCorruption);
    layout->addWidget(testRecoveryDialog);
    
    // Connect test buttons to error scenarios
    QObject::connect(testThemeLoadError, &QPushButton::clicked, []() {
        ThemeErrorHandler::handleError(
            ThemeErrorHandler::ErrorType::ThemeLoadFailure,
            "Test theme load failure - this is a simulated error for testing",
            nullptr
        );
    });
    
    QObject::connect(testStyleApplicationError, &QPushButton::clicked, []() {
        ThemeErrorHandler::handleError(
            ThemeErrorHandler::ErrorType::StyleApplicationFailure,
            "Test style application failure - simulated error",
            nullptr
        );
    });
    
    QObject::connect(testCustomThemeCorruption, &QPushButton::clicked, []() {
        ThemeErrorHandler::handleError(
            ThemeErrorHandler::ErrorType::CustomThemeCorruption,
            "Test custom theme corruption - simulated error",
            nullptr
        );
    });
    
    QObject::connect(testRecoveryDialog, &QPushButton::clicked, []() {
        bool result = ThemeErrorHandler::showRecoveryDialog(
            ThemeErrorHandler::ErrorType::ThemeLoadFailure,
            "Test recovery dialog - this is a simulated critical error"
        );
        
        QMessageBox::information(nullptr, "Recovery Result", 
            result ? "Recovery was successful" : "Recovery was cancelled or failed");
    });
    
    // Initialize ThemeManager
    ThemeManager* themeManager = ThemeManager::instance();
    themeManager->applyToWidget(&window);
    
    window.show();
    
    return app.exec();
}