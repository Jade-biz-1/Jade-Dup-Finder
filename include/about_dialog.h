#ifndef ABOUT_DIALOG_H
#define ABOUT_DIALOG_H

#include <QtWidgets/QDialog>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QTabWidget>

/**
 * @brief About dialog showing application information
 * 
 * Displays:
 * - Application name and version
 * - Copyright and license information
 * - Author/contributors information
 * - System information
 * - Third-party libraries and credits
 */
class AboutDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AboutDialog(QWidget* parent = nullptr);
    ~AboutDialog() override = default;

private:
    void initializeUI();
    void setupConnections();
    void applyTheme();
    
    QString getVersionInfo() const;
    QString getLicenseInfo() const;
    QString getAuthorsInfo() const;
    QString getSystemInfo() const;
    QString getLibrariesInfo() const;
    
    // UI Components
    QVBoxLayout* m_mainLayout;
    QHBoxLayout* m_headerLayout;
    QLabel* m_iconLabel;
    QLabel* m_titleLabel;
    QLabel* m_versionLabel;
    QTabWidget* m_tabWidget;
    QTextBrowser* m_aboutTab;
    QTextBrowser* m_licenseTab;
    QTextBrowser* m_authorsTab;
    QTextBrowser* m_systemTab;
    QTextBrowser* m_creditsTab;
    QPushButton* m_closeButton;
    
    static constexpr const char* APP_VERSION = "1.0.0";
    static constexpr const char* BUILD_DATE = __DATE__;
};

#endif // ABOUT_DIALOG_H
