#include "about_dialog.h"
#include "theme_manager.h"
// #include "ui_enhancements.h"  // Obsolete test include - removed
#include "logger.h"
#include <QtWidgets/QApplication>
#include <QtCore/QSysInfo>
#include <QtCore/QDateTime>
#include <QtCore/QStandardPaths>
#include <QtGui/QPixmap>
#include <QtGui/QIcon>

AboutDialog::AboutDialog(QWidget* parent)
    : QDialog(parent)
    , m_mainLayout(nullptr)
    , m_headerLayout(nullptr)
    , m_iconLabel(nullptr)
    , m_titleLabel(nullptr)
    , m_versionLabel(nullptr)
    , m_tabWidget(nullptr)
    , m_aboutTab(nullptr)
    , m_licenseTab(nullptr)
    , m_authorsTab(nullptr)
    , m_systemTab(nullptr)
    , m_creditsTab(nullptr)
    , m_closeButton(nullptr)
{
    setWindowTitle(tr("About DupFinder"));
    setMinimumSize(500, 400);
    resize(600, 500);
    
    initializeUI();
    setupConnections();
    applyTheme();
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(this);
    
    // Apply UI enhancements (Section 1.5) - handled by ThemeManager
    // UIEnhancements::setupLogicalTabOrder(this);  // Not implemented
    // UIEnhancements::setupEscapeKeyHandler(this);  // Not implemented
    // UIEnhancements::setupEnterKeyHandler(this);  // Not implemented
    // UIEnhancements::applyConsistentSpacing(this);  // Not implemented
}

void AboutDialog::initializeUI()
{
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(20, 20, 20, 20);
    m_mainLayout->setSpacing(12);
    
    // Header with icon and title
    m_headerLayout = new QHBoxLayout();
    m_headerLayout->setSpacing(16);
    
    m_iconLabel = new QLabel(this);
    QPixmap icon(":/icons/app-icon.png");
    if (!icon.isNull()) {
        m_iconLabel->setPixmap(icon.scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    } else {
        // Fallback to text if icon not found
        m_iconLabel->setText("📁");
        QFont iconFont = m_iconLabel->font();
        iconFont.setPointSize(48);
        m_iconLabel->setFont(iconFont);
    }
    m_iconLabel->setAlignment(Qt::AlignCenter);
    
    QVBoxLayout* titleLayout = new QVBoxLayout();
    
    m_titleLabel = new QLabel(tr("DupFinder"), this);
    QFont titleFont = m_titleLabel->font();
    titleFont.setPointSize(titleFont.pointSize() + 8);
    titleFont.setBold(true);
    m_titleLabel->setFont(titleFont);
    
    m_versionLabel = new QLabel(getVersionInfo(), this);
    QFont versionFont = m_versionLabel->font();
    versionFont.setPointSize(versionFont.pointSize() + 2);
    m_versionLabel->setFont(versionFont);
    
    titleLayout->addWidget(m_titleLabel);
    titleLayout->addWidget(m_versionLabel);
    titleLayout->addStretch();
    
    m_headerLayout->addWidget(m_iconLabel);
    m_headerLayout->addLayout(titleLayout);
    m_headerLayout->addStretch();
    
    m_mainLayout->addLayout(m_headerLayout);
    
    // Tab widget with different information sections
    m_tabWidget = new QTabWidget(this);
    
    // About tab
    m_aboutTab = new QTextBrowser(this);
    m_aboutTab->setOpenExternalLinks(true);
    m_aboutTab->setHtml(
        "<p><b>" + tr("Duplicate File Finder and Manager") + "</b></p>"
        "<p>" + tr("DupFinder helps you find and remove duplicate files on your system, "
                   "freeing up valuable disk space.") + "</p>"
        "<p><b>" + tr("Features:") + "</b></p>"
        "<ul>"
        "<li>" + tr("Fast duplicate detection using hash algorithms") + "</li>"
        "<li>" + tr("Multiple scan presets for common use cases") + "</li>"
        "<li>" + tr("Advanced filtering and file type selection") + "</li>"
        "<li>" + tr("Safe file operations with automatic backups") + "</li>"
        "<li>" + tr("Smart selection recommendations") + "</li>"
        "<li>" + tr("Comprehensive scan history") + "</li>"
        "<li>" + tr("Theme support (Light/Dark modes)") + "</li>"
        "</ul>"
        "<p><b>" + tr("Copyright:") + "</b> © 2024-2025 DupFinder Project</p>"
        "<p><b>" + tr("Website:") + "</b> <a href='https://dupfinder.org'>dupfinder.org</a></p>"
    );
    m_tabWidget->addTab(m_aboutTab, tr("About"));
    
    // License tab
    m_licenseTab = new QTextBrowser(this);
    m_licenseTab->setHtml(getLicenseInfo());
    m_tabWidget->addTab(m_licenseTab, tr("License"));
    
    // Authors tab
    m_authorsTab = new QTextBrowser(this);
    m_authorsTab->setHtml(getAuthorsInfo());
    m_tabWidget->addTab(m_authorsTab, tr("Authors"));
    
    // System Info tab
    m_systemTab = new QTextBrowser(this);
    m_systemTab->setHtml(getSystemInfo());
    m_tabWidget->addTab(m_systemTab, tr("System"));
    
    // Credits tab
    m_creditsTab = new QTextBrowser(this);
    m_creditsTab->setHtml(getLibrariesInfo());
    m_tabWidget->addTab(m_creditsTab, tr("Credits"));
    
    m_mainLayout->addWidget(m_tabWidget, 1);
    
    // Close button
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();
    m_closeButton = new QPushButton(tr("Close"), this);
    m_closeButton->setMinimumWidth(100);
    m_closeButton->setDefault(true);
    buttonLayout->addWidget(m_closeButton);
    
    m_mainLayout->addLayout(buttonLayout);
}

void AboutDialog::setupConnections()
{
    connect(m_closeButton, &QPushButton::clicked, this, &QDialog::accept);
    
    // Connect theme change signal
    connect(ThemeManager::instance(), &ThemeManager::themeChanged,
            this, &AboutDialog::applyTheme);
}

void AboutDialog::applyTheme()
{
    // Apply theme-aware styling to all components
    ThemeManager::instance()->applyToWidget(this);
}

QString AboutDialog::getVersionInfo() const
{
    return tr("Version %1 (Built: %2)")
        .arg(APP_VERSION)
        .arg(BUILD_DATE);
}

QString AboutDialog::getLicenseInfo() const
{
    return 
        "<h3>" + tr("License") + "</h3>"
        "<p>" + tr("DupFinder is released under the MIT License.") + "</p>"
        "<pre>"
        "MIT License\n"
        "\n"
        "Copyright (c) 2024-2025 DupFinder Project\n"
        "\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n"
        "\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n"
        "\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE."
        "</pre>";
}

QString AboutDialog::getAuthorsInfo() const
{
    return
        "<h3>" + tr("Development Team") + "</h3>"
        "<p><b>" + tr("Project Lead & Main Developer:") + "</b></p>"
        "<ul>"
        "<li>Deepak</li>"
        "</ul>"
        "<p><b>" + tr("Contributors:") + "</b></p>"
        "<ul>"
        "<li>" + tr("Community contributors on GitHub") + "</li>"
        "</ul>"
        "<p>" + tr("Special thanks to all contributors and testers who helped make DupFinder better.") + "</p>"
        "<p><b>" + tr("Contact:") + "</b></p>"
        "<ul>"
        "<li>" + tr("GitHub: <a href='https://github.com/dupfinder/dupfinder'>github.com/dupfinder/dupfinder</a>") + "</li>"
        "<li>" + tr("Issues: <a href='https://github.com/dupfinder/dupfinder/issues'>Report bugs or request features</a>") + "</li>"
        "</ul>";
}

QString AboutDialog::getSystemInfo() const
{
    QString info = "<h3>" + tr("System Information") + "</h3>";
    info += "<table width='100%' cellpadding='4'>";
    
    // Application info
    info += "<tr><td><b>" + tr("Application Version:") + "</b></td><td>" + QString(APP_VERSION) + "</td></tr>";
    info += "<tr><td><b>" + tr("Build Date:") + "</b></td><td>" + QString(BUILD_DATE) + "</td></tr>";
    info += "<tr><td><b>" + tr("Qt Version:") + "</b></td><td>" + QString(qVersion()) + "</td></tr>";
    
    // System info
    info += "<tr><td colspan='2'><hr></td></tr>";
    info += "<tr><td><b>" + tr("Operating System:") + "</b></td><td>" + QSysInfo::prettyProductName() + "</td></tr>";
    info += "<tr><td><b>" + tr("Kernel Version:") + "</b></td><td>" + QSysInfo::kernelVersion() + "</td></tr>";
    info += "<tr><td><b>" + tr("CPU Architecture:") + "</b></td><td>" + QSysInfo::currentCpuArchitecture() + "</td></tr>";
    info += "<tr><td><b>" + tr("Build ABI:") + "</b></td><td>" + QSysInfo::buildAbi() + "</td></tr>";
    
    // Application paths
    info += "<tr><td colspan='2'><hr></td></tr>";
    info += "<tr><td><b>" + tr("Application Path:") + "</b></td><td>" + QApplication::applicationDirPath() + "</td></tr>";
    info += "<tr><td><b>" + tr("Config Location:") + "</b></td><td>" + QStandardPaths::writableLocation(QStandardPaths::ConfigLocation) + "</td></tr>";
    
    info += "</table>";
    
    return info;
}

QString AboutDialog::getLibrariesInfo() const
{
    return
        "<h3>" + tr("Third-Party Libraries and Credits") + "</h3>"
        "<p>" + tr("DupFinder uses the following open-source libraries:") + "</p>"
        "<ul>"
        "<li><b>Qt " + QString(qVersion()) + "</b> - " + tr("Cross-platform application framework") + "<br>"
        "   " + tr("License: LGPL / GPL / Commercial") + "<br>"
        "   <a href='https://www.qt.io'>qt.io</a></li>"
        "</ul>"
        "<h4>" + tr("Hash Algorithms") + "</h4>"
        "<ul>"
        "<li><b>MD5</b> - " + tr("Fast hash for quick comparisons") + "</li>"
        "<li><b>SHA-256</b> - " + tr("Secure cryptographic hash") + "</li>"
        "<li><b>XXHash</b> - " + tr("Extremely fast non-cryptographic hash (planned)") + "</li>"
        "</ul>"
        "<h4>" + tr("Icons and Graphics") + "</h4>"
        "<ul>"
        "<li>" + tr("Application icons from various open-source icon sets") + "</li>"
        "<li>" + tr("Emoji icons from Unicode Consortium") + "</li>"
        "</ul>"
        "<h4>" + tr("Special Thanks") + "</h4>"
        "<p>" + tr("Thanks to the Qt community, Stack Overflow contributors, and all open-source "
                   "projects that inspired this application.") + "</p>";
}
