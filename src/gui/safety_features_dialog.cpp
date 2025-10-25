#include "safety_features_dialog.h"
#include "theme_manager.h"
#include "logger.h"
#include <QApplication>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QDesktopServices>
#include <QUrl>
#include <QFileDialog>
#include <QStandardPaths>
#include <QFileIconProvider>

SafetyFeaturesDialog::SafetyFeaturesDialog(SafetyManager* safetyManager, QWidget* parent)
    : QDialog(parent)
    , m_tabWidget(nullptr)
    , m_safetyManager(safetyManager)
{
    setupUI();
    setupConnections();
    refreshData();
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(this);
}

void SafetyFeaturesDialog::setupUI() {
    setWindowTitle(tr("Safety Features"));
    setModal(false);
    setMinimumSize(900, 700);
    resize(1100, 800);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(10);
    mainLayout->setContentsMargins(15, 15, 15, 15);

    // Tab widget
    m_tabWidget = new QTabWidget(this);
    
    setupProtectionTab();
    setupSettingsTab();
    setupSystemPathsTab();
    setupStatisticsTab();
    
    mainLayout->addWidget(m_tabWidget);

    // Dialog buttons
    m_buttonLayout = new QHBoxLayout();
    
    m_refreshButton = new QPushButton(tr("Refresh"), this);
    m_refreshButton->setToolTip(tr("Reload all data from SafetyManager"));
    m_buttonLayout->addWidget(m_refreshButton);
    
    m_resetButton = new QPushButton(tr("Reset to Defaults"), this);
    m_resetButton->setToolTip(tr("Reset all safety settings to default values"));
    m_buttonLayout->addWidget(m_resetButton);
    
    m_buttonLayout->addStretch();
    
    m_closeButton = new QPushButton(tr("Close"), this);
    m_buttonLayout->addWidget(m_closeButton);
    
    mainLayout->addLayout(m_buttonLayout);
}

void SafetyFeaturesDialog::setupProtectionTab() {
    m_protectionTab = new QWidget();
    m_tabWidget->addTab(m_protectionTab, tr("Protection Rules"));
    
    auto* layout = new QVBoxLayout(m_protectionTab);
    
    // Simple label for now
    auto* label = new QLabel(tr("Protection rules will be displayed here."), this);
    layout->addWidget(label);
}

void SafetyFeaturesDialog::setupSettingsTab() {
    m_settingsTab = new QWidget();
    m_tabWidget->addTab(m_settingsTab, tr("Settings"));
    
    auto* layout = new QVBoxLayout(m_settingsTab);
    
    // Simple label for now
    auto* label = new QLabel(tr("Safety settings will be displayed here."), this);
    layout->addWidget(label);
}

void SafetyFeaturesDialog::setupSystemPathsTab() {
    m_systemPathsTab = new QWidget();
    m_tabWidget->addTab(m_systemPathsTab, tr("System Paths"));
    
    auto* layout = new QVBoxLayout(m_systemPathsTab);
    
    // Simple label for now
    auto* label = new QLabel(tr("System protected paths will be displayed here."), this);
    layout->addWidget(label);
}

void SafetyFeaturesDialog::setupStatisticsTab() {
    m_statisticsTab = new QWidget();
    m_tabWidget->addTab(m_statisticsTab, tr("Statistics"));
    
    auto* layout = new QVBoxLayout(m_statisticsTab);
    
    // Simple label for now
    auto* label = new QLabel(tr("Safety statistics will be displayed here."), this);
    layout->addWidget(label);
}

void SafetyFeaturesDialog::setupConnections() {
    // Dialog buttons
    connect(m_refreshButton, &QPushButton::clicked, this, &SafetyFeaturesDialog::onRefreshClicked);
    connect(m_resetButton, &QPushButton::clicked, this, &SafetyFeaturesDialog::onResetToDefaultsClicked);
    connect(m_closeButton, &QPushButton::clicked, this, &QDialog::accept);
}

void SafetyFeaturesDialog::refreshData() {
    if (!m_safetyManager) {
        LOG_WARNING(LogCategories::UI, "Cannot refresh data: SafetyManager not set");
        return;
    }
    
    LOG_INFO(LogCategories::UI, "Refreshing safety features data");
    
    // Update all tabs with current data from SafetyManager
    updateProtectionDetails();
    updateSystemPaths();
    updateStatistics();
    
    LOG_DEBUG(LogCategories::UI, "Safety features data refreshed");
}

void SafetyFeaturesDialog::showProtectionForFile(const QString& filePath) {
    Q_UNUSED(filePath)
    show();
    raise();
    activateWindow();
}

// Stub implementations for required methods
void SafetyFeaturesDialog::onProtectionRuleSelectionChanged() {}
void SafetyFeaturesDialog::onAddProtectionRuleClicked() {}
void SafetyFeaturesDialog::onEditProtectionRuleClicked() {}
void SafetyFeaturesDialog::onRemoveProtectionRuleClicked() {}
void SafetyFeaturesDialog::onTestFileProtectionClicked() {}
void SafetyFeaturesDialog::onSafetyLevelChanged() {}
void SafetyFeaturesDialog::onBackupStrategyChanged() {}
void SafetyFeaturesDialog::onRefreshClicked() { refreshData(); }
void SafetyFeaturesDialog::onResetToDefaultsClicked() {
    QMessageBox::information(this, tr("Reset"), tr("Reset functionality not yet implemented."));
}
void SafetyFeaturesDialog::updateProtectionDetails() {}
void SafetyFeaturesDialog::updateStatistics() {}
void SafetyFeaturesDialog::updateSystemPaths() {}

// ProtectionRuleDialog minimal implementation
ProtectionRuleDialog::ProtectionRuleDialog(QWidget* parent)
    : QDialog(parent), m_isEdit(false)
{
    setupUI();
    setupConnections();
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(this);
}

ProtectionRuleDialog::ProtectionRuleDialog(const SafetyManager::ProtectionEntry& rule, QWidget* parent)
    : QDialog(parent), m_rule(rule), m_isEdit(true)
{
    setupUI();
    setupConnections();
    populateFields();
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(this);
}

void ProtectionRuleDialog::setupUI() {
    setWindowTitle(m_isEdit ? tr("Edit Protection Rule") : tr("Add Protection Rule"));
    setModal(true);
    setMinimumSize(400, 300);
    
    m_layout = new QVBoxLayout(this);
    
    // Simple form for now
    auto* label = new QLabel(tr("Protection rule dialog - not yet implemented"), this);
    m_layout->addWidget(label);
    
    // Buttons
    m_buttonLayout = new QHBoxLayout();
    m_buttonLayout->addStretch();
    
    m_okButton = new QPushButton(m_isEdit ? tr("Update") : tr("Add"), this);
    m_okButton->setDefault(true);
    m_buttonLayout->addWidget(m_okButton);
    
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    m_buttonLayout->addWidget(m_cancelButton);
    
    m_layout->addLayout(m_buttonLayout);
}

void ProtectionRuleDialog::setupConnections() {
    connect(m_okButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(m_cancelButton, &QPushButton::clicked, this, &QDialog::reject);
}

void ProtectionRuleDialog::populateFields() {}
void ProtectionRuleDialog::onPatternChanged() {}
void ProtectionRuleDialog::onBrowseClicked() {}
void ProtectionRuleDialog::validateInput() {}

SafetyManager::ProtectionEntry ProtectionRuleDialog::getProtectionRule() const {
    return SafetyManager::ProtectionEntry();
}