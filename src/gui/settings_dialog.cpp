#include "settings_dialog.h"
#include "theme_manager.h"
#include "logger.h"
#include <QSettings>
#include <QFileDialog>
#include <QStandardPaths>
#include <QDesktopServices>
#include <QUrl>
#include <QMessageBox>

SettingsDialog::SettingsDialog(QWidget* parent)
    : QDialog(parent)
    , m_tabWidget(nullptr)
    , m_buttonBox(nullptr)
    , m_applyButton(nullptr)
    , m_restoreDefaultsButton(nullptr)
    , m_uiFeaturesTab(nullptr)
    , m_thumbnailSizeSpin(nullptr)
    , m_thumbnailCacheSizeSpin(nullptr)
    , m_enableThumbnailsCheck(nullptr)
    , m_operationQueueSizeSpin(nullptr)
    , m_selectionHistorySizeSpin(nullptr)
    , m_enableAdvancedFiltersCheck(nullptr)
    , m_enableSmartSelectionCheck(nullptr)
    , m_enableOperationQueueCheck(nullptr)
    , m_showDetailedProgressCheck(nullptr)
    , m_gpuTab(nullptr)
    , m_enableGPUCheck(nullptr)
    , m_preferCUDACheck(nullptr)
    , m_gpuFallbackCheck(nullptr)
    , m_gpuMemoryLimitSpin(nullptr)
    , m_gpuBatchSizeSpin(nullptr)
    , m_gpuCachingCheck(nullptr)
    , m_gpuStatusLabel(nullptr)
    , m_gpuTestButton(nullptr)
{
    setWindowTitle(tr("Settings"));
    setMinimumSize(700, 500);
    resize(750, 550);
    setModal(true);
    
    setupUI();
    loadSettings();
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(this);
    
    LOG_INFO(LogCategories::UI, "Settings dialog created");
}

SettingsDialog::~SettingsDialog()
{
    LOG_DEBUG(LogCategories::UI, "Settings dialog destroyed");
}

void SettingsDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    mainLayout->setSpacing(10);
    
    // Create tab widget
    m_tabWidget = new QTabWidget(this);
    
    // Create tabs
    createGeneralTab();
    createScanningTab();
    createSafetyTab();
    createLoggingTab();
    createAdvancedTab();
    createUIFeaturesTab();  // Task 32
    createGPUTab();  // Task 27
    
    mainLayout->addWidget(m_tabWidget);
    
    // Create button box
    m_buttonBox = new QDialogButtonBox(this);
    
    QPushButton* okButton = m_buttonBox->addButton(QDialogButtonBox::Ok);
    QPushButton* cancelButton = m_buttonBox->addButton(QDialogButtonBox::Cancel);
    m_applyButton = m_buttonBox->addButton(QDialogButtonBox::Apply);
    m_restoreDefaultsButton = new QPushButton(tr("Restore Defaults"), this);
    
    m_buttonBox->addButton(m_restoreDefaultsButton, QDialogButtonBox::ResetRole);
    
    connect(okButton, &QPushButton::clicked, this, &SettingsDialog::onOkClicked);
    connect(cancelButton, &QPushButton::clicked, this, &SettingsDialog::onCancelClicked);
    connect(m_applyButton, &QPushButton::clicked, this, &SettingsDialog::onApplyClicked);
    connect(m_restoreDefaultsButton, &QPushButton::clicked, this, &SettingsDialog::onRestoreDefaultsClicked);
    
    mainLayout->addWidget(m_buttonBox);
    
    // Connect theme changes to immediate application
    connect(m_themeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SettingsDialog::onThemeChanged);
}

void SettingsDialog::createGeneralTab()
{
    m_generalTab = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_generalTab);
    layout->setContentsMargins(20, 20, 20, 20);
    layout->setSpacing(15);
    
    // Language
    QGroupBox* languageGroup = new QGroupBox(tr("Language"), m_generalTab);
    QFormLayout* languageLayout = new QFormLayout(languageGroup);
    
    m_languageCombo = new QComboBox(m_generalTab);
    m_languageCombo->addItem(tr("English"), "en");
    m_languageCombo->addItem(tr("Spanish"), "es");
    m_languageCombo->addItem(tr("French"), "fr");
    m_languageCombo->addItem(tr("German"), "de");
    languageLayout->addRow(tr("Language:"), m_languageCombo);
    
    layout->addWidget(languageGroup);
    
    // Appearance
    QGroupBox* appearanceGroup = new QGroupBox(tr("Appearance"), m_generalTab);
    QFormLayout* appearanceLayout = new QFormLayout(appearanceGroup);
    
    m_themeCombo = new QComboBox(m_generalTab);
    m_themeCombo->addItem(tr("System Default"), "system");
    m_themeCombo->addItem(tr("Light"), "light");
    m_themeCombo->addItem(tr("Dark"), "dark");
    appearanceLayout->addRow(tr("Theme:"), m_themeCombo);
    
    layout->addWidget(appearanceGroup);
    
    // Startup
    QGroupBox* startupGroup = new QGroupBox(tr("Startup"), m_generalTab);
    QVBoxLayout* startupLayout = new QVBoxLayout(startupGroup);
    
    m_startupCheck = new QCheckBox(tr("Launch on system startup"), m_generalTab);
    m_updateCheck = new QCheckBox(tr("Check for updates automatically"), m_generalTab);
    
    startupLayout->addWidget(m_startupCheck);
    startupLayout->addWidget(m_updateCheck);
    
    layout->addWidget(startupGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(m_generalTab, tr("General"));
}

void SettingsDialog::createScanningTab()
{
    m_scanningTab = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_scanningTab);
    layout->setContentsMargins(20, 20, 20, 20);
    layout->setSpacing(15);
    
    // Default Scan Options
    QGroupBox* defaultsGroup = new QGroupBox(tr("Default Scan Options"), m_scanningTab);
    QFormLayout* defaultsLayout = new QFormLayout(defaultsGroup);
    
    m_minFileSizeSpin = new QSpinBox(m_scanningTab);
    m_minFileSizeSpin->setRange(0, 1024);
    m_minFileSizeSpin->setSuffix(tr(" MB"));
    m_minFileSizeSpin->setToolTip(tr("Default minimum file size for scans"));
    defaultsLayout->addRow(tr("Minimum file size:"), m_minFileSizeSpin);
    
    m_includeHiddenCheck = new QCheckBox(tr("Include hidden files by default"), m_scanningTab);
    m_followSymlinksCheck = new QCheckBox(tr("Follow symbolic links by default"), m_scanningTab);
    
    defaultsLayout->addRow("", m_includeHiddenCheck);
    defaultsLayout->addRow("", m_followSymlinksCheck);
    
    layout->addWidget(defaultsGroup);
    
    // Performance
    QGroupBox* performanceGroup = new QGroupBox(tr("Performance"), m_scanningTab);
    QFormLayout* performanceLayout = new QFormLayout(performanceGroup);
    
    m_threadCountSpin = new QSpinBox(m_scanningTab);
    m_threadCountSpin->setRange(1, 16);
    m_threadCountSpin->setToolTip(tr("Number of threads for scanning (more = faster but more CPU)"));
    performanceLayout->addRow(tr("Thread count:"), m_threadCountSpin);
    
    m_cacheSizeSpin = new QSpinBox(m_scanningTab);
    m_cacheSizeSpin->setRange(10, 1000);
    m_cacheSizeSpin->setSuffix(tr(" MB"));
    m_cacheSizeSpin->setToolTip(tr("Cache size for scan results"));
    performanceLayout->addRow(tr("Cache size:"), m_cacheSizeSpin);
    
    layout->addWidget(performanceGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(m_scanningTab, tr("Scanning"));
}

void SettingsDialog::createSafetyTab()
{
    m_safetyTab = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_safetyTab);
    layout->setContentsMargins(20, 20, 20, 20);
    layout->setSpacing(15);
    
    // Backup Settings
    QGroupBox* backupGroup = new QGroupBox(tr("Backup Settings"), m_safetyTab);
    QVBoxLayout* backupLayout = new QVBoxLayout(backupGroup);
    
    QHBoxLayout* locationLayout = new QHBoxLayout();
    QLabel* locationLabel = new QLabel(tr("Backup location:"), m_safetyTab);
    m_backupLocationEdit = new QLineEdit(m_safetyTab);
    m_backupLocationEdit->setReadOnly(true);
    m_browseBackupButton = new QPushButton(tr("Browse..."), m_safetyTab);
    connect(m_browseBackupButton, &QPushButton::clicked, this, &SettingsDialog::onBrowseBackupDirectory);
    
    locationLayout->addWidget(locationLabel);
    locationLayout->addWidget(m_backupLocationEdit, 1);
    locationLayout->addWidget(m_browseBackupButton);
    backupLayout->addLayout(locationLayout);
    
    QFormLayout* retentionLayout = new QFormLayout();
    m_backupRetentionSpin = new QSpinBox(m_safetyTab);
    m_backupRetentionSpin->setRange(1, 365);
    m_backupRetentionSpin->setSuffix(tr(" days"));
    m_backupRetentionSpin->setToolTip(tr("How long to keep backup files"));
    retentionLayout->addRow(tr("Backup retention:"), m_backupRetentionSpin);
    backupLayout->addLayout(retentionLayout);
    
    layout->addWidget(backupGroup);
    
    // Protected Paths
    QGroupBox* protectedGroup = new QGroupBox(tr("Protected Paths"), m_safetyTab);
    QVBoxLayout* protectedLayout = new QVBoxLayout(protectedGroup);
    
    QLabel* protectedLabel = new QLabel(tr("Files in these paths cannot be deleted:"), m_safetyTab);
    protectedLayout->addWidget(protectedLabel);
    
    m_protectedPathsList = new QListWidget(m_safetyTab);
    m_protectedPathsList->setMaximumHeight(100);
    protectedLayout->addWidget(m_protectedPathsList);
    
    QHBoxLayout* pathButtonsLayout = new QHBoxLayout();
    m_addPathButton = new QPushButton(tr("Add Path..."), m_safetyTab);
    m_removePathButton = new QPushButton(tr("Remove"), m_safetyTab);
    connect(m_addPathButton, &QPushButton::clicked, this, &SettingsDialog::onAddProtectedPath);
    connect(m_removePathButton, &QPushButton::clicked, this, &SettingsDialog::onRemoveProtectedPath);
    
    pathButtonsLayout->addWidget(m_addPathButton);
    pathButtonsLayout->addWidget(m_removePathButton);
    pathButtonsLayout->addStretch();
    protectedLayout->addLayout(pathButtonsLayout);
    
    layout->addWidget(protectedGroup);
    
    // Confirmations
    QGroupBox* confirmGroup = new QGroupBox(tr("Confirmations"), m_safetyTab);
    QVBoxLayout* confirmLayout = new QVBoxLayout(confirmGroup);
    
    m_confirmDeleteCheck = new QCheckBox(tr("Confirm before deleting files"), m_safetyTab);
    m_confirmMoveCheck = new QCheckBox(tr("Confirm before moving files"), m_safetyTab);
    
    confirmLayout->addWidget(m_confirmDeleteCheck);
    confirmLayout->addWidget(m_confirmMoveCheck);
    
    layout->addWidget(confirmGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(m_safetyTab, tr("Safety"));
}

void SettingsDialog::createLoggingTab()
{
    m_loggingTab = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_loggingTab);
    layout->setContentsMargins(20, 20, 20, 20);
    layout->setSpacing(15);
    
    // Log Level
    QGroupBox* levelGroup = new QGroupBox(tr("Log Level"), m_loggingTab);
    QFormLayout* levelLayout = new QFormLayout(levelGroup);
    
    m_logLevelCombo = new QComboBox(m_loggingTab);
    m_logLevelCombo->addItem(tr("Debug (Most Verbose)"), "debug");
    m_logLevelCombo->addItem(tr("Info"), "info");
    m_logLevelCombo->addItem(tr("Warning"), "warning");
    m_logLevelCombo->addItem(tr("Error"), "error");
    m_logLevelCombo->addItem(tr("Critical (Least Verbose)"), "critical");
    m_logLevelCombo->setToolTip(tr("Lower levels show more detailed information"));
    levelLayout->addRow(tr("Log level:"), m_logLevelCombo);
    
    layout->addWidget(levelGroup);
    
    // Log Output
    QGroupBox* outputGroup = new QGroupBox(tr("Log Output"), m_loggingTab);
    QVBoxLayout* outputLayout = new QVBoxLayout(outputGroup);
    
    m_logToFileCheck = new QCheckBox(tr("Log to file"), m_loggingTab);
    m_logToConsoleCheck = new QCheckBox(tr("Log to console"), m_loggingTab);
    
    outputLayout->addWidget(m_logToFileCheck);
    outputLayout->addWidget(m_logToConsoleCheck);
    
    layout->addWidget(outputGroup);
    
    // Log Directory
    QGroupBox* dirGroup = new QGroupBox(tr("Log Directory"), m_loggingTab);
    QVBoxLayout* dirLayout = new QVBoxLayout(dirGroup);
    
    QHBoxLayout* dirPathLayout = new QHBoxLayout();
    m_logDirectoryEdit = new QLineEdit(m_loggingTab);
    m_logDirectoryEdit->setReadOnly(true);
    m_browseLogButton = new QPushButton(tr("Browse..."), m_loggingTab);
    m_openLogButton = new QPushButton(tr("Open Log Directory"), m_loggingTab);
    connect(m_browseLogButton, &QPushButton::clicked, this, &SettingsDialog::onBrowseLogDirectory);
    connect(m_openLogButton, &QPushButton::clicked, this, &SettingsDialog::onOpenLogDirectory);
    
    dirPathLayout->addWidget(m_logDirectoryEdit, 1);
    dirPathLayout->addWidget(m_browseLogButton);
    dirPathLayout->addWidget(m_openLogButton);
    dirLayout->addLayout(dirPathLayout);
    
    layout->addWidget(dirGroup);
    
    // Log Rotation
    QGroupBox* rotationGroup = new QGroupBox(tr("Log Rotation"), m_loggingTab);
    QFormLayout* rotationLayout = new QFormLayout(rotationGroup);
    
    m_maxLogFilesSpin = new QSpinBox(m_loggingTab);
    m_maxLogFilesSpin->setRange(1, 100);
    m_maxLogFilesSpin->setToolTip(tr("Maximum number of log files to keep"));
    rotationLayout->addRow(tr("Max log files:"), m_maxLogFilesSpin);
    
    m_maxLogSizeSpin = new QSpinBox(m_loggingTab);
    m_maxLogSizeSpin->setRange(1, 100);
    m_maxLogSizeSpin->setSuffix(tr(" MB"));
    m_maxLogSizeSpin->setToolTip(tr("Maximum size of each log file"));
    rotationLayout->addRow(tr("Max file size:"), m_maxLogSizeSpin);
    
    layout->addWidget(rotationGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(m_loggingTab, tr("Logging"));
}

void SettingsDialog::createAdvancedTab()
{
    m_advancedTab = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_advancedTab);
    layout->setContentsMargins(20, 20, 20, 20);
    layout->setSpacing(15);
    
    // Storage
    QGroupBox* storageGroup = new QGroupBox(tr("Storage"), m_advancedTab);
    QFormLayout* storageLayout = new QFormLayout(storageGroup);
    
    m_databaseLocationEdit = new QLineEdit(m_advancedTab);
    m_databaseLocationEdit->setReadOnly(true);
    storageLayout->addRow(tr("Database location:"), m_databaseLocationEdit);
    
    m_cacheDirectoryEdit = new QLineEdit(m_advancedTab);
    m_cacheDirectoryEdit->setReadOnly(true);
    storageLayout->addRow(tr("Cache directory:"), m_cacheDirectoryEdit);
    
    layout->addWidget(storageGroup);
    
    // Export
    QGroupBox* exportGroup = new QGroupBox(tr("Export Defaults"), m_advancedTab);
    QFormLayout* exportLayout = new QFormLayout(exportGroup);
    
    m_exportFormatCombo = new QComboBox(m_advancedTab);
    m_exportFormatCombo->addItem(tr("CSV"), "csv");
    m_exportFormatCombo->addItem(tr("JSON"), "json");
    m_exportFormatCombo->addItem(tr("Text"), "txt");
    exportLayout->addRow(tr("Default format:"), m_exportFormatCombo);
    
    layout->addWidget(exportGroup);
    
    // Performance
    QGroupBox* perfGroup = new QGroupBox(tr("Performance Monitoring"), m_advancedTab);
    QVBoxLayout* perfLayout = new QVBoxLayout(perfGroup);
    
    m_enablePerformanceCheck = new QCheckBox(tr("Enable performance monitoring"), m_advancedTab);
    m_enablePerformanceCheck->setToolTip(tr("Track and log performance metrics"));
    perfLayout->addWidget(m_enablePerformanceCheck);
    
    layout->addWidget(perfGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(m_advancedTab, tr("Advanced"));
}

void SettingsDialog::createUIFeaturesTab()
{
    m_uiFeaturesTab = new QWidget();
    auto* layout = new QVBoxLayout(m_uiFeaturesTab);
    
    // Thumbnail Settings
    auto* thumbnailGroup = new QGroupBox(tr("Thumbnail Settings"), this);
    auto* thumbnailLayout = new QFormLayout(thumbnailGroup);
    
    m_enableThumbnailsCheck = new QCheckBox(tr("Enable thumbnails in results"), this);
    thumbnailLayout->addRow(m_enableThumbnailsCheck);
    
    m_thumbnailSizeSpin = new QSpinBox(this);
    m_thumbnailSizeSpin->setRange(32, 256);
    m_thumbnailSizeSpin->setValue(64);
    m_thumbnailSizeSpin->setSuffix(tr(" pixels"));
    thumbnailLayout->addRow(tr("Thumbnail size:"), m_thumbnailSizeSpin);
    
    m_thumbnailCacheSizeSpin = new QSpinBox(this);
    m_thumbnailCacheSizeSpin->setRange(100, 10000);
    m_thumbnailCacheSizeSpin->setValue(1000);
    m_thumbnailCacheSizeSpin->setSuffix(tr(" items"));
    thumbnailLayout->addRow(tr("Thumbnail cache size:"), m_thumbnailCacheSizeSpin);
    
    layout->addWidget(thumbnailGroup);
    
    // Selection Settings
    auto* selectionGroup = new QGroupBox(tr("Selection Settings"), this);
    auto* selectionLayout = new QFormLayout(selectionGroup);
    
    m_selectionHistorySizeSpin = new QSpinBox(this);
    m_selectionHistorySizeSpin->setRange(10, 200);
    m_selectionHistorySizeSpin->setValue(50);
    m_selectionHistorySizeSpin->setSuffix(tr(" items"));
    selectionLayout->addRow(tr("Selection history size:"), m_selectionHistorySizeSpin);
    
    m_enableSmartSelectionCheck = new QCheckBox(tr("Enable smart selection features"), this);
    selectionLayout->addRow(m_enableSmartSelectionCheck);
    
    layout->addWidget(selectionGroup);
    
    // Filter Settings
    auto* filterGroup = new QGroupBox(tr("Filter Settings"), this);
    auto* filterLayout = new QFormLayout(filterGroup);
    
    m_enableAdvancedFiltersCheck = new QCheckBox(tr("Enable advanced filtering"), this);
    filterLayout->addRow(m_enableAdvancedFiltersCheck);
    
    layout->addWidget(filterGroup);
    
    // Operation Settings
    auto* operationGroup = new QGroupBox(tr("File Operation Settings"), this);
    auto* operationLayout = new QFormLayout(operationGroup);
    
    m_enableOperationQueueCheck = new QCheckBox(tr("Enable operation queue"), this);
    operationLayout->addRow(m_enableOperationQueueCheck);
    
    m_operationQueueSizeSpin = new QSpinBox(this);
    m_operationQueueSizeSpin->setRange(10, 1000);
    m_operationQueueSizeSpin->setValue(100);
    m_operationQueueSizeSpin->setSuffix(tr(" operations"));
    operationLayout->addRow(tr("Max queue size:"), m_operationQueueSizeSpin);
    
    m_showDetailedProgressCheck = new QCheckBox(tr("Show detailed progress dialogs"), this);
    operationLayout->addRow(m_showDetailedProgressCheck);
    
    layout->addWidget(operationGroup);
    
    layout->addStretch();
    
    m_tabWidget->addTab(m_uiFeaturesTab, tr("UI Features"));
}

void SettingsDialog::createGPUTab()
{
    m_gpuTab = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_gpuTab);
    layout->setContentsMargins(20, 20, 20, 20);
    layout->setSpacing(15);
    
    // GPU Status
    QGroupBox* statusGroup = new QGroupBox(tr("GPU Status"), m_gpuTab);
    QVBoxLayout* statusLayout = new QVBoxLayout(statusGroup);
    
    m_gpuStatusLabel = new QLabel(tr("GPU status will be detected on test..."), m_gpuTab);
    m_gpuStatusLabel->setWordWrap(true);
    statusLayout->addWidget(m_gpuStatusLabel);
    
    m_gpuTestButton = new QPushButton(tr("Test GPU Acceleration"), m_gpuTab);
    connect(m_gpuTestButton, &QPushButton::clicked, this, &SettingsDialog::onGPUTestClicked);
    statusLayout->addWidget(m_gpuTestButton);
    
    layout->addWidget(statusGroup);
    
    // GPU Settings
    QGroupBox* settingsGroup = new QGroupBox(tr("GPU Acceleration Settings"), m_gpuTab);
    QFormLayout* settingsLayout = new QFormLayout(settingsGroup);
    
    m_enableGPUCheck = new QCheckBox(tr("Enable GPU acceleration"), m_gpuTab);
    m_enableGPUCheck->setToolTip(tr("Use GPU for hash calculations when available"));
    settingsLayout->addRow(m_enableGPUCheck);
    
    m_preferCUDACheck = new QCheckBox(tr("Prefer CUDA over OpenCL"), m_gpuTab);
    m_preferCUDACheck->setToolTip(tr("Use CUDA when both CUDA and OpenCL are available"));
    settingsLayout->addRow(m_preferCUDACheck);
    
    m_gpuFallbackCheck = new QCheckBox(tr("Enable CPU fallback"), m_gpuTab);
    m_gpuFallbackCheck->setToolTip(tr("Fall back to CPU if GPU acceleration fails"));
    settingsLayout->addRow(m_gpuFallbackCheck);
    
    m_gpuMemoryLimitSpin = new QSpinBox(m_gpuTab);
    m_gpuMemoryLimitSpin->setRange(100, 4096);
    m_gpuMemoryLimitSpin->setValue(1024);
    m_gpuMemoryLimitSpin->setSuffix(tr(" MB"));
    m_gpuMemoryLimitSpin->setToolTip(tr("Maximum GPU memory to use for hashing"));
    settingsLayout->addRow(tr("GPU memory limit:"), m_gpuMemoryLimitSpin);
    
    m_gpuBatchSizeSpin = new QSpinBox(m_gpuTab);
    m_gpuBatchSizeSpin->setRange(1, 100);
    m_gpuBatchSizeSpin->setValue(10);
    m_gpuBatchSizeSpin->setToolTip(tr("Number of files to process in each GPU batch"));
    settingsLayout->addRow(tr("Batch size:"), m_gpuBatchSizeSpin);
    
    m_gpuCachingCheck = new QCheckBox(tr("Enable GPU result caching"), m_gpuTab);
    m_gpuCachingCheck->setToolTip(tr("Cache GPU hash results for better performance"));
    settingsLayout->addRow(m_gpuCachingCheck);
    
    layout->addWidget(settingsGroup);
    
    // Performance Info
    QGroupBox* infoGroup = new QGroupBox(tr("Performance Information"), m_gpuTab);
    QVBoxLayout* infoLayout = new QVBoxLayout(infoGroup);
    
    QLabel* infoLabel = new QLabel(tr("GPU acceleration can significantly speed up hash calculations for large files. "
                                     "Files smaller than 1MB may not benefit from GPU acceleration due to transfer overhead."), m_gpuTab);
    infoLabel->setWordWrap(true);
    infoLayout->addWidget(infoLabel);
    
    layout->addWidget(infoGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(m_gpuTab, tr("GPU"));
}

void SettingsDialog::loadSettings()
{
    LOG_INFO(LogCategories::CONFIG, "Loading settings");
    
    QSettings settings("CloneClean", "CloneClean");
    
    // General
    QString language = settings.value("general/language", "en").toString();
    int langIndex = m_languageCombo->findData(language);
    if (langIndex >= 0) m_languageCombo->setCurrentIndex(langIndex);
    
    // Load theme from ThemeManager instead of old settings
    ThemeManager* themeManager = ThemeManager::instance();
    QString currentThemeString;
    switch (themeManager->currentTheme()) {
        case ThemeManager::Light:
            currentThemeString = "light";
            break;
        case ThemeManager::Dark:
            currentThemeString = "dark";
            break;
        case ThemeManager::Custom:
            currentThemeString = "custom";
            break;
        case ThemeManager::SystemDefault:
        default:
            currentThemeString = "system";
            break;
    }
    int themeIndex = m_themeCombo->findData(currentThemeString);
    if (themeIndex >= 0) m_themeCombo->setCurrentIndex(themeIndex);
    
    m_startupCheck->setChecked(settings.value("general/launchOnStartup", false).toBool());
    m_updateCheck->setChecked(settings.value("general/checkUpdates", true).toBool());
    
    // Scanning
    m_minFileSizeSpin->setValue(settings.value("scanning/minFileSize", 0).toInt());
    m_includeHiddenCheck->setChecked(settings.value("scanning/includeHidden", false).toBool());
    m_followSymlinksCheck->setChecked(settings.value("scanning/followSymlinks", true).toBool());
    m_threadCountSpin->setValue(settings.value("scanning/threadCount", 4).toInt());
    m_cacheSizeSpin->setValue(settings.value("scanning/cacheSize", 100).toInt());
    
    // Safety
    QString defaultBackup = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/backups";
    m_backupLocationEdit->setText(settings.value("safety/backupLocation", defaultBackup).toString());
    m_backupRetentionSpin->setValue(settings.value("safety/backupRetention", 30).toInt());
    m_confirmDeleteCheck->setChecked(settings.value("safety/confirmDelete", true).toBool());
    m_confirmMoveCheck->setChecked(settings.value("safety/confirmMove", true).toBool());
    
    // Protected paths
    QStringList protectedPaths = settings.value("safety/protectedPaths").toStringList();
    m_protectedPathsList->clear();
    m_protectedPathsList->addItems(protectedPaths);
    
    // Logging
    QString logLevel = settings.value("logging/level", "info").toString();
    int logIndex = m_logLevelCombo->findData(logLevel);
    if (logIndex >= 0) m_logLevelCombo->setCurrentIndex(logIndex);
    
    m_logToFileCheck->setChecked(settings.value("logging/toFile", true).toBool());
    m_logToConsoleCheck->setChecked(settings.value("logging/toConsole", true).toBool());
    
    QString defaultLogDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/logs";
    m_logDirectoryEdit->setText(settings.value("logging/directory", defaultLogDir).toString());
    m_maxLogFilesSpin->setValue(settings.value("logging/maxFiles", 10).toInt());
    m_maxLogSizeSpin->setValue(settings.value("logging/maxSize", 10).toInt());
    
    // Advanced
    QString defaultDb = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/cloneclean.db";
    m_databaseLocationEdit->setText(settings.value("advanced/databaseLocation", defaultDb).toString());
    
    QString defaultCache = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
    m_cacheDirectoryEdit->setText(settings.value("advanced/cacheDirectory", defaultCache).toString());
    
    QString exportFormat = settings.value("advanced/exportFormat", "csv").toString();
    int exportIndex = m_exportFormatCombo->findData(exportFormat);
    if (exportIndex >= 0) m_exportFormatCombo->setCurrentIndex(exportIndex);
    
    m_enablePerformanceCheck->setChecked(settings.value("advanced/enablePerformance", false).toBool());
    
    // UI Features (Task 32)
    if (m_enableThumbnailsCheck) {
        m_enableThumbnailsCheck->setChecked(settings.value("ui/enableThumbnails", true).toBool());
    }
    if (m_thumbnailSizeSpin) {
        m_thumbnailSizeSpin->setValue(settings.value("ui/thumbnailSize", 64).toInt());
    }
    if (m_thumbnailCacheSizeSpin) {
        m_thumbnailCacheSizeSpin->setValue(settings.value("ui/thumbnailCacheSize", 1000).toInt());
    }
    if (m_selectionHistorySizeSpin) {
        m_selectionHistorySizeSpin->setValue(settings.value("ui/selectionHistorySize", 50).toInt());
    }
    if (m_enableSmartSelectionCheck) {
        m_enableSmartSelectionCheck->setChecked(settings.value("ui/enableSmartSelection", true).toBool());
    }
    if (m_enableAdvancedFiltersCheck) {
        m_enableAdvancedFiltersCheck->setChecked(settings.value("ui/enableAdvancedFilters", true).toBool());
    }
    if (m_enableOperationQueueCheck) {
        m_enableOperationQueueCheck->setChecked(settings.value("ui/enableOperationQueue", true).toBool());
    }
    if (m_operationQueueSizeSpin) {
        m_operationQueueSizeSpin->setValue(settings.value("ui/operationQueueSize", 100).toInt());
    }
    if (m_showDetailedProgressCheck) {
        m_showDetailedProgressCheck->setChecked(settings.value("ui/showDetailedProgress", true).toBool());
    }
    
    // GPU Settings (Task 27)
    if (m_enableGPUCheck) {
        m_enableGPUCheck->setChecked(settings.value("gpu/enabled", true).toBool());
    }
    if (m_preferCUDACheck) {
        m_preferCUDACheck->setChecked(settings.value("gpu/preferCUDA", true).toBool());
    }
    if (m_gpuFallbackCheck) {
        m_gpuFallbackCheck->setChecked(settings.value("gpu/fallbackEnabled", true).toBool());
    }
    if (m_gpuMemoryLimitSpin) {
        m_gpuMemoryLimitSpin->setValue(settings.value("gpu/memoryLimit", 1024).toInt());
    }
    if (m_gpuBatchSizeSpin) {
        m_gpuBatchSizeSpin->setValue(settings.value("gpu/batchSize", 10).toInt());
    }
    if (m_gpuCachingCheck) {
        m_gpuCachingCheck->setChecked(settings.value("gpu/cachingEnabled", true).toBool());
    }
    
    LOG_INFO(LogCategories::CONFIG, "Settings loaded successfully");
}

void SettingsDialog::saveSettings()
{
    LOG_INFO(LogCategories::CONFIG, "Saving settings");
    
    QSettings settings("CloneClean", "CloneClean");
    
    // General
    settings.setValue("general/language", m_languageCombo->currentData());
    
    // Apply theme change through ThemeManager instead of old settings
    QString selectedTheme = m_themeCombo->currentData().toString();
    ThemeManager* themeManager = ThemeManager::instance();
    ThemeManager::Theme newTheme = ThemeManager::SystemDefault;
    if (selectedTheme == "light") {
        newTheme = ThemeManager::Light;
    } else if (selectedTheme == "dark") {
        newTheme = ThemeManager::Dark;
    } else if (selectedTheme == "custom") {
        newTheme = ThemeManager::Custom;
    }
    themeManager->setTheme(newTheme);
    
    // Remove old theme setting to avoid conflicts
    if (settings.contains("general/theme")) {
        settings.remove("general/theme");
    }
    
    settings.setValue("general/launchOnStartup", m_startupCheck->isChecked());
    settings.setValue("general/checkUpdates", m_updateCheck->isChecked());
    
    // Scanning
    settings.setValue("scanning/minFileSize", m_minFileSizeSpin->value());
    settings.setValue("scanning/includeHidden", m_includeHiddenCheck->isChecked());
    settings.setValue("scanning/followSymlinks", m_followSymlinksCheck->isChecked());
    settings.setValue("scanning/threadCount", m_threadCountSpin->value());
    settings.setValue("scanning/cacheSize", m_cacheSizeSpin->value());
    
    // Safety
    settings.setValue("safety/backupLocation", m_backupLocationEdit->text());
    settings.setValue("safety/backupRetention", m_backupRetentionSpin->value());
    settings.setValue("safety/confirmDelete", m_confirmDeleteCheck->isChecked());
    settings.setValue("safety/confirmMove", m_confirmMoveCheck->isChecked());
    
    // Protected paths
    QStringList protectedPaths;
    for (int i = 0; i < m_protectedPathsList->count(); ++i) {
        protectedPaths << m_protectedPathsList->item(i)->text();
    }
    settings.setValue("safety/protectedPaths", protectedPaths);
    
    // Logging
    settings.setValue("logging/level", m_logLevelCombo->currentData());
    settings.setValue("logging/toFile", m_logToFileCheck->isChecked());
    settings.setValue("logging/toConsole", m_logToConsoleCheck->isChecked());
    settings.setValue("logging/directory", m_logDirectoryEdit->text());
    settings.setValue("logging/maxFiles", m_maxLogFilesSpin->value());
    settings.setValue("logging/maxSize", m_maxLogSizeSpin->value());
    
    // Advanced
    settings.setValue("advanced/databaseLocation", m_databaseLocationEdit->text());
    settings.setValue("advanced/cacheDirectory", m_cacheDirectoryEdit->text());
    settings.setValue("advanced/exportFormat", m_exportFormatCombo->currentData());
    settings.setValue("advanced/enablePerformance", m_enablePerformanceCheck->isChecked());
    
    // UI Features (Task 32)
    if (m_enableThumbnailsCheck) {
        settings.setValue("ui/enableThumbnails", m_enableThumbnailsCheck->isChecked());
    }
    if (m_thumbnailSizeSpin) {
        settings.setValue("ui/thumbnailSize", m_thumbnailSizeSpin->value());
    }
    if (m_thumbnailCacheSizeSpin) {
        settings.setValue("ui/thumbnailCacheSize", m_thumbnailCacheSizeSpin->value());
    }
    if (m_selectionHistorySizeSpin) {
        settings.setValue("ui/selectionHistorySize", m_selectionHistorySizeSpin->value());
    }
    if (m_enableSmartSelectionCheck) {
        settings.setValue("ui/enableSmartSelection", m_enableSmartSelectionCheck->isChecked());
    }
    if (m_enableAdvancedFiltersCheck) {
        settings.setValue("ui/enableAdvancedFilters", m_enableAdvancedFiltersCheck->isChecked());
    }
    if (m_enableOperationQueueCheck) {
        settings.setValue("ui/enableOperationQueue", m_enableOperationQueueCheck->isChecked());
    }
    if (m_operationQueueSizeSpin) {
        settings.setValue("ui/operationQueueSize", m_operationQueueSizeSpin->value());
    }
    if (m_showDetailedProgressCheck) {
        settings.setValue("ui/showDetailedProgress", m_showDetailedProgressCheck->isChecked());
    }
    
    // GPU Settings (Task 27)
    if (m_enableGPUCheck) {
        settings.setValue("gpu/enabled", m_enableGPUCheck->isChecked());
    }
    if (m_preferCUDACheck) {
        settings.setValue("gpu/preferCUDA", m_preferCUDACheck->isChecked());
    }
    if (m_gpuFallbackCheck) {
        settings.setValue("gpu/fallbackEnabled", m_gpuFallbackCheck->isChecked());
    }
    if (m_gpuMemoryLimitSpin) {
        settings.setValue("gpu/memoryLimit", m_gpuMemoryLimitSpin->value());
    }
    if (m_gpuBatchSizeSpin) {
        settings.setValue("gpu/batchSize", m_gpuBatchSizeSpin->value());
    }
    if (m_gpuCachingCheck) {
        settings.setValue("gpu/cachingEnabled", m_gpuCachingCheck->isChecked());
    }
    
    settings.sync();
    
    LOG_INFO(LogCategories::CONFIG, "Settings saved successfully");
    emit settingsChanged();
}

void SettingsDialog::onApplyClicked()
{
    LOG_INFO(LogCategories::UI, "User clicked 'Apply' in settings dialog");
    saveSettings();
    QMessageBox::information(this, tr("Settings Applied"), 
                           tr("Settings have been saved. Some changes may require restarting the application."));
}

void SettingsDialog::onOkClicked()
{
    LOG_INFO(LogCategories::UI, "User clicked 'OK' in settings dialog");
    saveSettings();
    accept();
}

void SettingsDialog::onCancelClicked()
{
    LOG_INFO(LogCategories::UI, "User clicked 'Cancel' in settings dialog");
    reject();
}

void SettingsDialog::onRestoreDefaultsClicked()
{
    LOG_INFO(LogCategories::UI, "User clicked 'Restore Defaults' in settings dialog");
    
    QMessageBox::StandardButton reply = QMessageBox::question(this, tr("Restore Defaults"),
        tr("Are you sure you want to restore all settings to their default values?"),
        QMessageBox::Yes | QMessageBox::No);
    
    if (reply == QMessageBox::Yes) {
        QSettings settings("CloneClean", "CloneClean");
        settings.clear();
        loadSettings();
        LOG_INFO(LogCategories::CONFIG, "Settings restored to defaults");
        QMessageBox::information(this, tr("Defaults Restored"), 
                               tr("All settings have been restored to their default values."));
    }
}

void SettingsDialog::onBrowseLogDirectory()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Select Log Directory"),
                                                    m_logDirectoryEdit->text());
    if (!dir.isEmpty()) {
        m_logDirectoryEdit->setText(dir);
        LOG_INFO(LogCategories::CONFIG, QString("Log directory changed to: %1").arg(dir));
    }
}

void SettingsDialog::onBrowseBackupDirectory()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Select Backup Directory"),
                                                    m_backupLocationEdit->text());
    if (!dir.isEmpty()) {
        m_backupLocationEdit->setText(dir);
        LOG_INFO(LogCategories::CONFIG, QString("Backup directory changed to: %1").arg(dir));
    }
}

void SettingsDialog::onAddProtectedPath()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Select Protected Path"));
    if (!dir.isEmpty()) {
        m_protectedPathsList->addItem(dir);
        LOG_INFO(LogCategories::CONFIG, QString("Added protected path: %1").arg(dir));
    }
}

void SettingsDialog::onRemoveProtectedPath()
{
    QListWidgetItem* item = m_protectedPathsList->currentItem();
    if (item) {
        QString path = item->text();
        delete item;
        LOG_INFO(LogCategories::CONFIG, QString("Removed protected path: %1").arg(path));
    }
}

void SettingsDialog::onOpenLogDirectory()
{
    QString logDir = m_logDirectoryEdit->text();
    QDesktopServices::openUrl(QUrl::fromLocalFile(logDir));
    LOG_INFO(LogCategories::UI, QString("Opened log directory: %1").arg(logDir));
}

void SettingsDialog::applyTheme()
{
    // Apply comprehensive theme using enhanced ThemeManager
    ThemeManager::instance()->applyComprehensiveTheme(this);
    
    // Force update of tab widget specifically
    if (m_tabWidget) {
        m_tabWidget->update();
        // Force repaint of all tabs
        for (int i = 0; i < m_tabWidget->count(); ++i) {
            if (QWidget* tab = m_tabWidget->widget(i)) {
                tab->update();
            }
        }
    }
}
void SettingsDialog::onThemeChanged()
{
    QString themeStr = m_themeCombo->currentData().toString();
    
    ThemeManager::Theme theme = ThemeManager::SystemDefault;
    if (themeStr == "light") {
        theme = ThemeManager::Light;
    } else if (themeStr == "dark") {
        theme = ThemeManager::Dark;
    }
    
    // Apply theme immediately
    ThemeManager::instance()->setTheme(theme);
    
    // Force immediate theme application to this dialog
    applyTheme();
    
    LOG_INFO(LogCategories::UI, QString("Theme changed to: %1").arg(themeStr));
}

void SettingsDialog::onGPUTestClicked()
{
    LOG_INFO(LogCategories::UI, "User clicked 'Test GPU Acceleration'");
    
    // Import GPU detector
    try {
        // Create a temporary GPU detector to test capabilities
        // Note: This is a simplified test - in real implementation, 
        // we'd use the actual GpuDetector from the hash calculator
        
        QString statusText;
        bool gpuAvailable = false;
        
        // Check for CUDA availability (simplified check)
        #ifdef HAS_CUDA
        statusText += tr("CUDA: Available\n");
        gpuAvailable = true;
        #else
        statusText += tr("CUDA: Not available\n");
        #endif
        
        // Check for OpenCL availability (simplified check)
        #ifdef HAS_OPENCL
        statusText += tr("OpenCL: Available\n");
        gpuAvailable = true;
        #else
        statusText += tr("OpenCL: Not available\n");
        #endif
        
        if (gpuAvailable) {
            statusText += tr("\nGPU acceleration is available and ready to use.");
            m_gpuStatusLabel->setStyleSheet("color: green;");
        } else {
            statusText += tr("\nNo GPU acceleration available. CPU will be used for all calculations.");
            m_gpuStatusLabel->setStyleSheet("color: orange;");
        }
        
        m_gpuStatusLabel->setText(statusText);
        
        QMessageBox::information(this, tr("GPU Test Results"), 
                               gpuAvailable ? tr("GPU acceleration is available!") : 
                                            tr("GPU acceleration is not available on this system."));
        
    } catch (const std::exception& e) {
        LOG_ERROR(LogCategories::UI, QString("GPU test failed: %1").arg(e.what()));
        m_gpuStatusLabel->setText(tr("GPU test failed. Check logs for details."));
        m_gpuStatusLabel->setStyleSheet("color: red;");
        QMessageBox::warning(this, tr("GPU Test Failed"), 
                           tr("Failed to test GPU acceleration. See logs for details."));
    }
}