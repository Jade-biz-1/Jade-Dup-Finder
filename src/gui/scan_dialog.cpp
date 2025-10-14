#include "scan_dialog.h"
#include "file_scanner.h"
#include "core/logger.h"
#include <QtWidgets/QApplication>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QTreeWidgetItem>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>
#include <QtCore/QDebug>
#include <QtCore/QSettings>
#include <QtCore/QDirIterator>
#include <QtGui/QShowEvent>
#include <QtWidgets/QStyle>

// ScanConfiguration validation methods
bool ScanSetupDialog::ScanConfiguration::isValid() const
{
    if (targetPaths.isEmpty()) {
        return false;
    }
    
    if (minimumFileSize < 0) {
        return false;
    }
    
    // Check if at least one target path exists
    for (const QString& path : targetPaths) {
        if (QDir(path).exists()) {
            return true;
        }
    }
    
    return false;
}

QString ScanSetupDialog::ScanConfiguration::validationError() const
{
    if (targetPaths.isEmpty()) {
        return "No scan locations selected";
    }
    
    if (minimumFileSize < 0) {
        return "Invalid minimum file size";
    }
    
    // Check if paths exist
    QStringList invalidPaths;
    for (const QString& path : targetPaths) {
        if (!QDir(path).exists()) {
            invalidPaths << path;
        }
    }
    
    if (invalidPaths.size() == targetPaths.size()) {
        return "None of the selected paths exist";
    }
    
    if (!invalidPaths.isEmpty()) {
        return QString("Some paths do not exist: %1").arg(invalidPaths.join(", "));
    }
    
    return QString();
}

// ScanSetupDialog Implementation
ScanSetupDialog::ScanSetupDialog(QWidget* parent)
    : QDialog(parent)
    , m_mainLayout(nullptr)
    , m_locationsGroup(nullptr)
    , m_locationsLayout(nullptr)
    , m_directoryTree(nullptr)
    , m_directoryButtonsLayout(nullptr)
    , m_addFolderButton(nullptr)
    , m_removeFolderButton(nullptr)
    , m_presetsWidget(nullptr)
    , m_presetsLayout(nullptr)
    , m_downloadsButton(nullptr)
    , m_photosButton(nullptr)
    , m_documentsButton(nullptr)
    , m_mediaButton(nullptr)
    , m_customButton(nullptr)
    , m_fullSystemButton(nullptr)
    , m_optionsGroup(nullptr)
    , m_optionsLayout(nullptr)
    , m_detectionMode(nullptr)
    , m_minimumSize(nullptr)
    , m_maxDepth(nullptr)
    , m_includeHidden(nullptr)
    , m_includeSystem(nullptr)
    , m_followSymlinks(nullptr)
    , m_scanArchives(nullptr)
    , m_excludePatterns(nullptr)
    , m_excludeFoldersTree(nullptr)
    , m_addExcludeFolderButton(nullptr)
    , m_removeExcludeFolderButton(nullptr)
    , m_fileTypesWidget(nullptr)
    , m_allTypesCheck(nullptr)
    , m_imagesCheck(nullptr)
    , m_documentsCheck(nullptr)
    , m_videosCheck(nullptr)
    , m_audioCheck(nullptr)
    , m_archivesCheck(nullptr)
    , m_previewGroup(nullptr)
    , m_previewLayout(nullptr)
    , m_estimateLabel(nullptr)
    , m_estimationProgress(nullptr)
    , m_limitWarning(nullptr)
    , m_upgradeButton(nullptr)
    , m_buttonBox(nullptr)
    , m_startScanButton(nullptr)
    , m_savePresetButton(nullptr)
    , m_cancelButton(nullptr)
    , m_estimationTimer(new QTimer(this))
    , m_estimationInProgress(false)
{
    setWindowTitle(tr("New Scan Configuration"));
    setMinimumSize(900, 600);
    resize(950, 650);
    setModal(true);
    
    // Make dialog clearly distinguishable from main window
    setWindowFlags(Qt::Dialog | Qt::WindowTitleHint | Qt::WindowCloseButtonHint | Qt::WindowSystemMenuHint);
    
    // Add distinctive lighter background for the dialog only
    setStyleSheet(
        "QDialog {"
        "    background-color: palette(light);"
        "    border: 2px solid palette(highlight);"
        "    border-radius: 8px;"
        "}"
    );
    
    // Initialize configuration with defaults
    m_currentConfig.detectionMode = DetectionMode::Smart;
    m_currentConfig.minimumFileSize = 0; // 0 MB - include all files
    m_currentConfig.maximumDepth = -1; // Unlimited
    m_currentConfig.includeHidden = false;
    m_currentConfig.includeSystem = false;
    m_currentConfig.followSymlinks = true;
    m_currentConfig.scanArchives = false;
    m_currentConfig.excludePatterns << "*.tmp" << "*.log" << "Thumbs.db";
    
    // Set up estimation timer
    m_estimationTimer->setSingleShot(true);
    m_estimationTimer->setInterval(500); // 500ms delay for real-time updates
    
    setupUI();
    setupConnections();
    populateDirectoryTree();
    updateEstimates();
    applyTheme();
}

ScanSetupDialog::~ScanSetupDialog()
{
    // Qt will handle cleanup
}

void ScanSetupDialog::setupUI()
{
    m_mainLayout = new QHBoxLayout(this);
    m_mainLayout->setContentsMargins(20, 20, 20, 20);
    m_mainLayout->setSpacing(20);
    
    createLocationsPanel();
    createOptionsPanel();
    createPreviewPanel();
    createButtonBar();
}

void ScanSetupDialog::createLocationsPanel()
{
    m_locationsGroup = new QGroupBox(tr("📁 Scan Locations"), this);
    m_locationsLayout = new QVBoxLayout(m_locationsGroup);
    m_locationsLayout->setContentsMargins(16, 25, 16, 16);
    m_locationsLayout->setSpacing(12);
    
    // Directory tree
    m_directoryTree = new QTreeWidget(this);
    m_directoryTree->setHeaderLabel(tr("Select folders to scan"));
    m_directoryTree->setMinimumHeight(220);
    m_directoryTree->setMaximumHeight(280);
    m_directoryTree->setAlternatingRowColors(true);
    m_directoryTree->setRootIsDecorated(true);
    
    // Improve tree widget appearance
    m_directoryTree->setStyleSheet(
        "QTreeWidget {"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    padding: 4px;"
        "    background: palette(base);"
        "    selection-background-color: palette(highlight);"
        "}"
        "QTreeWidget::item {"
        "    padding: 4px;"
        "    margin: 1px;"
        "}"
    );
    
    // Directory buttons
    m_directoryButtonsLayout = new QHBoxLayout();
    m_directoryButtonsLayout->setSpacing(8);
    
    m_addFolderButton = new QPushButton(tr("+ Add Folder..."), this);
    m_addFolderButton->setToolTip(tr("Add a folder to scan for duplicate files"));
    m_removeFolderButton = new QPushButton(tr("- Remove"), this);
    m_removeFolderButton->setToolTip(tr("Remove selected folder from scan list"));
    m_removeFolderButton->setEnabled(false);
    
    // Style the directory buttons
    QString directoryButtonStyle = 
        "QPushButton {"
        "    padding: 6px 12px;"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(button);"
        "    color: palette(button-text);"
        "    font-weight: normal;"
        "}"
        "QPushButton:hover {"
        "    background: palette(light);"
        "    border-color: palette(highlight);"
        "}"
        "QPushButton:disabled {"
        "    color: palette(mid);"
        "    background: palette(window);"
        "}"
    ;
    
    m_addFolderButton->setStyleSheet(directoryButtonStyle);
    m_removeFolderButton->setStyleSheet(directoryButtonStyle);
    
    m_directoryButtonsLayout->addWidget(m_addFolderButton);
    m_directoryButtonsLayout->addWidget(m_removeFolderButton);
    m_directoryButtonsLayout->addStretch();
    
    // Quick presets
    QLabel* presetsLabel = new QLabel(tr("📋 Quick Presets:"), this);
    presetsLabel->setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 8px;");
    
    m_presetsWidget = new QWidget(this);
    m_presetsLayout = new QGridLayout(m_presetsWidget);
    m_presetsLayout->setSpacing(8);
    m_presetsLayout->setContentsMargins(0, 4, 0, 0);
    
    m_downloadsButton = new QPushButton(tr("Downloads"), this);
    m_downloadsButton->setToolTip(tr("Scan Downloads folder"));
    m_photosButton = new QPushButton(tr("Photos"), this);
    m_photosButton->setToolTip(tr("Scan Pictures folder"));
    m_documentsButton = new QPushButton(tr("Documents"), this);
    m_documentsButton->setToolTip(tr("Scan Documents folder"));
    m_mediaButton = new QPushButton(tr("Media"), this);
    m_mediaButton->setToolTip(tr("Scan Music and Videos folders"));
    m_customButton = new QPushButton(tr("Custom"), this);
    m_fullSystemButton = new QPushButton(tr("Full System"), this);
    
    // Set minimum size for preset buttons
    const QSize presetButtonSize(90, 32);
    m_downloadsButton->setMinimumSize(presetButtonSize);
    m_photosButton->setMinimumSize(presetButtonSize);
    m_documentsButton->setMinimumSize(presetButtonSize);
    m_mediaButton->setMinimumSize(presetButtonSize);
    m_customButton->setMinimumSize(presetButtonSize);
    m_fullSystemButton->setMinimumSize(presetButtonSize);
    
    // Arrange in 2x3 grid
    m_presetsLayout->addWidget(m_downloadsButton, 0, 0);
    m_presetsLayout->addWidget(m_photosButton, 0, 1);
    m_presetsLayout->addWidget(m_documentsButton, 0, 2);
    m_presetsLayout->addWidget(m_mediaButton, 1, 0);
    m_presetsLayout->addWidget(m_customButton, 1, 1);
    m_presetsLayout->addWidget(m_fullSystemButton, 1, 2);
    
    // Add to layout
    m_locationsLayout->addWidget(m_directoryTree);
    m_locationsLayout->addLayout(m_directoryButtonsLayout);
    m_locationsLayout->addWidget(presetsLabel);
    m_locationsLayout->addWidget(m_presetsWidget);
    
    m_mainLayout->addWidget(m_locationsGroup);
}

void ScanSetupDialog::createOptionsPanel()
{
    m_optionsGroup = new QGroupBox(tr("⚙️ Options"), this);
    m_optionsLayout = new QVBoxLayout(m_optionsGroup);
    m_optionsLayout->setContentsMargins(16, 25, 16, 16);
    m_optionsLayout->setSpacing(10);
    
    // Detection mode
    QLabel* detectionLabel = new QLabel(tr("Detection:"), this);
    m_detectionMode = new QComboBox(this);
    m_detectionMode->addItem(tr("Quick (Size + Name)"), static_cast<int>(DetectionMode::Quick));
    m_detectionMode->addItem(tr("Smart (Recommended)"), static_cast<int>(DetectionMode::Smart));
    m_detectionMode->addItem(tr("Deep (Hash-based)"), static_cast<int>(DetectionMode::Deep));
    m_detectionMode->addItem(tr("Media (With Metadata)"), static_cast<int>(DetectionMode::Media));
    m_detectionMode->setCurrentIndex(1); // Smart by default
    
    // Minimum size
    QLabel* sizeLabel = new QLabel(tr("Min Size:"), this);
    m_minimumSize = new QSpinBox(this);
    m_minimumSize->setRange(0, 1024);
    m_minimumSize->setValue(0);
    m_minimumSize->setSuffix(tr(" MB"));
    
    // Maximum depth
    QLabel* depthLabel = new QLabel(tr("Max Depth:"), this);
    m_maxDepth = new QComboBox(this);
    m_maxDepth->addItem(tr("Unlimited"), -1);
    m_maxDepth->addItem(tr("1 level"), 1);
    m_maxDepth->addItem(tr("2 levels"), 2);
    m_maxDepth->addItem(tr("3 levels"), 3);
    m_maxDepth->addItem(tr("5 levels"), 5);
    
    // Include options
    QLabel* includeLabel = new QLabel(tr("Include:"), this);
    includeLabel->setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 12px; margin-bottom: 4px;");
    
    m_includeHidden = new QCheckBox(tr("Hidden files"), this);
    m_includeHidden->setToolTip(tr("Include hidden files and folders in scan"));
    m_includeSystem = new QCheckBox(tr("System files"), this);
    m_includeSystem->setToolTip(tr("Include system files (use with caution)"));
    m_followSymlinks = new QCheckBox(tr("Symlinks"), this);
    m_followSymlinks->setToolTip(tr("Follow symbolic links to other directories"));
    m_followSymlinks->setChecked(true);
    m_scanArchives = new QCheckBox(tr("Archives"), this);
    
    // Style checkboxes for better visibility
    QString checkboxStyle = "QCheckBox { padding: 2px; margin: 2px; } QCheckBox::indicator { width: 16px; height: 16px; }";
    m_includeHidden->setStyleSheet(checkboxStyle);
    m_includeSystem->setStyleSheet(checkboxStyle);
    m_followSymlinks->setStyleSheet(checkboxStyle);
    m_scanArchives->setStyleSheet(checkboxStyle);
    
    // File types
    QLabel* typesLabel = new QLabel(tr("File Types:"), this);
    typesLabel->setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 12px; margin-bottom: 4px;");
    
    m_fileTypesWidget = new QWidget(this);
    QGridLayout* typesLayout = new QGridLayout(m_fileTypesWidget);
    typesLayout->setSpacing(6);
    typesLayout->setContentsMargins(0, 0, 0, 0);
    
    m_allTypesCheck = new QCheckBox(tr("All"), this);
    m_allTypesCheck->setChecked(true);
    m_imagesCheck = new QCheckBox(tr("Images"), this);
    m_documentsCheck = new QCheckBox(tr("Documents"), this);
    m_videosCheck = new QCheckBox(tr("Videos"), this);
    m_audioCheck = new QCheckBox(tr("Audio"), this);
    m_archivesCheck = new QCheckBox(tr("Archives"), this);
    
    // Apply checkbox styling to file type checkboxes
    m_allTypesCheck->setStyleSheet(checkboxStyle);
    m_imagesCheck->setStyleSheet(checkboxStyle);
    m_documentsCheck->setStyleSheet(checkboxStyle);
    m_videosCheck->setStyleSheet(checkboxStyle);
    m_audioCheck->setStyleSheet(checkboxStyle);
    m_archivesCheck->setStyleSheet(checkboxStyle);
    
    typesLayout->addWidget(m_allTypesCheck, 0, 0);
    typesLayout->addWidget(m_imagesCheck, 0, 1);
    typesLayout->addWidget(m_documentsCheck, 0, 2);
    typesLayout->addWidget(m_videosCheck, 1, 0);
    typesLayout->addWidget(m_audioCheck, 1, 1);
    typesLayout->addWidget(m_archivesCheck, 1, 2);
    
    // Exclude patterns
    QLabel* excludeLabel = new QLabel(tr("Exclude Patterns:"), this);
    excludeLabel->setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 12px; margin-bottom: 4px;");
    m_excludePatterns = new QLineEdit("*.tmp, *.log, Thumbs.db", this);
    m_excludePatterns->setStyleSheet(
        "QLineEdit {"
        "    padding: 6px;"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(base);"
        "}"
    );
    
    // Exclude folders
    QLabel* excludeFoldersLabel = new QLabel(tr("Exclude Folders:"), this);
    excludeFoldersLabel->setStyleSheet("font-weight: bold; font-size: 11pt; margin-top: 12px; margin-bottom: 4px;");
    
    m_excludeFoldersTree = new QTreeWidget(this);
    m_excludeFoldersTree->setHeaderLabel(tr("Folders to exclude from scan"));
    m_excludeFoldersTree->setMaximumHeight(100);
    m_excludeFoldersTree->setMinimumHeight(80);
    m_excludeFoldersTree->setAlternatingRowColors(true);
    m_excludeFoldersTree->setRootIsDecorated(false);
    m_excludeFoldersTree->setStyleSheet(
        "QTreeWidget {"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    padding: 2px;"
        "    background: palette(base);"
        "}"
        "QTreeWidget::item {"
        "    padding: 2px;"
        "    margin: 1px;"
        "}"
    );
    
    // Exclude folder buttons
    QHBoxLayout* excludeFolderButtonsLayout = new QHBoxLayout();
    excludeFolderButtonsLayout->setSpacing(6);
    
    m_addExcludeFolderButton = new QPushButton(tr("+ Add Folder"), this);
    m_removeExcludeFolderButton = new QPushButton(tr("- Remove"), this);
    m_removeExcludeFolderButton->setEnabled(false);
    
    QString excludeButtonStyle = 
        "QPushButton {"
        "    padding: 4px 8px;"
        "    border: 1px solid palette(mid);"
        "    border-radius: 3px;"
        "    background: palette(button);"
        "    color: palette(button-text);"
        "    font-size: 9pt;"
        "}"
        "QPushButton:hover {"
        "    background: palette(light);"
        "    border-color: palette(highlight);"
        "}"
        "QPushButton:disabled {"
        "    color: palette(mid);"
        "    background: palette(window);"
        "}"
    ;
    
    m_addExcludeFolderButton->setStyleSheet(excludeButtonStyle);
    m_removeExcludeFolderButton->setStyleSheet(excludeButtonStyle);
    
    excludeFolderButtonsLayout->addWidget(m_addExcludeFolderButton);
    excludeFolderButtonsLayout->addWidget(m_removeExcludeFolderButton);
    excludeFolderButtonsLayout->addStretch();
    
    // Layout
    QGridLayout* optionsGrid = new QGridLayout();
    optionsGrid->setSpacing(8);
    optionsGrid->setColumnStretch(1, 1);
    
    optionsGrid->addWidget(detectionLabel, 0, 0);
    optionsGrid->addWidget(m_detectionMode, 0, 1);
    optionsGrid->addWidget(sizeLabel, 1, 0);
    optionsGrid->addWidget(m_minimumSize, 1, 1);
    optionsGrid->addWidget(depthLabel, 2, 0);
    optionsGrid->addWidget(m_maxDepth, 2, 1);
    
    m_optionsLayout->addLayout(optionsGrid);
    m_optionsLayout->addWidget(includeLabel);
    m_optionsLayout->addWidget(m_includeHidden);
    m_optionsLayout->addWidget(m_followSymlinks);
    m_optionsLayout->addWidget(typesLabel);
    m_optionsLayout->addWidget(m_fileTypesWidget);
    m_optionsLayout->addWidget(excludeLabel);
    m_optionsLayout->addWidget(m_excludePatterns);
    m_optionsLayout->addWidget(excludeFoldersLabel);
    m_optionsLayout->addWidget(m_excludeFoldersTree);
    m_optionsLayout->addLayout(excludeFolderButtonsLayout);
    m_optionsLayout->addStretch();
    
    m_mainLayout->addWidget(m_optionsGroup);
}

void ScanSetupDialog::createPreviewPanel()
{
    QVBoxLayout* rightLayout = new QVBoxLayout();
    
    m_previewGroup = new QGroupBox(tr("📊 Preview & Limits"), this);
    m_previewLayout = new QVBoxLayout(m_previewGroup);
    m_previewLayout->setContentsMargins(16, 25, 16, 16);
    m_previewLayout->setSpacing(12);
    
    m_estimateLabel = new QLabel(tr("Estimated: Calculating..."), this);
    m_estimateLabel->setWordWrap(true);
    m_estimateLabel->setStyleSheet(
        "QLabel {"
        "    font-size: 11pt;"
        "    padding: 8px;"
        "    background: palette(base);"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    color: palette(window-text);"
        "}"
    );
    
    m_estimationProgress = new QProgressBar(this);
    m_estimationProgress->setRange(0, 0); // Indeterminate progress
    m_estimationProgress->setVisible(false);
    m_estimationProgress->setFixedHeight(20);
    m_estimationProgress->setStyleSheet(
        "QProgressBar {"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(base);"
        "}"
        "QProgressBar::chunk {"
        "    background: palette(highlight);"
        "    border-radius: 2px;"
        "}"
    );
    
    m_limitWarning = new QLabel(this);
    m_limitWarning->setWordWrap(true);
    m_limitWarning->setStyleSheet(
        "QLabel {"
        "    color: palette(window-text);"
        "    background: palette(base);"
        "    padding: 12px;"
        "    border: 2px solid orange;"
        "    border-radius: 6px;"
        "    font-weight: bold;"
        "}"
    );
    
    m_upgradeButton = new QPushButton(tr("🔒 Upgrade to Premium"), this);
    m_upgradeButton->setVisible(false);
    m_upgradeButton->setStyleSheet(
        "QPushButton {"
        "    background: palette(highlight);"
        "    color: palette(highlighted-text);"
        "    border: none;"
        "    padding: 8px 16px;"
        "    border-radius: 4px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background: palette(dark);"
        "}"
    );
    
    m_previewLayout->addWidget(m_estimateLabel);
    m_previewLayout->addWidget(m_estimationProgress);
    m_previewLayout->addWidget(m_limitWarning);
    m_previewLayout->addWidget(m_upgradeButton);
    m_previewLayout->addStretch();
    
    rightLayout->addWidget(m_previewGroup);
    rightLayout->addStretch();
    
    m_mainLayout->addLayout(rightLayout);
}

void ScanSetupDialog::createButtonBar()
{
    m_buttonBox = new QDialogButtonBox(this);
    m_buttonBox->setContentsMargins(16, 8, 16, 8);
    
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    m_cancelButton->setToolTip(tr("Close dialog without starting scan"));
    m_savePresetButton = new QPushButton(tr("Save as Preset"), this);
    m_savePresetButton->setToolTip(tr("Save current configuration as a preset for future use"));
    m_startScanButton = new QPushButton(tr("▶ Start Scan"), this);
    m_startScanButton->setToolTip(tr("Start scanning with current configuration"));
    
    // Style buttons with better visibility
    QString cancelButtonStyle = 
        "QPushButton {"
        "    background: palette(button);"
        "    border: 1px solid palette(mid);"
        "    padding: 8px 16px;"
        "    border-radius: 4px;"
        "    color: palette(button-text);"
        "    min-width: 80px;"
        "}"
        "QPushButton:hover {"
        "    background: palette(light);"
        "    border-color: palette(highlight);"
        "}"
    ;
    
    QString actionButtonStyle = 
        "QPushButton {"
        "    background: palette(button);"
        "    border: 1px solid palette(mid);"
        "    padding: 8px 16px;"
        "    border-radius: 4px;"
        "    color: palette(button-text);"
        "    min-width: 100px;"
        "}"
        "QPushButton:hover {"
        "    background: palette(light);"
        "    border-color: palette(highlight);"
        "}"
    ;
    
    QString primaryButtonStyle = 
        "QPushButton {"
        "    background: palette(highlight);"
        "    border: 1px solid palette(highlight);"
        "    padding: 8px 20px;"
        "    border-radius: 4px;"
        "    color: palette(highlighted-text);"
        "    font-weight: bold;"
        "    min-width: 120px;"
        "}"
        "QPushButton:hover {"
        "    background: palette(dark);"
        "    border-color: palette(dark);"
        "}"
        "QPushButton:pressed {"
        "    background: palette(shadow);"
        "}"
    ;
    
    m_cancelButton->setStyleSheet(cancelButtonStyle);
    m_savePresetButton->setStyleSheet(actionButtonStyle);
    m_startScanButton->setStyleSheet(primaryButtonStyle);
    m_startScanButton->setDefault(true);
    
    m_buttonBox->addButton(m_cancelButton, QDialogButtonBox::RejectRole);
    m_buttonBox->addButton(m_savePresetButton, QDialogButtonBox::ActionRole);
    m_buttonBox->addButton(m_startScanButton, QDialogButtonBox::AcceptRole);
    
    // Add button box to main layout in a separate row
    QVBoxLayout* mainVLayout = new QVBoxLayout();
    mainVLayout->setContentsMargins(0, 0, 0, 0);
    mainVLayout->setSpacing(0);
    
    QWidget* mainWidget = new QWidget(this);
    mainWidget->setLayout(m_mainLayout);
    
    // Add separator line above buttons
    QFrame* separator = new QFrame(this);
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);
    separator->setStyleSheet("QFrame { color: palette(mid); margin: 8px 0px; }");
    
    mainVLayout->addWidget(mainWidget);
    mainVLayout->addWidget(separator);
    mainVLayout->addWidget(m_buttonBox);
    
    // Replace the dialog's layout
    delete layout();
    setLayout(mainVLayout);
}

void ScanSetupDialog::setupConnections()
{
    // Estimation timer
    connect(m_estimationTimer, &QTimer::timeout, this, &ScanSetupDialog::performEstimation);
    
    // Directory management
    connect(m_addFolderButton, &QPushButton::clicked, this, &ScanSetupDialog::addFolder);
    connect(m_removeFolderButton, &QPushButton::clicked, this, &ScanSetupDialog::removeSelectedFolder);
    connect(m_directoryTree, &QTreeWidget::itemSelectionChanged, this, [this]() {
        m_removeFolderButton->setEnabled(m_directoryTree->currentItem() != nullptr);
    });
    connect(m_directoryTree, &QTreeWidget::itemChanged, this, &ScanSetupDialog::onDirectoryItemChanged);
    
    // Quick presets
    connect(m_downloadsButton, &QPushButton::clicked, this, &ScanSetupDialog::applyDownloadsPreset);
    connect(m_photosButton, &QPushButton::clicked, this, &ScanSetupDialog::applyPhotosPreset);
    connect(m_documentsButton, &QPushButton::clicked, this, &ScanSetupDialog::applyDocumentsPreset);
    connect(m_mediaButton, &QPushButton::clicked, this, &ScanSetupDialog::applyMediaPreset);
    connect(m_customButton, &QPushButton::clicked, this, &ScanSetupDialog::applyCustomPreset);
    connect(m_fullSystemButton, &QPushButton::clicked, this, &ScanSetupDialog::applyFullSystemPreset);
    
    // Options changes
    connect(m_detectionMode, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &ScanSetupDialog::onOptionsChanged);
    connect(m_minimumSize, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ScanSetupDialog::onOptionsChanged);
    connect(m_maxDepth, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ScanSetupDialog::onOptionsChanged);
    
    connect(m_includeHidden, &QCheckBox::toggled, this, &ScanSetupDialog::onOptionsChanged);
    connect(m_includeSystem, &QCheckBox::toggled, this, &ScanSetupDialog::onOptionsChanged);
    connect(m_followSymlinks, &QCheckBox::toggled, this, &ScanSetupDialog::onOptionsChanged);
    connect(m_scanArchives, &QCheckBox::toggled, this, &ScanSetupDialog::onOptionsChanged);
    
    connect(m_excludePatterns, &QLineEdit::textChanged, this, &ScanSetupDialog::onOptionsChanged);
    
    // Exclude folder connections
    connect(m_addExcludeFolderButton, &QPushButton::clicked, this, &ScanSetupDialog::addExcludeFolder);
    connect(m_removeExcludeFolderButton, &QPushButton::clicked, this, &ScanSetupDialog::removeSelectedExcludeFolder);
    connect(m_excludeFoldersTree, &QTreeWidget::itemSelectionChanged, this, [this]() {
        m_removeExcludeFolderButton->setEnabled(m_excludeFoldersTree->currentItem() != nullptr);
    });
    connect(m_excludeFoldersTree, &QTreeWidget::itemChanged, this, &ScanSetupDialog::onExcludeFolderItemChanged);
    
    // File type checkboxes
    connect(m_allTypesCheck, &QCheckBox::toggled, this, &ScanSetupDialog::onAllTypesToggled);
    connect(m_imagesCheck, &QCheckBox::toggled, this, &ScanSetupDialog::onFileTypeChanged);
    connect(m_documentsCheck, &QCheckBox::toggled, this, &ScanSetupDialog::onFileTypeChanged);
    connect(m_videosCheck, &QCheckBox::toggled, this, &ScanSetupDialog::onFileTypeChanged);
    connect(m_audioCheck, &QCheckBox::toggled, this, &ScanSetupDialog::onFileTypeChanged);
    connect(m_archivesCheck, &QCheckBox::toggled, this, &ScanSetupDialog::onFileTypeChanged);
    
    // Buttons
    connect(m_startScanButton, &QPushButton::clicked, this, &ScanSetupDialog::startScan);
    connect(m_savePresetButton, &QPushButton::clicked, this, &ScanSetupDialog::savePreset);
    connect(m_cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    connect(m_upgradeButton, &QPushButton::clicked, this, &ScanSetupDialog::showUpgradeDialog);
}

void ScanSetupDialog::populateDirectoryTree()
{
    m_directoryTree->clear();
    
    // Add common directories as examples
    QStringList commonPaths = {
        QStandardPaths::writableLocation(QStandardPaths::HomeLocation),
        QStandardPaths::writableLocation(QStandardPaths::DownloadLocation),
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
        QStandardPaths::writableLocation(QStandardPaths::PicturesLocation),
        QStandardPaths::writableLocation(QStandardPaths::MusicLocation),
        QStandardPaths::writableLocation(QStandardPaths::MoviesLocation)
    };
    
    for (const QString& path : commonPaths) {
        if (!path.isEmpty() && QDir(path).exists()) {
            QTreeWidgetItem* item = new QTreeWidgetItem(m_directoryTree);
            item->setText(0, QDir(path).dirName());
            item->setData(0, Qt::UserRole, path);
            item->setToolTip(0, path);
            item->setCheckState(0, Qt::Unchecked);
        }
    }
}

void ScanSetupDialog::updateEstimates()
{
    if (m_estimationInProgress) {
        return;
    }
    
    // Restart timer for debounced updates
    m_estimationTimer->stop();
    m_estimationTimer->start();
}

void ScanSetupDialog::performEstimation()
{
    if (m_estimationInProgress) {
        return;
    }
    
    m_estimationInProgress = true;
    m_estimationProgress->setVisible(true);
    
    // Get current configuration
    ScanConfiguration config = getCurrentConfiguration();
    
    // Simple estimation based on selected paths
    qint64 totalSize = 0;
    int estimatedFiles = 0;
    
    for (const QString& path : config.targetPaths) {
        QDir dir(path);
        if (dir.exists()) {
            // Quick directory size estimation
            QDirIterator iterator(path, QDir::Files | QDir::NoDotAndDotDot, 
                                 QDirIterator::Subdirectories);
            int fileCount = 0;
            while (iterator.hasNext() && fileCount < 1000) { // Limit for performance
                iterator.next();
                QFileInfo info = iterator.fileInfo();
                totalSize += info.size();
                fileCount++;
            }
            estimatedFiles += fileCount;
        }
    }
    
    // Update estimate display
    QString estimate = tr("Files: ~%1, Size: ~%2")
                      .arg(estimatedFiles > 1000 ? QString("%1k+").arg(estimatedFiles/1000) : QString::number(estimatedFiles))
                      .arg(formatFileSize(totalSize));
    
    // Check limits (example: 10GB for free version)
    const qint64 freeLimit = 10LL * 1024 * 1024 * 1024; // 10GB
    
    if (totalSize > freeLimit) {
        m_limitWarning->setText(tr("⚠️ Large scan detected (%1). Consider upgrading for unlimited scanning.").arg(formatFileSize(totalSize)));
        m_limitWarning->setVisible(true);
        m_upgradeButton->setVisible(true);
    } else {
        m_limitWarning->setVisible(false);
        m_upgradeButton->setVisible(false);
    }
    
    m_estimateLabel->setText(tr("Estimated: %1").arg(estimate));
    m_estimationProgress->setVisible(false);
    m_estimationInProgress = false;
}

void ScanSetupDialog::applyTheme()
{
    QPalette palette = QApplication::palette();
    
    // Apply theme-aware styling
    QString buttonStyle = QString(
        "QPushButton {"
        "    background: palette(button);"
        "    border: 1px solid palette(mid);"
        "    padding: 4px 8px;"
        "    border-radius: 4px;"
        "    color: palette(button-text);"
        "}"
        "QPushButton:hover {"
        "    background: palette(light);"
        "}"
        "QPushButton:pressed {"
        "    background: palette(mid);"
        "}"
    );
    
    // Apply to preset buttons
    m_downloadsButton->setStyleSheet(buttonStyle);
    m_photosButton->setStyleSheet(buttonStyle);
    m_documentsButton->setStyleSheet(buttonStyle);
    m_mediaButton->setStyleSheet(buttonStyle);
    m_customButton->setStyleSheet(buttonStyle);
    m_fullSystemButton->setStyleSheet(buttonStyle);
    
    // Let group boxes use their default styling
}

// Slot implementations
void ScanSetupDialog::addFolder()
{
    QString folder = QFileDialog::getExistingDirectory(this, tr("Select Folder to Scan"));
    if (!folder.isEmpty()) {
        // Check if already exists
        for (int i = 0; i < m_directoryTree->topLevelItemCount(); ++i) {
            QTreeWidgetItem* item = m_directoryTree->topLevelItem(i);
            if (item->data(0, Qt::UserRole).toString() == folder) {
                item->setCheckState(0, Qt::Checked);
                return;
            }
        }
        
        // Add new folder
        QTreeWidgetItem* item = new QTreeWidgetItem(m_directoryTree);
        item->setText(0, QDir(folder).dirName());
        item->setData(0, Qt::UserRole, folder);
        item->setToolTip(0, folder);
        item->setCheckState(0, Qt::Checked);
        
        updateEstimates();
    }
}

void ScanSetupDialog::removeSelectedFolder()
{
    QTreeWidgetItem* current = m_directoryTree->currentItem();
    if (current) {
        delete current;
        updateEstimates();
    }
}

void ScanSetupDialog::onDirectoryItemChanged(QTreeWidgetItem* item, int column)
{
    Q_UNUSED(column)
    if (item) {
        updateEstimates();
    }
}

// Quick preset implementations
void ScanSetupDialog::applyDownloadsPreset()
{
    clearAllSelections();
    QString downloads = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
    selectPath(downloads);
    
    // Configure for downloads
    m_detectionMode->setCurrentIndex(1); // Smart
    m_minimumSize->setValue(5); // 5MB+
    m_includeHidden->setChecked(false);
    m_scanArchives->setChecked(true);
    
    updateEstimates();
}

void ScanSetupDialog::applyPhotosPreset()
{
    clearAllSelections();
    QString pictures = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    selectPath(pictures);
    
    // Configure for photos
    m_detectionMode->setCurrentIndex(3); // Media
    m_minimumSize->setValue(0); // Include all files
    
    // Select only image types
    m_allTypesCheck->setChecked(false);
    m_imagesCheck->setChecked(true);
    m_documentsCheck->setChecked(false);
    m_videosCheck->setChecked(false);
    m_audioCheck->setChecked(false);
    m_archivesCheck->setChecked(false);
    
    updateEstimates();
}

void ScanSetupDialog::applyDocumentsPreset()
{
    clearAllSelections();
    QString documents = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    selectPath(documents);
    
    // Configure for documents
    m_detectionMode->setCurrentIndex(2); // Deep
    m_minimumSize->setValue(0); // Include all files
    
    // Select only document types
    m_allTypesCheck->setChecked(false);
    m_imagesCheck->setChecked(false);
    m_documentsCheck->setChecked(true);
    m_videosCheck->setChecked(false);
    m_audioCheck->setChecked(false);
    m_archivesCheck->setChecked(false);
    
    updateEstimates();
}

void ScanSetupDialog::applyMediaPreset()
{
    clearAllSelections();
    QStringList mediaPaths = {
        QStandardPaths::writableLocation(QStandardPaths::MusicLocation),
        QStandardPaths::writableLocation(QStandardPaths::MoviesLocation),
        QStandardPaths::writableLocation(QStandardPaths::PicturesLocation)
    };
    
    for (const QString& path : mediaPaths) {
        selectPath(path);
    }
    
    // Configure for media
    m_detectionMode->setCurrentIndex(3); // Media
    m_minimumSize->setValue(10); // 10MB+
    
    // Select media types
    m_allTypesCheck->setChecked(false);
    m_imagesCheck->setChecked(true);
    m_documentsCheck->setChecked(false);
    m_videosCheck->setChecked(true);
    m_audioCheck->setChecked(true);
    m_archivesCheck->setChecked(false);
    
    updateEstimates();
}

void ScanSetupDialog::applyCustomPreset()
{
    // Just clear selections and let user configure manually
    clearAllSelections();
    
    // Reset to defaults
    m_detectionMode->setCurrentIndex(1); // Smart
    m_minimumSize->setValue(0); // Include all files
    m_allTypesCheck->setChecked(true);
    
    QMessageBox::information(this, tr("Custom Preset"), 
                           tr("Please select folders and configure options manually."));
}

void ScanSetupDialog::applyFullSystemPreset()
{
    clearAllSelections();
    selectPath("/home");
    
    // Configure for system scan
    m_detectionMode->setCurrentIndex(1); // Smart
    m_minimumSize->setValue(50); // 50MB+ to avoid small files
    m_maxDepth->setCurrentIndex(3); // 3 levels to avoid too deep
    m_includeHidden->setChecked(false);
    m_includeSystem->setChecked(false);
    
    updateEstimates();
    
    QMessageBox::warning(this, tr("Full System Scan"), 
                        tr("This will scan your entire home directory. This may take a long time!"));
}

// Options change handlers
void ScanSetupDialog::onOptionsChanged()
{
    updateEstimates();
}

void ScanSetupDialog::onAllTypesToggled(bool checked)
{
    // Prevent recursion
    m_imagesCheck->blockSignals(true);
    m_documentsCheck->blockSignals(true);
    m_videosCheck->blockSignals(true);
    m_audioCheck->blockSignals(true);
    m_archivesCheck->blockSignals(true);
    
    m_imagesCheck->setChecked(checked);
    m_documentsCheck->setChecked(checked);
    m_videosCheck->setChecked(checked);
    m_audioCheck->setChecked(checked);
    m_archivesCheck->setChecked(checked);
    
    m_imagesCheck->blockSignals(false);
    m_documentsCheck->blockSignals(false);
    m_videosCheck->blockSignals(false);
    m_audioCheck->blockSignals(false);
    m_archivesCheck->blockSignals(false);
    
    updateEstimates();
}

void ScanSetupDialog::onFileTypeChanged()
{
    // Update "All" checkbox based on individual selections
    bool allChecked = m_imagesCheck->isChecked() && 
                     m_documentsCheck->isChecked() && 
                     m_videosCheck->isChecked() && 
                     m_audioCheck->isChecked() && 
                     m_archivesCheck->isChecked();
    
    m_allTypesCheck->blockSignals(true);
    m_allTypesCheck->setChecked(allChecked);
    m_allTypesCheck->blockSignals(false);
    
    updateEstimates();
}

// Folder exclusion implementations
void ScanSetupDialog::addExcludeFolder()
{
    QString folder = QFileDialog::getExistingDirectory(this, tr("Select Folder to Exclude"));
    if (!folder.isEmpty()) {
        // Check if already exists
        for (int i = 0; i < m_excludeFoldersTree->topLevelItemCount(); ++i) {
            QTreeWidgetItem* item = m_excludeFoldersTree->topLevelItem(i);
            if (item->data(0, Qt::UserRole).toString() == folder) {
                QMessageBox::information(this, tr("Folder Already Excluded"), 
                                        tr("The selected folder is already in the exclusion list."));
                return;
            }
        }
        
        // Add new excluded folder
        QTreeWidgetItem* item = new QTreeWidgetItem(m_excludeFoldersTree);
        item->setText(0, QDir(folder).dirName());
        item->setData(0, Qt::UserRole, folder);
        item->setToolTip(0, folder);
        item->setIcon(0, style()->standardIcon(QStyle::SP_DirIcon));
        
        updateEstimates();
    }
}

void ScanSetupDialog::removeSelectedExcludeFolder()
{
    QTreeWidgetItem* current = m_excludeFoldersTree->currentItem();
    if (current) {
        delete current;
        updateEstimates();
    }
}

void ScanSetupDialog::onExcludeFolderItemChanged(QTreeWidgetItem* item, int column)
{
    Q_UNUSED(item)
    Q_UNUSED(column)
    updateEstimates();
}

void ScanSetupDialog::startScan()
{
    // Validate configuration
    ScanConfiguration config = getCurrentConfiguration();
    
    QString error = config.validationError();
    if (!error.isEmpty()) {
        QMessageBox::warning(this, tr("Invalid Configuration"), error);
        return;
    }
    
    // Emit signal and close dialog
    emit scanConfigured(config);
    accept();
}

void ScanSetupDialog::savePreset()
{
    bool ok;
    QString name = QInputDialog::getText(this, tr("Save Preset"), 
                                       tr("Preset name:"), QLineEdit::Normal, 
                                       QString(), &ok);
    
    if (ok && !name.isEmpty()) {
        ScanConfiguration config = getCurrentConfiguration();
        
        // Save to settings
        QSettings settings;
        settings.beginGroup("ScanPresets");
        settings.beginGroup(name);
        
        // Save configuration
        settings.setValue("targetPaths", config.targetPaths);
        settings.setValue("detectionMode", static_cast<int>(config.detectionMode));
        settings.setValue("minimumFileSize", config.minimumFileSize);
        settings.setValue("maximumDepth", config.maximumDepth);
        settings.setValue("includeHidden", config.includeHidden);
        settings.setValue("includeSystem", config.includeSystem);
        settings.setValue("followSymlinks", config.followSymlinks);
        settings.setValue("scanArchives", config.scanArchives);
        settings.setValue("fileTypeFilter", static_cast<int>(config.fileTypeFilter));
        settings.setValue("excludePatterns", config.excludePatterns);
        settings.setValue("excludeFolders", config.excludeFolders);
        
        settings.endGroup();
        settings.endGroup();
        
        QMessageBox::information(this, tr("Preset Saved"), 
                               tr("Preset '%1' has been saved.").arg(name));
        
        emit presetSaved(name, config);
    }
}

void ScanSetupDialog::showUpgradeDialog()
{
    QMessageBox::information(this, tr("Upgrade to Premium"), 
                           tr("Premium features include:\n"
                              "• Unlimited file scanning\n"
                              "• Advanced duplicate detection\n"
                              "• Batch operations\n"
                              "• Priority support\n\n"
                              "Contact sales for more information."));
}

// Utility methods
void ScanSetupDialog::clearAllSelections()
{
    for (int i = 0; i < m_directoryTree->topLevelItemCount(); ++i) {
        m_directoryTree->topLevelItem(i)->setCheckState(0, Qt::Unchecked);
    }
}

void ScanSetupDialog::selectPath(const QString& path)
{
    if (path.isEmpty() || !QDir(path).exists()) {
        return;
    }
    
    // Find existing item
    for (int i = 0; i < m_directoryTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem* item = m_directoryTree->topLevelItem(i);
        if (item->data(0, Qt::UserRole).toString() == path) {
            item->setCheckState(0, Qt::Checked);
            return;
        }
    }
    
    // Add new item if not found
    QTreeWidgetItem* item = new QTreeWidgetItem(m_directoryTree);
    item->setText(0, QDir(path).dirName());
    item->setData(0, Qt::UserRole, path);
    item->setToolTip(0, path);
    item->setCheckState(0, Qt::Checked);
}

ScanSetupDialog::ScanConfiguration ScanSetupDialog::getCurrentConfiguration() const
{
    ScanConfiguration config = m_currentConfig;
    
    // Get selected paths
    config.targetPaths.clear();
    for (int i = 0; i < m_directoryTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem* item = m_directoryTree->topLevelItem(i);
        if (item->checkState(0) == Qt::Checked) {
            config.targetPaths << item->data(0, Qt::UserRole).toString();
        }
    }
    
    // Get detection mode
    config.detectionMode = static_cast<DetectionMode>(m_detectionMode->currentData().toInt());
    
    // Get size settings
    qDebug() << "getCurrentConfiguration: m_minimumSize->value() =" << m_minimumSize->value() << "MB";
    config.minimumFileSize = static_cast<qint64>(m_minimumSize->value()) * 1024 * 1024; // Convert MB to bytes
    qDebug() << "getCurrentConfiguration: config.minimumFileSize =" << config.minimumFileSize << "bytes";
    config.maximumDepth = m_maxDepth->currentData().toInt();
    
    // Get include options
    config.includeHidden = m_includeHidden->isChecked();
    config.includeSystem = m_includeSystem->isChecked();
    config.followSymlinks = m_followSymlinks->isChecked();
    config.scanArchives = m_scanArchives->isChecked();
    
    // Get file type filter
    if (m_allTypesCheck->isChecked()) {
        config.fileTypeFilter = FileTypeFilter::All;
    } else {
        int filter = 0;
        if (m_imagesCheck->isChecked()) filter |= static_cast<int>(FileTypeFilter::Images);
        if (m_documentsCheck->isChecked()) filter |= static_cast<int>(FileTypeFilter::Documents);
        if (m_videosCheck->isChecked()) filter |= static_cast<int>(FileTypeFilter::Videos);
        if (m_audioCheck->isChecked()) filter |= static_cast<int>(FileTypeFilter::Audio);
        if (m_archivesCheck->isChecked()) filter |= static_cast<int>(FileTypeFilter::Archives);
        config.fileTypeFilter = static_cast<FileTypeFilter>(filter);
    }
    
    // Get exclude patterns
    config.excludePatterns = m_excludePatterns->text().split(",", Qt::SkipEmptyParts);
    for (QString& pattern : config.excludePatterns) {
        pattern = pattern.trimmed();
    }
    
    // Get excluded folders
    config.excludeFolders.clear();
    for (int i = 0; i < m_excludeFoldersTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem* item = m_excludeFoldersTree->topLevelItem(i);
        config.excludeFolders << item->data(0, Qt::UserRole).toString();
    }
    
    return config;
}

void ScanSetupDialog::setConfiguration(const ScanConfiguration& config)
{
    m_currentConfig = config;
    
    // Clear and set paths
    m_directoryTree->clear();
    for (const QString& path : config.targetPaths) {
        selectPath(path);
    }
    
    // Set detection mode
    int modeIndex = m_detectionMode->findData(static_cast<int>(config.detectionMode));
    if (modeIndex >= 0) {
        m_detectionMode->setCurrentIndex(modeIndex);
    }
    
    // Set size settings
    m_minimumSize->setValue(static_cast<int>(config.minimumFileSize / (1024 * 1024))); // Convert bytes to MB
    
    int depthIndex = m_maxDepth->findData(config.maximumDepth);
    if (depthIndex >= 0) {
        m_maxDepth->setCurrentIndex(depthIndex);
    }
    
    // Set include options
    m_includeHidden->setChecked(config.includeHidden);
    m_includeSystem->setChecked(config.includeSystem);
    m_followSymlinks->setChecked(config.followSymlinks);
    m_scanArchives->setChecked(config.scanArchives);
    
    // Set file type filter
    if (config.fileTypeFilter == FileTypeFilter::All) {
        m_allTypesCheck->setChecked(true);
    } else {
        m_allTypesCheck->setChecked(false);
        int filter = static_cast<int>(config.fileTypeFilter);
        m_imagesCheck->setChecked(filter & static_cast<int>(FileTypeFilter::Images));
        m_documentsCheck->setChecked(filter & static_cast<int>(FileTypeFilter::Documents));
        m_videosCheck->setChecked(filter & static_cast<int>(FileTypeFilter::Videos));
        m_audioCheck->setChecked(filter & static_cast<int>(FileTypeFilter::Audio));
        m_archivesCheck->setChecked(filter & static_cast<int>(FileTypeFilter::Archives));
    }
    
    // Set exclude patterns
    m_excludePatterns->setText(config.excludePatterns.join(", "));
    
    // Set excluded folders
    m_excludeFoldersTree->clear();
    for (const QString& folder : config.excludeFolders) {
        if (!folder.isEmpty() && QDir(folder).exists()) {
            QTreeWidgetItem* item = new QTreeWidgetItem(m_excludeFoldersTree);
            item->setText(0, QDir(folder).dirName());
            item->setData(0, Qt::UserRole, folder);
            item->setToolTip(0, folder);
            item->setIcon(0, style()->standardIcon(QStyle::SP_DirIcon));
        }
    }
    
    updateEstimates();
}

QString ScanSetupDialog::formatFileSize(qint64 bytes) const
{
    if (bytes < 1024) {
        return tr("%1 B").arg(bytes);
    } else if (bytes < 1024 * 1024) {
        return tr("%1 KB").arg(QString::number(static_cast<double>(bytes) / 1024.0, 'f', 1));
    } else if (bytes < 1024 * 1024 * 1024) {
        return tr("%1 MB").arg(QString::number(static_cast<double>(bytes) / (1024.0 * 1024.0), 'f', 1));
    } else {
        return tr("%1 GB").arg(QString::number(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 'f', 2));
    }
}

void ScanSetupDialog::loadPreset(const QString& presetName)
{
    Logger::instance()->info(LogCategories::UI, QString("Loading preset: %1").arg(presetName));
    
    // Clear current selections
    clearAllSelections();
    
    if (presetName == "quick") {
        // Quick scan: Home, Downloads, Documents
        QStringList paths;
        paths << QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
        paths << QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
        paths << QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
        
        for (const QString& path : paths) {
            if (!path.isEmpty() && QDir(path).exists()) {
                selectPath(path);
            }
        }
        
        m_minimumSize->setValue(1); // 1 MB
        m_includeHidden->setChecked(false);
        m_followSymlinks->setChecked(true);
        m_allTypesCheck->setChecked(true);
        
    } else if (presetName == "downloads") {
        // Downloads cleanup
        QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
        if (!downloadsPath.isEmpty() && QDir(downloadsPath).exists()) {
            selectPath(downloadsPath);
        }
        
        qDebug() << "Downloads preset: Setting minimum size to 0";
        m_minimumSize->setValue(0); // All files
        qDebug() << "Downloads preset: Minimum size spinbox value is now:" << m_minimumSize->value();
        m_includeHidden->setChecked(false);
        m_allTypesCheck->setChecked(true);
        
    } else if (presetName == "photos") {
        // Photo cleanup
        QString picturesPath = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
        if (!picturesPath.isEmpty() && QDir(picturesPath).exists()) {
            selectPath(picturesPath);
        }
        
        m_minimumSize->setValue(0);
        m_includeHidden->setChecked(false);
        
        // Select only images
        m_allTypesCheck->setChecked(false);
        m_imagesCheck->setChecked(true);
        m_documentsCheck->setChecked(false);
        m_videosCheck->setChecked(false);
        m_audioCheck->setChecked(false);
        m_archivesCheck->setChecked(false);
        
    } else if (presetName == "documents") {
        // Documents scan
        QString documentsPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
        if (!documentsPath.isEmpty() && QDir(documentsPath).exists()) {
            selectPath(documentsPath);
        }
        
        m_minimumSize->setValue(0);
        m_includeHidden->setChecked(false);
        
        // Select only documents
        m_allTypesCheck->setChecked(false);
        m_imagesCheck->setChecked(false);
        m_documentsCheck->setChecked(true);
        m_videosCheck->setChecked(false);
        m_audioCheck->setChecked(false);
        m_archivesCheck->setChecked(false);
        
    } else if (presetName == "fullsystem") {
        // Full system scan
        QString homePath = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
        if (!homePath.isEmpty() && QDir(homePath).exists()) {
            selectPath(homePath);
        }
        
        m_minimumSize->setValue(1); // 1 MB
        m_includeHidden->setChecked(true);
        m_followSymlinks->setChecked(false);
        m_allTypesCheck->setChecked(true);
        
    } else if (presetName == "custom") {
        // Custom - just reset to defaults
        resetToDefaults();
    }
    
    // Update estimates
    updateEstimates();
    
    Logger::instance()->info(LogCategories::UI, QString("Preset '%1' loaded successfully").arg(presetName));
}

void ScanSetupDialog::resetToDefaults()
{
    // Reset to default configuration
    ScanConfiguration defaultConfig;
    defaultConfig.detectionMode = DetectionMode::Smart;
    defaultConfig.minimumFileSize = 0; // 0 MB - include all files
    defaultConfig.maximumDepth = -1;
    defaultConfig.includeHidden = false;
    defaultConfig.includeSystem = false;
    defaultConfig.followSymlinks = true;
    defaultConfig.scanArchives = false;
    defaultConfig.fileTypeFilter = FileTypeFilter::All;
    defaultConfig.excludePatterns << "*.tmp" << "*.log" << "Thumbs.db";
    defaultConfig.excludeFolders.clear(); // No folders excluded by default
    
    setConfiguration(defaultConfig);
}

void ScanSetupDialog::showEvent(QShowEvent* event)
{
    QDialog::showEvent(event);
    
    // Update estimates when dialog is shown
    updateEstimates();
    
    // Apply current theme
    applyTheme();
}
