#include "smart_selection_dialog.h"
#include <QApplication>
#include <QMessageBox>
#include <QInputDialog>
#include <QSettings>
#include <QFileInfo>
#include <QDir>
#include <QRegularExpression>
#include <QTimer>
#include <QDebug>
#include <QtAlgorithms>

// Static constants
const QStringList SmartSelectionDialog::DEFAULT_FILE_TYPES = {
    "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", // Images
    "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm",   // Videos
    "mp3", "wav", "flac", "aac", "ogg", "wma",          // Audio
    "pdf", "doc", "docx", "txt", "rtf", "odt",          // Documents
    "zip", "rar", "7z", "tar", "gz", "bz2"              // Archives
};

const QStringList SmartSelectionDialog::COMMON_LOCATION_PATTERNS = {
    "*/Downloads/*", "*/Desktop/*", "*/Documents/*", "*/Pictures/*",
    "*/Music/*", "*/Videos/*", "*/Temp/*", "*/tmp/*", "*/.cache/*"
};

SmartSelectionDialog::SmartSelectionDialog(QWidget* parent)
    : QDialog(parent)
    , m_mainLayout(nullptr)
    , m_updating(false)
{
    setWindowTitle(tr("Smart File Selection"));
    setMinimumSize(600, 500);
    resize(700, 600);
    setModal(true);
    
    setupUI();
    setupConnections();
    loadPresets();
    updateUIFromCriteria();
}

SmartSelectionDialog::~SmartSelectionDialog()
{
    savePresets();
}vo
id SmartSelectionDialog::setupUI()
{
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(20, 20, 20, 20);
    m_mainLayout->setSpacing(15);
    
    createModeSection();
    createCriteriaSection();
    createPreviewSection();
    createPresetSection();
    createButtonBox();
}

void SmartSelectionDialog::createModeSection()
{
    m_modeGroup = new QGroupBox(tr("Selection Mode"), this);
    m_modeLayout = new QVBoxLayout(m_modeGroup);
    m_modeLayout->setSpacing(10);
    
    m_modeCombo = new QComboBox(this);
    m_modeCombo->addItem(tr("Oldest Files"), static_cast<int>(OldestFiles));
    m_modeCombo->addItem(tr("Newest Files"), static_cast<int>(NewestFiles));
    m_modeCombo->addItem(tr("Largest Files"), static_cast<int>(LargestFiles));
    m_modeCombo->addItem(tr("Smallest Files"), static_cast<int>(SmallestFiles));
    m_modeCombo->addItem(tr("By Path Pattern"), static_cast<int>(ByPath));
    m_modeCombo->addItem(tr("By Multiple Criteria"), static_cast<int>(ByCriteria));
    m_modeCombo->addItem(tr("By File Type"), static_cast<int>(ByFileType));
    m_modeCombo->addItem(tr("By Location"), static_cast<int>(ByLocation));
    
    m_modeDescription = new QLabel(this);
    m_modeDescription->setWordWrap(true);
    m_modeDescription->setStyleSheet("color: #666; font-style: italic; padding: 5px;");
    
    m_modeLayout->addWidget(m_modeCombo);
    m_modeLayout->addWidget(m_modeDescription);
    
    m_mainLayout->addWidget(m_modeGroup);
}voi
d SmartSelectionDialog::createCriteriaSection()
{
    m_criteriaGroup = new QGroupBox(tr("Selection Criteria"), this);
    m_criteriaLayout = new QGridLayout(m_criteriaGroup);
    m_criteriaLayout->setSpacing(10);
    
    int row = 0;
    
    // Date range
    m_useDateRangeCheck = new QCheckBox(tr("Date Range:"), this);
    m_dateFromEdit = new QDateTimeEdit(QDateTime::currentDateTime().addDays(-30), this);
    m_dateFromEdit->setCalendarPopup(true);
    m_dateToEdit = new QDateTimeEdit(QDateTime::currentDateTime(), this);
    m_dateToEdit->setCalendarPopup(true);
    m_dateRangeLabel = new QLabel(tr("to"), this);
    
    m_criteriaLayout->addWidget(m_useDateRangeCheck, row, 0);
    m_criteriaLayout->addWidget(m_dateFromEdit, row, 1);
    m_criteriaLayout->addWidget(m_dateRangeLabel, row, 2);
    m_criteriaLayout->addWidget(m_dateToEdit, row, 3);
    row++;
    
    // Size range
    m_useSizeRangeCheck = new QCheckBox(tr("Size Range:"), this);
    m_minSizeSpin = new QSpinBox(this);
    m_minSizeSpin->setRange(0, 999999);
    m_minSizeSpin->setSuffix(" MB");
    m_maxSizeSpin = new QSpinBox(this);
    m_maxSizeSpin->setRange(1, 999999);
    m_maxSizeSpin->setValue(1000);
    m_maxSizeSpin->setSuffix(" MB");
    m_sizeRangeLabel = new QLabel(tr("to"), this);
    
    m_criteriaLayout->addWidget(m_useSizeRangeCheck, row, 0);
    m_criteriaLayout->addWidget(m_minSizeSpin, row, 1);
    m_criteriaLayout->addWidget(m_sizeRangeLabel, row, 2);
    m_criteriaLayout->addWidget(m_maxSizeSpin, row, 3);
    row++;
    
    // File types
    m_useFileTypesCheck = new QCheckBox(tr("File Types:"), this);
    m_fileTypesEdit = new QLineEdit(this);
    m_fileTypesEdit->setPlaceholderText(tr("jpg,png,pdf,txt (comma-separated)"));
    m_fileTypesLabel = new QLabel(tr("Extensions"), this);
    
    m_criteriaLayout->addWidget(m_useFileTypesCheck, row, 0);
    m_criteriaLayout->addWidget(m_fileTypesEdit, row, 1, 1, 2);
    m_criteriaLayout->addWidget(m_fileTypesLabel, row, 3);
    row++;
} 
   // Location patterns
    m_useLocationPatternsCheck = new QCheckBox(tr("Location Patterns:"), this);
    m_locationPatternsEdit = new QLineEdit(this);
    m_locationPatternsEdit->setPlaceholderText(tr("*/Downloads/*,*/Desktop/* (comma-separated)"));
    m_locationPatternsLabel = new QLabel(tr("Patterns"), this);
    
    m_criteriaLayout->addWidget(m_useLocationPatternsCheck, row, 0);
    m_criteriaLayout->addWidget(m_locationPatternsEdit, row, 1, 1, 2);
    m_criteriaLayout->addWidget(m_locationPatternsLabel, row, 3);
    row++;
    
    // Path pattern (for ByPath mode)
    m_pathPatternLabel = new QLabel(tr("Path Pattern:"), this);
    m_pathPatternEdit = new QLineEdit(this);
    m_pathPatternEdit->setPlaceholderText(tr("*/folder/* or *filename*"));
    
    m_criteriaLayout->addWidget(m_pathPatternLabel, row, 0);
    m_criteriaLayout->addWidget(m_pathPatternEdit, row, 1, 1, 3);
    row++;
    
    // Logic
    m_logicLabel = new QLabel(tr("Combine Criteria:"), this);
    m_andRadio = new QRadioButton(tr("AND (all must match)"), this);
    m_orRadio = new QRadioButton(tr("OR (any can match)"), this);
    m_logicGroup = new QButtonGroup(this);
    m_logicGroup->addButton(m_andRadio, 0);
    m_logicGroup->addButton(m_orRadio, 1);
    m_andRadio->setChecked(true);
    
    m_criteriaLayout->addWidget(m_logicLabel, row, 0);
    m_criteriaLayout->addWidget(m_andRadio, row, 1);
    m_criteriaLayout->addWidget(m_orRadio, row, 2, 1, 2);
    row++;
    
    // Limits
    m_maxFilesLabel = new QLabel(tr("Max Files:"), this);
    m_maxFilesSpin = new QSpinBox(this);
    m_maxFilesSpin->setRange(1, 10000);
    m_maxFilesSpin->setValue(100);
    
    m_percentageLabel = new QLabel(tr("Percentage:"), this);
    m_percentageSlider = new QSlider(Qt::Horizontal, this);
    m_percentageSlider->setRange(1, 100);
    m_percentageSlider->setValue(50);
    m_percentageValueLabel = new QLabel(tr("50%"), this);
    
    m_criteriaLayout->addWidget(m_maxFilesLabel, row, 0);
    m_criteriaLayout->addWidget(m_maxFilesSpin, row, 1);
    m_criteriaLayout->addWidget(m_percentageLabel, row, 2);
    m_criteriaLayout->addWidget(m_percentageSlider, row, 3);
    row++;
    
    m_criteriaLayout->addWidget(new QLabel(), row, 0);
    m_criteriaLayout->addWidget(new QLabel(), row, 1);
    m_criteriaLayout->addWidget(new QLabel(), row, 2);
    m_criteriaLayout->addWidget(m_percentageValueLabel, row, 3);
    
    m_mainLayout->addWidget(m_criteriaGroup);
}voi
d SmartSelectionDialog::createPreviewSection()
{
    m_previewGroup = new QGroupBox(tr("Selection Preview"), this);
    m_previewLayout = new QVBoxLayout(m_previewGroup);
    m_previewLayout->setSpacing(10);
    
    // Preview controls
    QHBoxLayout* previewControlsLayout = new QHBoxLayout();
    m_previewButton = new QPushButton(tr("Update Preview"), this);
    m_previewCountLabel = new QLabel(tr("No files selected"), this);
    m_previewCountLabel->setStyleSheet("font-weight: bold; color: #2c3e50;");
    
    previewControlsLayout->addWidget(m_previewButton);
    previewControlsLayout->addStretch();
    previewControlsLayout->addWidget(m_previewCountLabel);
    
    // Preview list
    m_previewList = new QTextEdit(this);
    m_previewList->setMaximumHeight(150);
    m_previewList->setReadOnly(true);
    m_previewList->setPlaceholderText(tr("Click 'Update Preview' to see selected files..."));
    
    // Preview progress
    m_previewProgress = new QProgressBar(this);
    m_previewProgress->setVisible(false);
    
    m_previewLayout->addLayout(previewControlsLayout);
    m_previewLayout->addWidget(m_previewList);
    m_previewLayout->addWidget(m_previewProgress);
    
    m_mainLayout->addWidget(m_previewGroup);
}

void SmartSelectionDialog::createPresetSection()
{
    m_presetGroup = new QGroupBox(tr("Selection Presets"), this);
    m_presetLayout = new QHBoxLayout(m_presetGroup);
    m_presetLayout->setSpacing(10);
    
    m_presetCombo = new QComboBox(this);
    m_presetCombo->setEditable(false);
    m_presetCombo->setMinimumWidth(200);
    
    m_savePresetButton = new QPushButton(tr("Save"), this);
    m_savePresetButton->setMaximumWidth(60);
    
    m_deletePresetButton = new QPushButton(tr("Delete"), this);
    m_deletePresetButton->setMaximumWidth(60);
    
    m_presetLayout->addWidget(new QLabel(tr("Preset:"), this));
    m_presetLayout->addWidget(m_presetCombo);
    m_presetLayout->addWidget(m_savePresetButton);
    m_presetLayout->addWidget(m_deletePresetButton);
    m_presetLayout->addStretch();
    
    m_mainLayout->addWidget(m_presetGroup);
}void 
SmartSelectionDialog::createButtonBox()
{
    m_buttonBox = new QDialogButtonBox(this);
    
    m_selectButton = new QPushButton(tr("Select Files"), this);
    m_selectButton->setDefault(true);
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    m_resetButton = new QPushButton(tr("Reset"), this);
    
    m_buttonBox->addButton(m_selectButton, QDialogButtonBox::AcceptRole);
    m_buttonBox->addButton(m_cancelButton, QDialogButtonBox::RejectRole);
    m_buttonBox->addButton(m_resetButton, QDialogButtonBox::ResetRole);
    
    m_mainLayout->addWidget(m_buttonBox);
}

void SmartSelectionDialog::setupConnections()
{
    // Mode combo
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SmartSelectionDialog::onModeComboChanged);
    
    // Criteria checkboxes
    connect(m_useDateRangeCheck, &QCheckBox::toggled,
            this, &SmartSelectionDialog::onDateRangeToggled);
    connect(m_useSizeRangeCheck, &QCheckBox::toggled,
            this, &SmartSelectionDialog::onSizeRangeToggled);
    connect(m_useFileTypesCheck, &QCheckBox::toggled,
            this, &SmartSelectionDialog::onFileTypesToggled);
    connect(m_useLocationPatternsCheck, &QCheckBox::toggled,
            this, &SmartSelectionDialog::onLocationPatternsToggled);
    
    // Criteria controls
    connect(m_dateFromEdit, &QDateTimeEdit::dateTimeChanged,
            this, &SmartSelectionDialog::onCriteriaChanged);
    connect(m_dateToEdit, &QDateTimeEdit::dateTimeChanged,
            this, &SmartSelectionDialog::onCriteriaChanged);
    connect(m_minSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SmartSelectionDialog::onCriteriaChanged);
    connect(m_maxSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SmartSelectionDialog::onCriteriaChanged);
    connect(m_fileTypesEdit, &QLineEdit::textChanged,
            this, &SmartSelectionDialog::onCriteriaChanged);
    connect(m_locationPatternsEdit, &QLineEdit::textChanged,
            this, &SmartSelectionDialog::onCriteriaChanged);
    connect(m_pathPatternEdit, &QLineEdit::textChanged,
            this, &SmartSelectionDialog::onCriteriaChanged);
    
    // Logic radio buttons
    connect(m_andRadio, &QRadioButton::toggled,
            this, &SmartSelectionDialog::onLogicChanged);
    connect(m_orRadio, &QRadioButton::toggled,
            this, &SmartSelectionDialog::onLogicChanged);
    
    // Limits
    connect(m_maxFilesSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SmartSelectionDialog::onMaxFilesChanged);
    connect(m_percentageSlider, &QSlider::valueChanged,
            this, &SmartSelectionDialog::onPercentageChanged);
    
    // Preview
    connect(m_previewButton, &QPushButton::clicked,
            this, &SmartSelectionDialog::onPreviewRequested);
    
    // Presets
    connect(m_presetCombo, &QComboBox::currentTextChanged,
            this, &SmartSelectionDialog::onPresetComboChanged);
    connect(m_savePresetButton, &QPushButton::clicked,
            this, &SmartSelectionDialog::onSavePresetClicked);
    connect(m_deletePresetButton, &QPushButton::clicked,
            this, &SmartSelectionDialog::onDeletePresetClicked);
    
    // Button box
    connect(m_selectButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(m_cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    connect(m_resetButton, &QPushButton::clicked,
            this, &SmartSelectionDialog::onResetToDefaults);
}