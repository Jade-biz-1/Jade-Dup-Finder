#include "advanced_filter_dialog.h"
#include <QMessageBox>
#include <QTime>
#include <QInputDialog>
#include <QSettings>

AdvancedFilterDialog::AdvancedFilterDialog(QWidget* parent)
    : QDialog(parent)
    , m_dateGroup(nullptr)
    , m_enableDateFilter(nullptr)
    , m_dateTypeCombo(nullptr)
    , m_dateFromEdit(nullptr)
    , m_dateToEdit(nullptr)
    , m_extensionGroup(nullptr)
    , m_enableExtensionFilter(nullptr)
    , m_extensionInput(nullptr)
    , m_addExtensionButton(nullptr)
    , m_extensionList(nullptr)
    , m_removeExtensionButton(nullptr)
    , m_pathGroup(nullptr)
    , m_enablePathFilter(nullptr)
    , m_pathPatternInput(nullptr)
    , m_addPathButton(nullptr)
    , m_pathPatternList(nullptr)
    , m_removePathButton(nullptr)
    , m_pathCaseSensitive(nullptr)
    , m_sizeGroup(nullptr)
    , m_enableSizeFilter(nullptr)
    , m_minSizeSpinBox(nullptr)
    , m_maxSizeSpinBox(nullptr)
    , m_sizeUnitCombo(nullptr)
    , m_combineGroup(nullptr)
    , m_combineModeCombo(nullptr)
    , m_applyButton(nullptr)
    , m_resetButton(nullptr)
    , m_closeButton(nullptr)
    , m_presetGroup(nullptr)
    , m_presetCombo(nullptr)
    , m_savePresetButton(nullptr)
    , m_loadPresetButton(nullptr)
    , m_deletePresetButton(nullptr)
{
    setupUI();
    connectSignals();
}

void AdvancedFilterDialog::setupUI() {
    setWindowTitle(tr("Advanced Filters"));
    setModal(true);
    setMinimumSize(600, 500);
    resize(700, 600);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    setupDateSection();
    mainLayout->addWidget(m_dateGroup);

    setupExtensionSection();
    mainLayout->addWidget(m_extensionGroup);

    setupPathSection();
    mainLayout->addWidget(m_pathGroup);

    setupSizeSection();
    mainLayout->addWidget(m_sizeGroup);

    setupCombineSection();
    mainLayout->addWidget(m_combineGroup);

    setupPresetSection();
    mainLayout->addWidget(m_presetGroup);

    mainLayout->addStretch();

    auto* buttonLayout = setupButtons();
    mainLayout->addLayout(buttonLayout);
}

void AdvancedFilterDialog::setupDateSection() {
    m_dateGroup = new QGroupBox(tr("Date Filter"), this);
    auto* layout = new QVBoxLayout(m_dateGroup);

    m_enableDateFilter = new QCheckBox(tr("Enable date filtering"), this);
    layout->addWidget(m_enableDateFilter);

    auto* dateLayout = new QGridLayout();
    
    // Date type
    dateLayout->addWidget(new QLabel(tr("Date type:"), this), 0, 0);
    m_dateTypeCombo = new QComboBox(this);
    m_dateTypeCombo->addItem(tr("Modified Date"), static_cast<int>(FilterCriteria::ModifiedDate));
    m_dateTypeCombo->addItem(tr("Created Date"), static_cast<int>(FilterCriteria::CreatedDate));
    dateLayout->addWidget(m_dateTypeCombo, 0, 1);

    // Date range
    dateLayout->addWidget(new QLabel(tr("From:"), this), 1, 0);
    m_dateFromEdit = new QDateEdit(QDate::currentDate().addDays(-30), this);
    m_dateFromEdit->setCalendarPopup(true);
    dateLayout->addWidget(m_dateFromEdit, 1, 1);

    dateLayout->addWidget(new QLabel(tr("To:"), this), 2, 0);
    m_dateToEdit = new QDateEdit(QDate::currentDate(), this);
    m_dateToEdit->setCalendarPopup(true);
    dateLayout->addWidget(m_dateToEdit, 2, 1);

    layout->addLayout(dateLayout);

    // Enable/disable controls based on checkbox
    connect(m_enableDateFilter, &QCheckBox::toggled, [this](bool enabled) {
        m_dateTypeCombo->setEnabled(enabled);
        m_dateFromEdit->setEnabled(enabled);
        m_dateToEdit->setEnabled(enabled);
    });
    
    // Initially disabled
    m_dateTypeCombo->setEnabled(false);
    m_dateFromEdit->setEnabled(false);
    m_dateToEdit->setEnabled(false);
}

void AdvancedFilterDialog::setupExtensionSection() {
    m_extensionGroup = new QGroupBox(tr("File Extension Filter"), this);
    auto* layout = new QVBoxLayout(m_extensionGroup);

    m_enableExtensionFilter = new QCheckBox(tr("Enable extension filtering"), this);
    layout->addWidget(m_enableExtensionFilter);

    auto* extLayout = new QHBoxLayout();
    
    m_extensionInput = new QLineEdit(this);
    m_extensionInput->setPlaceholderText(tr("Enter extension (e.g., jpg, txt, pdf)"));
    extLayout->addWidget(m_extensionInput);

    m_addExtensionButton = new QPushButton(tr("Add"), this);
    extLayout->addWidget(m_addExtensionButton);

    layout->addLayout(extLayout);

    m_extensionList = new QListWidget(this);
    m_extensionList->setMaximumHeight(100);
    layout->addWidget(m_extensionList);

    m_removeExtensionButton = new QPushButton(tr("Remove Selected"), this);
    layout->addWidget(m_removeExtensionButton);

    // Enable/disable controls
    connect(m_enableExtensionFilter, &QCheckBox::toggled, [this](bool enabled) {
        m_extensionInput->setEnabled(enabled);
        m_addExtensionButton->setEnabled(enabled);
        m_extensionList->setEnabled(enabled);
        m_removeExtensionButton->setEnabled(enabled);
    });
    
    // Initially disabled
    m_extensionInput->setEnabled(false);
    m_addExtensionButton->setEnabled(false);
    m_extensionList->setEnabled(false);
    m_removeExtensionButton->setEnabled(false);
}

void AdvancedFilterDialog::setupPathSection() {
    m_pathGroup = new QGroupBox(tr("Path Pattern Filter"), this);
    auto* layout = new QVBoxLayout(m_pathGroup);

    m_enablePathFilter = new QCheckBox(tr("Enable path filtering"), this);
    layout->addWidget(m_enablePathFilter);

    auto* pathLayout = new QHBoxLayout();
    
    m_pathPatternInput = new QLineEdit(this);
    m_pathPatternInput->setPlaceholderText(tr("Enter path pattern (e.g., */Documents/*, *temp*)"));
    pathLayout->addWidget(m_pathPatternInput);

    m_addPathButton = new QPushButton(tr("Add"), this);
    pathLayout->addWidget(m_addPathButton);

    layout->addLayout(pathLayout);

    m_pathPatternList = new QListWidget(this);
    m_pathPatternList->setMaximumHeight(100);
    layout->addWidget(m_pathPatternList);

    m_removePathButton = new QPushButton(tr("Remove Selected"), this);
    layout->addWidget(m_removePathButton);

    m_pathCaseSensitive = new QCheckBox(tr("Case sensitive"), this);
    layout->addWidget(m_pathCaseSensitive);

    // Enable/disable controls
    connect(m_enablePathFilter, &QCheckBox::toggled, [this](bool enabled) {
        m_pathPatternInput->setEnabled(enabled);
        m_addPathButton->setEnabled(enabled);
        m_pathPatternList->setEnabled(enabled);
        m_removePathButton->setEnabled(enabled);
        m_pathCaseSensitive->setEnabled(enabled);
    });
    
    // Initially disabled
    m_pathPatternInput->setEnabled(false);
    m_addPathButton->setEnabled(false);
    m_pathPatternList->setEnabled(false);
    m_removePathButton->setEnabled(false);
    m_pathCaseSensitive->setEnabled(false);
}

void AdvancedFilterDialog::setupSizeSection() {
    m_sizeGroup = new QGroupBox(tr("File Size Filter"), this);
    auto* layout = new QVBoxLayout(m_sizeGroup);

    m_enableSizeFilter = new QCheckBox(tr("Enable size filtering"), this);
    layout->addWidget(m_enableSizeFilter);

    auto* sizeLayout = new QGridLayout();
    
    sizeLayout->addWidget(new QLabel(tr("Min size:"), this), 0, 0);
    m_minSizeSpinBox = new QSpinBox(this);
    m_minSizeSpinBox->setRange(0, 999999);
    m_minSizeSpinBox->setValue(0);
    sizeLayout->addWidget(m_minSizeSpinBox, 0, 1);

    sizeLayout->addWidget(new QLabel(tr("Max size:"), this), 1, 0);
    m_maxSizeSpinBox = new QSpinBox(this);
    m_maxSizeSpinBox->setRange(0, 999999);
    m_maxSizeSpinBox->setValue(1000);
    sizeLayout->addWidget(m_maxSizeSpinBox, 1, 1);

    sizeLayout->addWidget(new QLabel(tr("Unit:"), this), 2, 0);
    m_sizeUnitCombo = new QComboBox(this);
    m_sizeUnitCombo->addItem(tr("Bytes"), static_cast<int>(FilterCriteria::Bytes));
    m_sizeUnitCombo->addItem(tr("KB"), static_cast<int>(FilterCriteria::KB));
    m_sizeUnitCombo->addItem(tr("MB"), static_cast<int>(FilterCriteria::MB));
    m_sizeUnitCombo->addItem(tr("GB"), static_cast<int>(FilterCriteria::GB));
    m_sizeUnitCombo->setCurrentIndex(2); // Default to MB
    sizeLayout->addWidget(m_sizeUnitCombo, 2, 1);

    layout->addLayout(sizeLayout);

    // Enable/disable controls
    connect(m_enableSizeFilter, &QCheckBox::toggled, [this](bool enabled) {
        m_minSizeSpinBox->setEnabled(enabled);
        m_maxSizeSpinBox->setEnabled(enabled);
        m_sizeUnitCombo->setEnabled(enabled);
    });
    
    // Initially disabled
    m_minSizeSpinBox->setEnabled(false);
    m_maxSizeSpinBox->setEnabled(false);
    m_sizeUnitCombo->setEnabled(false);
}

void AdvancedFilterDialog::setupCombineSection() {
    m_combineGroup = new QGroupBox(tr("Combine Filters"), this);
    auto* layout = new QHBoxLayout(m_combineGroup);

    layout->addWidget(new QLabel(tr("Combine mode:"), this));
    m_combineModeCombo = new QComboBox(this);
    m_combineModeCombo->addItem(tr("AND (all conditions must match)"), static_cast<int>(FilterCriteria::AND));
    m_combineModeCombo->addItem(tr("OR (any condition can match)"), static_cast<int>(FilterCriteria::OR));
    layout->addWidget(m_combineModeCombo);
    layout->addStretch();
}

void AdvancedFilterDialog::setupPresetSection() {
    m_presetGroup = new QGroupBox(tr("Filter Presets"), this);
    auto* layout = new QHBoxLayout(m_presetGroup);

    layout->addWidget(new QLabel(tr("Preset:"), this));
    
    m_presetCombo = new QComboBox(this);
    m_presetCombo->setMinimumWidth(200);
    layout->addWidget(m_presetCombo);

    m_savePresetButton = new QPushButton(tr("Save"), this);
    layout->addWidget(m_savePresetButton);

    m_loadPresetButton = new QPushButton(tr("Load"), this);
    layout->addWidget(m_loadPresetButton);

    m_deletePresetButton = new QPushButton(tr("Delete"), this);
    layout->addWidget(m_deletePresetButton);

    layout->addStretch();
    
    loadPresetList();
}

QHBoxLayout* AdvancedFilterDialog::setupButtons() {
    auto* buttonLayout = new QHBoxLayout();
    
    m_applyButton = new QPushButton(tr("Apply Filters"), this);
    m_applyButton->setDefault(true);
    buttonLayout->addWidget(m_applyButton);

    m_resetButton = new QPushButton(tr("Reset"), this);
    buttonLayout->addWidget(m_resetButton);

    buttonLayout->addStretch();

    m_closeButton = new QPushButton(tr("Close"), this);
    buttonLayout->addWidget(m_closeButton);

    return buttonLayout;
}

void AdvancedFilterDialog::connectSignals() {
    connect(m_addExtensionButton, &QPushButton::clicked, this, &AdvancedFilterDialog::onAddExtensionClicked);
    connect(m_removeExtensionButton, &QPushButton::clicked, this, &AdvancedFilterDialog::onRemoveExtensionClicked);
    connect(m_addPathButton, &QPushButton::clicked, this, &AdvancedFilterDialog::onAddPathPatternClicked);
    connect(m_removePathButton, &QPushButton::clicked, this, &AdvancedFilterDialog::onRemovePathPatternClicked);
    connect(m_applyButton, &QPushButton::clicked, this, &AdvancedFilterDialog::onApplyClicked);
    connect(m_resetButton, &QPushButton::clicked, this, &AdvancedFilterDialog::onResetClicked);
    connect(m_closeButton, &QPushButton::clicked, this, &QDialog::accept);
    
    // Preset connections (Task 12)
    connect(m_savePresetButton, &QPushButton::clicked, this, &AdvancedFilterDialog::onSavePresetClicked);
    connect(m_loadPresetButton, &QPushButton::clicked, this, &AdvancedFilterDialog::onLoadPresetClicked);
    connect(m_deletePresetButton, &QPushButton::clicked, this, &AdvancedFilterDialog::onDeletePresetClicked);
    connect(m_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &AdvancedFilterDialog::onPresetSelectionChanged);
    
    // Allow Enter key to add extensions/patterns
    connect(m_extensionInput, &QLineEdit::returnPressed, this, &AdvancedFilterDialog::onAddExtensionClicked);
    connect(m_pathPatternInput, &QLineEdit::returnPressed, this, &AdvancedFilterDialog::onAddPathPatternClicked);
}

void AdvancedFilterDialog::onAddExtensionClicked() {
    QString ext = m_extensionInput->text().trimmed();
    if (ext.isEmpty()) {
        return;
    }
    
    // Remove leading dot if present
    if (ext.startsWith('.')) {
        ext = ext.mid(1);
    }
    
    // Check if already exists
    for (int i = 0; i < m_extensionList->count(); ++i) {
        if (m_extensionList->item(i)->text() == ext) {
            QMessageBox::information(this, tr("Duplicate"), tr("Extension '%1' is already in the list.").arg(ext));
            return;
        }
    }
    
    m_extensionList->addItem(ext);
    m_extensionInput->clear();
}

void AdvancedFilterDialog::onRemoveExtensionClicked() {
    int currentRow = m_extensionList->currentRow();
    if (currentRow >= 0) {
        delete m_extensionList->takeItem(currentRow);
    }
}

void AdvancedFilterDialog::onAddPathPatternClicked() {
    QString pattern = m_pathPatternInput->text().trimmed();
    if (pattern.isEmpty()) {
        return;
    }
    
    // Check if already exists
    for (int i = 0; i < m_pathPatternList->count(); ++i) {
        if (m_pathPatternList->item(i)->text() == pattern) {
            QMessageBox::information(this, tr("Duplicate"), tr("Pattern '%1' is already in the list.").arg(pattern));
            return;
        }
    }
    
    m_pathPatternList->addItem(pattern);
    m_pathPatternInput->clear();
}

void AdvancedFilterDialog::onRemovePathPatternClicked() {
    int currentRow = m_pathPatternList->currentRow();
    if (currentRow >= 0) {
        delete m_pathPatternList->takeItem(currentRow);
    }
}

void AdvancedFilterDialog::onApplyClicked() {
    FilterCriteria criteria = getFilterCriteria();
    emit filtersChanged(criteria);
    accept();
}

void AdvancedFilterDialog::onResetClicked() {
    resetFilters();
}

AdvancedFilterDialog::FilterCriteria AdvancedFilterDialog::getFilterCriteria() const {
    FilterCriteria criteria;
    
    // Date filter
    criteria.enableDateFilter = m_enableDateFilter->isChecked();
    if (criteria.enableDateFilter) {
        criteria.dateType = static_cast<FilterCriteria::DateType>(m_dateTypeCombo->currentData().toInt());
        criteria.dateFrom = QDateTime(m_dateFromEdit->date(), QTime(0, 0, 0));
        criteria.dateTo = QDateTime(m_dateToEdit->date(), QTime(23, 59, 59)); // End of day
    }
    
    // Extension filter
    criteria.enableExtensionFilter = m_enableExtensionFilter->isChecked();
    if (criteria.enableExtensionFilter) {
        for (int i = 0; i < m_extensionList->count(); ++i) {
            criteria.includedExtensions << m_extensionList->item(i)->text();
        }
    }
    
    // Path filter
    criteria.enablePathFilter = m_enablePathFilter->isChecked();
    if (criteria.enablePathFilter) {
        for (int i = 0; i < m_pathPatternList->count(); ++i) {
            criteria.pathPatterns << m_pathPatternList->item(i)->text();
        }
        criteria.pathCaseSensitive = m_pathCaseSensitive->isChecked();
    }
    
    // Size filter
    criteria.enableSizeFilter = m_enableSizeFilter->isChecked();
    if (criteria.enableSizeFilter) {
        criteria.sizeUnit = static_cast<FilterCriteria::SizeUnit>(m_sizeUnitCombo->currentData().toInt());
        criteria.minSize = convertSizeToBytes(m_minSizeSpinBox->value(), criteria.sizeUnit);
        criteria.maxSize = convertSizeToBytes(m_maxSizeSpinBox->value(), criteria.sizeUnit);
    }
    
    // Combine mode
    criteria.combineMode = static_cast<FilterCriteria::CombineMode>(m_combineModeCombo->currentData().toInt());
    
    return criteria;
}

void AdvancedFilterDialog::setFilterCriteria(const FilterCriteria& criteria) {
    // Date filter
    m_enableDateFilter->setChecked(criteria.enableDateFilter);
    m_dateTypeCombo->setCurrentIndex(static_cast<int>(criteria.dateType));
    m_dateFromEdit->setDate(criteria.dateFrom.date());
    m_dateToEdit->setDate(criteria.dateTo.date());
    
    // Extension filter
    m_enableExtensionFilter->setChecked(criteria.enableExtensionFilter);
    m_extensionList->clear();
    for (const QString& ext : criteria.includedExtensions) {
        m_extensionList->addItem(ext);
    }
    
    // Path filter
    m_enablePathFilter->setChecked(criteria.enablePathFilter);
    m_pathPatternList->clear();
    for (const QString& pattern : criteria.pathPatterns) {
        m_pathPatternList->addItem(pattern);
    }
    m_pathCaseSensitive->setChecked(criteria.pathCaseSensitive);
    
    // Size filter
    m_enableSizeFilter->setChecked(criteria.enableSizeFilter);
    m_sizeUnitCombo->setCurrentIndex(static_cast<int>(criteria.sizeUnit));
    m_minSizeSpinBox->setValue(convertSizeFromBytes(criteria.minSize, criteria.sizeUnit));
    m_maxSizeSpinBox->setValue(convertSizeFromBytes(criteria.maxSize, criteria.sizeUnit));
    
    // Combine mode
    m_combineModeCombo->setCurrentIndex(static_cast<int>(criteria.combineMode));
}

void AdvancedFilterDialog::resetFilters() {
    m_enableDateFilter->setChecked(false);
    m_enableExtensionFilter->setChecked(false);
    m_enablePathFilter->setChecked(false);
    m_enableSizeFilter->setChecked(false);
    
    m_extensionList->clear();
    m_pathPatternList->clear();
    
    m_dateFromEdit->setDate(QDate::currentDate().addDays(-30));
    m_dateToEdit->setDate(QDate::currentDate());
    
    m_minSizeSpinBox->setValue(0);
    m_maxSizeSpinBox->setValue(1000);
    m_sizeUnitCombo->setCurrentIndex(2); // MB
    
    m_pathCaseSensitive->setChecked(false);
    m_combineModeCombo->setCurrentIndex(0); // AND
}

qint64 AdvancedFilterDialog::convertSizeToBytes(int value, FilterCriteria::SizeUnit unit) const {
    qint64 bytes = value;
    switch (unit) {
        case FilterCriteria::Bytes:
            break;
        case FilterCriteria::KB:
            bytes *= 1024;
            break;
        case FilterCriteria::MB:
            bytes *= 1024 * 1024;
            break;
        case FilterCriteria::GB:
            bytes *= 1024LL * 1024 * 1024;
            break;
    }
    return bytes;
}

int AdvancedFilterDialog::convertSizeFromBytes(qint64 bytes, FilterCriteria::SizeUnit unit) const {
    switch (unit) {
        case FilterCriteria::Bytes:
            return static_cast<int>(bytes);
        case FilterCriteria::KB:
            return static_cast<int>(bytes / 1024);
        case FilterCriteria::MB:
            return static_cast<int>(bytes / (1024 * 1024));
        case FilterCriteria::GB:
            return static_cast<int>(bytes / (1024LL * 1024 * 1024));
    }
    return 0;
}

// Preset management methods (Task 12)

void AdvancedFilterDialog::savePreset(const QString& name) {
    if (name.isEmpty()) {
        return;
    }
    
    FilterCriteria criteria = getFilterCriteria();
    QSettings settings;
    
    settings.beginGroup("FilterPresets");
    settings.beginGroup(name);
    
    // Save date filter
    settings.setValue("enableDateFilter", criteria.enableDateFilter);
    settings.setValue("dateType", static_cast<int>(criteria.dateType));
    settings.setValue("dateFrom", criteria.dateFrom);
    settings.setValue("dateTo", criteria.dateTo);
    
    // Save extension filter
    settings.setValue("enableExtensionFilter", criteria.enableExtensionFilter);
    settings.setValue("includedExtensions", criteria.includedExtensions);
    
    // Save path filter
    settings.setValue("enablePathFilter", criteria.enablePathFilter);
    settings.setValue("pathPatterns", criteria.pathPatterns);
    settings.setValue("pathCaseSensitive", criteria.pathCaseSensitive);
    
    // Save size filter
    settings.setValue("enableSizeFilter", criteria.enableSizeFilter);
    settings.setValue("minSize", static_cast<qint64>(criteria.minSize));
    settings.setValue("maxSize", static_cast<qint64>(criteria.maxSize));
    settings.setValue("sizeUnit", static_cast<int>(criteria.sizeUnit));
    
    // Save combine mode
    settings.setValue("combineMode", static_cast<int>(criteria.combineMode));
    
    settings.endGroup();
    settings.endGroup();
    
    loadPresetList();
}

bool AdvancedFilterDialog::loadPreset(const QString& name) {
    if (name.isEmpty()) {
        return false;
    }
    
    QSettings settings;
    settings.beginGroup("FilterPresets");
    
    if (!settings.childGroups().contains(name)) {
        return false;
    }
    
    settings.beginGroup(name);
    
    FilterCriteria criteria;
    
    // Load date filter
    criteria.enableDateFilter = settings.value("enableDateFilter", false).toBool();
    criteria.dateType = static_cast<FilterCriteria::DateType>(settings.value("dateType", 0).toInt());
    criteria.dateFrom = settings.value("dateFrom", QDateTime::currentDateTime().addDays(-30)).toDateTime();
    criteria.dateTo = settings.value("dateTo", QDateTime::currentDateTime()).toDateTime();
    
    // Load extension filter
    criteria.enableExtensionFilter = settings.value("enableExtensionFilter", false).toBool();
    criteria.includedExtensions = settings.value("includedExtensions", QStringList()).toStringList();
    
    // Load path filter
    criteria.enablePathFilter = settings.value("enablePathFilter", false).toBool();
    criteria.pathPatterns = settings.value("pathPatterns", QStringList()).toStringList();
    criteria.pathCaseSensitive = settings.value("pathCaseSensitive", false).toBool();
    
    // Load size filter
    criteria.enableSizeFilter = settings.value("enableSizeFilter", false).toBool();
    criteria.minSize = settings.value("minSize", 0).toLongLong();
    criteria.maxSize = settings.value("maxSize", 1024*1024).toLongLong();
    criteria.sizeUnit = static_cast<FilterCriteria::SizeUnit>(settings.value("sizeUnit", 2).toInt());
    
    // Load combine mode
    criteria.combineMode = static_cast<FilterCriteria::CombineMode>(settings.value("combineMode", 0).toInt());
    
    settings.endGroup();
    settings.endGroup();
    
    setFilterCriteria(criteria);
    return true;
}

QStringList AdvancedFilterDialog::getAvailablePresets() const {
    QSettings settings;
    settings.beginGroup("FilterPresets");
    return settings.childGroups();
}

bool AdvancedFilterDialog::deletePreset(const QString& name) {
    if (name.isEmpty()) {
        return false;
    }
    
    QSettings settings;
    settings.beginGroup("FilterPresets");
    
    if (!settings.childGroups().contains(name)) {
        return false;
    }
    
    settings.remove(name);
    settings.endGroup();
    
    loadPresetList();
    return true;
}

void AdvancedFilterDialog::loadPresetList() {
    if (!m_presetCombo) {
        return;
    }
    
    m_presetCombo->clear();
    m_presetCombo->addItem(tr("-- Select Preset --"), QString());
    
    QStringList presets = getAvailablePresets();
    for (const QString& preset : presets) {
        m_presetCombo->addItem(preset, preset);
    }
    
    // Update button states
    bool hasPresets = !presets.isEmpty();
    m_loadPresetButton->setEnabled(hasPresets && m_presetCombo->currentIndex() > 0);
    m_deletePresetButton->setEnabled(hasPresets && m_presetCombo->currentIndex() > 0);
}

void AdvancedFilterDialog::onSavePresetClicked() {
    bool ok;
    QString name = QInputDialog::getText(this, tr("Save Preset"), 
                                        tr("Enter preset name:"), 
                                        QLineEdit::Normal, QString(), &ok);
    
    if (ok && !name.isEmpty()) {
        // Check if preset already exists
        if (getAvailablePresets().contains(name)) {
            int ret = QMessageBox::question(this, tr("Preset Exists"), 
                                          tr("A preset with this name already exists. Overwrite?"),
                                          QMessageBox::Yes | QMessageBox::No);
            if (ret != QMessageBox::Yes) {
                return;
            }
        }
        
        savePreset(name);
        
        // Select the newly saved preset
        int index = m_presetCombo->findData(name);
        if (index >= 0) {
            m_presetCombo->setCurrentIndex(index);
        }
    }
}

void AdvancedFilterDialog::onLoadPresetClicked() {
    QString presetName = m_presetCombo->currentData().toString();
    if (!presetName.isEmpty()) {
        if (!loadPreset(presetName)) {
            QMessageBox::warning(this, tr("Load Failed"), 
                                tr("Failed to load preset '%1'.").arg(presetName));
        }
    }
}

void AdvancedFilterDialog::onDeletePresetClicked() {
    QString presetName = m_presetCombo->currentData().toString();
    if (!presetName.isEmpty()) {
        int ret = QMessageBox::question(this, tr("Delete Preset"), 
                                      tr("Are you sure you want to delete preset '%1'?").arg(presetName),
                                      QMessageBox::Yes | QMessageBox::No);
        
        if (ret == QMessageBox::Yes) {
            if (!deletePreset(presetName)) {
                QMessageBox::warning(this, tr("Delete Failed"), 
                                    tr("Failed to delete preset '%1'.").arg(presetName));
            }
        }
    }
}

void AdvancedFilterDialog::onPresetSelectionChanged() {
    bool hasSelection = m_presetCombo->currentIndex() > 0;
    m_loadPresetButton->setEnabled(hasSelection);
    m_deletePresetButton->setEnabled(hasSelection);
}