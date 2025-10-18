#include "grouping_options_dialog.h"
#include "theme_manager.h"
#include <QApplication>

GroupingOptionsDialog::GroupingOptionsDialog(QWidget* parent)
    : QDialog(parent)
    , m_primaryGroup(nullptr)
    , m_primaryLayout(nullptr)
    , m_primaryButtonGroup(new QButtonGroup(this))
    , m_hashRadio(nullptr)
    , m_sizeRadio(nullptr)
    , m_typeRadio(nullptr)
    , m_creationDateRadio(nullptr)
    , m_modificationDateRadio(nullptr)
    , m_locationRadio(nullptr)
    , m_secondaryGroup(nullptr)
    , m_secondaryLayout(nullptr)
    , m_useSecondaryCheckBox(nullptr)
    , m_secondaryCombo(nullptr)
    , m_optionsGroup(nullptr)
    , m_optionsLayout(nullptr)
    , m_dateGroupingCombo(nullptr)
    , m_caseSensitiveCheckBox(nullptr)
    , m_groupByParentDirCheckBox(nullptr)
    , m_previewLabel(nullptr)
    , m_applyButton(nullptr)
    , m_resetButton(nullptr)
    , m_cancelButton(nullptr)
{
    setupUI();
    setupConnections();
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(this);
    
    // Set default options
    GroupingOptions defaultOptions;
    setGroupingOptions(defaultOptions);
}

void GroupingOptionsDialog::setupUI() {
    setWindowTitle(tr("Grouping Options"));
    setModal(true);
    setMinimumSize(450, 500);
    resize(500, 550);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    // Primary grouping criteria
    m_primaryGroup = new QGroupBox(tr("Primary Grouping Criteria"), this);
    m_primaryLayout = new QVBoxLayout(m_primaryGroup);
    
    m_hashRadio = new QRadioButton(tr("By Hash (identical content)"), this);
    m_hashRadio->setToolTip(tr("Group files with identical content (recommended)"));
    m_primaryButtonGroup->addButton(m_hashRadio, static_cast<int>(GroupingCriteria::Hash));
    m_primaryLayout->addWidget(m_hashRadio);
    
    m_sizeRadio = new QRadioButton(tr("By Size"), this);
    m_sizeRadio->setToolTip(tr("Group files with the same size"));
    m_primaryButtonGroup->addButton(m_sizeRadio, static_cast<int>(GroupingCriteria::Size));
    m_primaryLayout->addWidget(m_sizeRadio);
    
    m_typeRadio = new QRadioButton(tr("By Type/Extension"), this);
    m_typeRadio->setToolTip(tr("Group files with the same file extension"));
    m_primaryButtonGroup->addButton(m_typeRadio, static_cast<int>(GroupingCriteria::Type));
    m_primaryLayout->addWidget(m_typeRadio);
    
    m_creationDateRadio = new QRadioButton(tr("By Creation Date"), this);
    m_creationDateRadio->setToolTip(tr("Group files created on the same date"));
    m_primaryButtonGroup->addButton(m_creationDateRadio, static_cast<int>(GroupingCriteria::CreationDate));
    m_primaryLayout->addWidget(m_creationDateRadio);
    
    m_modificationDateRadio = new QRadioButton(tr("By Modification Date"), this);
    m_modificationDateRadio->setToolTip(tr("Group files modified on the same date"));
    m_primaryButtonGroup->addButton(m_modificationDateRadio, static_cast<int>(GroupingCriteria::ModificationDate));
    m_primaryLayout->addWidget(m_modificationDateRadio);
    
    m_locationRadio = new QRadioButton(tr("By Location/Directory"), this);
    m_locationRadio->setToolTip(tr("Group files from the same directory"));
    m_primaryButtonGroup->addButton(m_locationRadio, static_cast<int>(GroupingCriteria::Location));
    m_primaryLayout->addWidget(m_locationRadio);
    
    mainLayout->addWidget(m_primaryGroup);

    // Secondary grouping criteria
    m_secondaryGroup = new QGroupBox(tr("Secondary Grouping"), this);
    m_secondaryLayout = new QVBoxLayout(m_secondaryGroup);
    
    m_useSecondaryCheckBox = new QCheckBox(tr("Use secondary grouping criteria"), this);
    m_useSecondaryCheckBox->setToolTip(tr("Apply additional grouping within primary groups"));
    m_secondaryLayout->addWidget(m_useSecondaryCheckBox);
    
    auto* secondaryComboLayout = new QHBoxLayout();
    secondaryComboLayout->addWidget(new QLabel(tr("Secondary criteria:"), this));
    m_secondaryCombo = new QComboBox(this);
    m_secondaryCombo->setEnabled(false);
    secondaryComboLayout->addWidget(m_secondaryCombo);
    secondaryComboLayout->addStretch();
    m_secondaryLayout->addLayout(secondaryComboLayout);
    
    mainLayout->addWidget(m_secondaryGroup);

    // Additional options
    m_optionsGroup = new QGroupBox(tr("Additional Options"), this);
    m_optionsLayout = new QGridLayout(m_optionsGroup);
    
    // Date grouping precision
    m_optionsLayout->addWidget(new QLabel(tr("Date grouping:"), this), 0, 0);
    m_dateGroupingCombo = new QComboBox(this);
    m_dateGroupingCombo->addItem(tr("Exact date and time"), static_cast<int>(DateGrouping::ExactDate));
    m_dateGroupingCombo->addItem(tr("Same day"), static_cast<int>(DateGrouping::SameDay));
    m_dateGroupingCombo->addItem(tr("Same week"), static_cast<int>(DateGrouping::SameWeek));
    m_dateGroupingCombo->addItem(tr("Same month"), static_cast<int>(DateGrouping::SameMonth));
    m_dateGroupingCombo->addItem(tr("Same year"), static_cast<int>(DateGrouping::SameYear));
    m_dateGroupingCombo->setCurrentIndex(1); // Default to "Same day"
    m_optionsLayout->addWidget(m_dateGroupingCombo, 0, 1);
    
    // Case sensitive file types
    m_caseSensitiveCheckBox = new QCheckBox(tr("Case-sensitive file types"), this);
    m_caseSensitiveCheckBox->setToolTip(tr("Treat .JPG and .jpg as different types"));
    m_optionsLayout->addWidget(m_caseSensitiveCheckBox, 1, 0, 1, 2);
    
    // Group by parent directory
    m_groupByParentDirCheckBox = new QCheckBox(tr("Group by parent directory"), this);
    m_groupByParentDirCheckBox->setToolTip(tr("Consider parent directory when grouping by location"));
    m_optionsLayout->addWidget(m_groupByParentDirCheckBox, 2, 0, 1, 2);
    
    mainLayout->addWidget(m_optionsGroup);

    // Preview
    auto* previewGroup = new QGroupBox(tr("Preview"), this);
    auto* previewLayout = new QVBoxLayout(previewGroup);
    m_previewLabel = new QLabel(this);
    m_previewLabel->setWordWrap(true);
    // Theme-aware styling applied by ThemeManager
    m_previewLabel->setStyleSheet("padding: 8px; border-radius: 4px;");
    previewLayout->addWidget(m_previewLabel);
    mainLayout->addWidget(previewGroup);

    mainLayout->addStretch();

    // Buttons
    auto* buttonLayout = new QHBoxLayout();
    
    m_resetButton = new QPushButton(tr("Reset to Default"), this);
    buttonLayout->addWidget(m_resetButton);
    
    buttonLayout->addStretch();
    
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    buttonLayout->addWidget(m_cancelButton);
    
    m_applyButton = new QPushButton(tr("Apply"), this);
    m_applyButton->setDefault(true);
    buttonLayout->addWidget(m_applyButton);
    
    mainLayout->addLayout(buttonLayout);
}

void GroupingOptionsDialog::setupConnections() {
    // Primary criteria changes
    connect(m_primaryButtonGroup, QOverload<QAbstractButton*>::of(&QButtonGroup::buttonClicked),
            this, [this](QAbstractButton*) { onPrimaryCriteriaChanged(); });
    
    // Secondary criteria
    connect(m_useSecondaryCheckBox, &QCheckBox::toggled,
            this, &GroupingOptionsDialog::onSecondaryCriteriaToggled);
    connect(m_secondaryCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GroupingOptionsDialog::updatePreview);
    
    // Options
    connect(m_dateGroupingCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GroupingOptionsDialog::updatePreview);
    connect(m_caseSensitiveCheckBox, &QCheckBox::toggled,
            this, &GroupingOptionsDialog::updatePreview);
    connect(m_groupByParentDirCheckBox, &QCheckBox::toggled,
            this, &GroupingOptionsDialog::updatePreview);
    
    // Buttons
    connect(m_applyButton, &QPushButton::clicked, this, &GroupingOptionsDialog::onApplyClicked);
    connect(m_resetButton, &QPushButton::clicked, this, &GroupingOptionsDialog::onResetClicked);
    connect(m_cancelButton, &QPushButton::clicked, this, &QDialog::reject);
}

void GroupingOptionsDialog::populateSecondaryCombo() {
    if (!m_secondaryCombo) return;
    
    m_secondaryCombo->clear();
    
    GroupingCriteria primary = getSelectedPrimaryCriteria();
    
    // Add all criteria except the primary one
    if (primary != GroupingCriteria::Hash) {
        m_secondaryCombo->addItem(GroupingOptionsDialog::formatCriteriaName(GroupingCriteria::Hash), 
                                 static_cast<int>(GroupingCriteria::Hash));
    }
    if (primary != GroupingCriteria::Size) {
        m_secondaryCombo->addItem(GroupingOptionsDialog::formatCriteriaName(GroupingCriteria::Size), 
                                 static_cast<int>(GroupingCriteria::Size));
    }
    if (primary != GroupingCriteria::Type) {
        m_secondaryCombo->addItem(GroupingOptionsDialog::formatCriteriaName(GroupingCriteria::Type), 
                                 static_cast<int>(GroupingCriteria::Type));
    }
    if (primary != GroupingCriteria::CreationDate) {
        m_secondaryCombo->addItem(GroupingOptionsDialog::formatCriteriaName(GroupingCriteria::CreationDate), 
                                 static_cast<int>(GroupingCriteria::CreationDate));
    }
    if (primary != GroupingCriteria::ModificationDate) {
        m_secondaryCombo->addItem(GroupingOptionsDialog::formatCriteriaName(GroupingCriteria::ModificationDate), 
                                 static_cast<int>(GroupingCriteria::ModificationDate));
    }
    if (primary != GroupingCriteria::Location) {
        m_secondaryCombo->addItem(GroupingOptionsDialog::formatCriteriaName(GroupingCriteria::Location), 
                                 static_cast<int>(GroupingCriteria::Location));
    }
}

void GroupingOptionsDialog::onPrimaryCriteriaChanged() {
    populateSecondaryCombo();
    updateSecondaryOptions();
    updatePreview();
}

void GroupingOptionsDialog::onSecondaryCriteriaToggled(bool enabled) {
    m_secondaryCombo->setEnabled(enabled);
    updatePreview();
}

void GroupingOptionsDialog::updateSecondaryOptions() {
    GroupingCriteria primary = getSelectedPrimaryCriteria();
    
    // Enable/disable options based on primary criteria
    bool isDateCriteria = (primary == GroupingCriteria::CreationDate || 
                          primary == GroupingCriteria::ModificationDate);
    m_dateGroupingCombo->setEnabled(isDateCriteria);
    
    bool isTypeCriteria = (primary == GroupingCriteria::Type);
    m_caseSensitiveCheckBox->setEnabled(isTypeCriteria);
    
    bool isLocationCriteria = (primary == GroupingCriteria::Location);
    m_groupByParentDirCheckBox->setEnabled(isLocationCriteria);
}

void GroupingOptionsDialog::onApplyClicked() {
    GroupingOptions options = getGroupingOptions();
    emit groupingChanged(options);
    accept();
}

void GroupingOptionsDialog::onResetClicked() {
    GroupingOptions defaultOptions;
    setGroupingOptions(defaultOptions);
}

void GroupingOptionsDialog::updatePreview() {
    if (!m_previewLabel) return;
    
    GroupingOptions options = getGroupingOptions();
    QString description = getGroupingDescription(options);
    m_previewLabel->setText(description);
}

GroupingOptionsDialog::GroupingOptions GroupingOptionsDialog::getGroupingOptions() const {
    GroupingOptions options;
    
    options.primaryCriteria = getSelectedPrimaryCriteria();
    options.useSecondaryCriteria = m_useSecondaryCheckBox->isChecked();
    
    if (options.useSecondaryCriteria && m_secondaryCombo->currentIndex() >= 0) {
        options.secondaryCriteria = static_cast<GroupingCriteria>(
            m_secondaryCombo->currentData().toInt());
    }
    
    options.dateGrouping = static_cast<DateGrouping>(m_dateGroupingCombo->currentData().toInt());
    options.caseSensitiveTypes = m_caseSensitiveCheckBox->isChecked();
    options.groupByParentDirectory = m_groupByParentDirCheckBox->isChecked();
    
    return options;
}

void GroupingOptionsDialog::setGroupingOptions(const GroupingOptions& options) {
    // Set primary criteria
    QAbstractButton* primaryButton = m_primaryButtonGroup->button(static_cast<int>(options.primaryCriteria));
    if (primaryButton) {
        primaryButton->setChecked(true);
    }
    
    // Set secondary criteria
    m_useSecondaryCheckBox->setChecked(options.useSecondaryCriteria);
    
    populateSecondaryCombo();
    
    if (options.useSecondaryCriteria) {
        int secondaryIndex = m_secondaryCombo->findData(static_cast<int>(options.secondaryCriteria));
        if (secondaryIndex >= 0) {
            m_secondaryCombo->setCurrentIndex(secondaryIndex);
        }
    }
    
    // Set additional options
    int dateIndex = m_dateGroupingCombo->findData(static_cast<int>(options.dateGrouping));
    if (dateIndex >= 0) {
        m_dateGroupingCombo->setCurrentIndex(dateIndex);
    }
    
    m_caseSensitiveCheckBox->setChecked(options.caseSensitiveTypes);
    m_groupByParentDirCheckBox->setChecked(options.groupByParentDirectory);
    
    updateSecondaryOptions();
    updatePreview();
}

GroupingOptionsDialog::GroupingCriteria GroupingOptionsDialog::getSelectedPrimaryCriteria() const {
    int buttonId = m_primaryButtonGroup->checkedId();
    if (buttonId >= 0) {
        return static_cast<GroupingCriteria>(buttonId);
    }
    return GroupingCriteria::Hash; // Default
}

GroupingOptionsDialog::GroupingCriteria GroupingOptionsDialog::getSelectedSecondaryCriteria() const {
    if (m_secondaryCombo->currentIndex() >= 0) {
        return static_cast<GroupingCriteria>(m_secondaryCombo->currentData().toInt());
    }
    return GroupingCriteria::Size; // Default
}

QString GroupingOptionsDialog::formatCriteriaName(GroupingCriteria criteria) {
    switch (criteria) {
        case GroupingCriteria::Hash:
            return tr("Hash");
        case GroupingCriteria::Size:
            return tr("Size");
        case GroupingCriteria::Type:
            return tr("Type");
        case GroupingCriteria::CreationDate:
            return tr("Creation Date");
        case GroupingCriteria::ModificationDate:
            return tr("Modification Date");
        case GroupingCriteria::Location:
            return tr("Location");
    }
    return tr("Unknown");
}

QString GroupingOptionsDialog::getGroupingDescription(const GroupingOptions& options) {
    QString description = tr("Files will be grouped by %1")
                         .arg(GroupingOptionsDialog::formatCriteriaName(options.primaryCriteria).toLower());
    
    if (options.useSecondaryCriteria) {
        description += tr(", then by %1")
                      .arg(GroupingOptionsDialog::formatCriteriaName(options.secondaryCriteria).toLower());
    }
    
    // Add specific options
    if (options.primaryCriteria == GroupingCriteria::CreationDate || 
        options.primaryCriteria == GroupingCriteria::ModificationDate ||
        (options.useSecondaryCriteria && 
         (options.secondaryCriteria == GroupingCriteria::CreationDate ||
          options.secondaryCriteria == GroupingCriteria::ModificationDate))) {
        
        QString dateDesc;
        switch (options.dateGrouping) {
            case DateGrouping::ExactDate:
                dateDesc = tr("exact date and time");
                break;
            case DateGrouping::SameDay:
                dateDesc = tr("same day");
                break;
            case DateGrouping::SameWeek:
                dateDesc = tr("same week");
                break;
            case DateGrouping::SameMonth:
                dateDesc = tr("same month");
                break;
            case DateGrouping::SameYear:
                dateDesc = tr("same year");
                break;
        }
        description += tr(" (using %1)").arg(dateDesc);
    }
    
    if (options.caseSensitiveTypes && 
        (options.primaryCriteria == GroupingCriteria::Type ||
         (options.useSecondaryCriteria && options.secondaryCriteria == GroupingCriteria::Type))) {
        description += tr(". File types will be case-sensitive");
    }
    
    if (options.groupByParentDirectory && 
        (options.primaryCriteria == GroupingCriteria::Location ||
         (options.useSecondaryCriteria && options.secondaryCriteria == GroupingCriteria::Location))) {
        description += tr(". Parent directories will be considered");
    }
    
    description += tr(".");
    
    return description;
}