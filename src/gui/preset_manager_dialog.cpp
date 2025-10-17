#include "preset_manager_dialog.h"
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QSplitter>
#include <QtCore/QStandardPaths>

PresetManagerDialog::PresetManagerDialog(QWidget* parent)
    : QDialog(parent)
    , m_presetList(nullptr)
    , m_presetDetails(nullptr)
    , m_newButton(nullptr)
    , m_editButton(nullptr)
    , m_deleteButton(nullptr)
    , m_loadButton(nullptr)
    , m_closeButton(nullptr)
    , m_settings(new QSettings(this))
{
    setWindowTitle(tr("Manage Scan Presets"));
    setMinimumSize(700, 500);
    resize(750, 550);
    setModal(true);
    
    setupUI();
    setupConnections();
    loadPresets();
    updateButtonStates();
}

PresetManagerDialog::~PresetManagerDialog()
{
    // Qt handles cleanup
}

void PresetManagerDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(16, 16, 16, 16);
    mainLayout->setSpacing(12);
    
    // Title
    QLabel* titleLabel = new QLabel(tr("📋 Scan Configuration Presets"), this);
    titleLabel->setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 8px;");
    mainLayout->addWidget(titleLabel);
    
    // Splitter for list and details
    QSplitter* splitter = new QSplitter(Qt::Horizontal, this);
    
    // Left side: Preset list
    QWidget* listWidget = new QWidget(this);
    QVBoxLayout* listLayout = new QVBoxLayout(listWidget);
    listLayout->setContentsMargins(0, 0, 0, 0);
    listLayout->setSpacing(8);
    
    QLabel* listLabel = new QLabel(tr("Available Presets:"), this);
    listLabel->setStyleSheet("font-weight: bold;");
    listLayout->addWidget(listLabel);
    
    m_presetList = new QListWidget(this);
    m_presetList->setAlternatingRowColors(true);
    m_presetList->setStyleSheet(
        "QListWidget {"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    padding: 4px;"
        "    background: palette(base);"
        "}"
        "QListWidget::item {"
        "    padding: 8px;"
        "    margin: 2px;"
        "    border-radius: 3px;"
        "}"
        "QListWidget::item:selected {"
        "    background: palette(highlight);"
        "    color: palette(highlighted-text);"
        "}"
    );
    listLayout->addWidget(m_presetList);
    
    // Buttons for list
    QHBoxLayout* listButtonLayout = new QHBoxLayout();
    listButtonLayout->setSpacing(8);
    
    m_newButton = new QPushButton(tr("+ New"), this);
    m_newButton->setToolTip(tr("Create a new preset from current configuration"));
    m_editButton = new QPushButton(tr("✏ Edit"), this);
    m_editButton->setToolTip(tr("Edit the selected preset"));
    m_deleteButton = new QPushButton(tr("🗑 Delete"), this);
    m_deleteButton->setToolTip(tr("Delete the selected preset"));
    
    QString buttonStyle = 
        "QPushButton {"
        "    padding: 6px 12px;"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(button);"
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
    
    m_newButton->setStyleSheet(buttonStyle);
    m_editButton->setStyleSheet(buttonStyle);
    m_deleteButton->setStyleSheet(buttonStyle);
    
    listButtonLayout->addWidget(m_newButton);
    listButtonLayout->addWidget(m_editButton);
    listButtonLayout->addWidget(m_deleteButton);
    listButtonLayout->addStretch();
    
    listLayout->addLayout(listButtonLayout);
    
    // Right side: Preset details
    QWidget* detailsWidget = new QWidget(this);
    QVBoxLayout* detailsLayout = new QVBoxLayout(detailsWidget);
    detailsLayout->setContentsMargins(0, 0, 0, 0);
    detailsLayout->setSpacing(8);
    
    QLabel* detailsLabel = new QLabel(tr("Preset Details:"), this);
    detailsLabel->setStyleSheet("font-weight: bold;");
    detailsLayout->addWidget(detailsLabel);
    
    m_presetDetails = new QTextEdit(this);
    m_presetDetails->setReadOnly(true);
    m_presetDetails->setStyleSheet(
        "QTextEdit {"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    padding: 8px;"
        "    background: palette(base);"
        "    font-family: monospace;"
        "}"
    );
    detailsLayout->addWidget(m_presetDetails);
    
    splitter->addWidget(listWidget);
    splitter->addWidget(detailsWidget);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 2);
    
    mainLayout->addWidget(splitter);
    
    // Bottom buttons
    QHBoxLayout* bottomLayout = new QHBoxLayout();
    bottomLayout->setSpacing(8);
    
    m_loadButton = new QPushButton(tr("Load Preset"), this);
    m_loadButton->setToolTip(tr("Load the selected preset into scan configuration"));
    m_closeButton = new QPushButton(tr("Close"), this);
    
    QString primaryButtonStyle = 
        "QPushButton {"
        "    background: palette(highlight);"
        "    color: palette(highlighted-text);"
        "    border: none;"
        "    padding: 8px 16px;"
        "    border-radius: 4px;"
        "    font-weight: bold;"
        "    min-width: 100px;"
        "}"
        "QPushButton:hover {"
        "    background: palette(dark);"
        "}"
        "QPushButton:disabled {"
        "    background: palette(mid);"
        "    color: palette(window);"
        "}"
    ;
    
    QString normalButtonStyle = 
        "QPushButton {"
        "    padding: 8px 16px;"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(button);"
        "    min-width: 80px;"
        "}"
        "QPushButton:hover {"
        "    background: palette(light);"
        "    border-color: palette(highlight);"
        "}"
    ;
    
    m_loadButton->setStyleSheet(primaryButtonStyle);
    m_closeButton->setStyleSheet(normalButtonStyle);
    
    bottomLayout->addStretch();
    bottomLayout->addWidget(m_loadButton);
    bottomLayout->addWidget(m_closeButton);
    
    mainLayout->addLayout(bottomLayout);
}

void PresetManagerDialog::setupConnections()
{
    connect(m_presetList, &QListWidget::itemSelectionChanged, 
            this, &PresetManagerDialog::onPresetSelectionChanged);
    connect(m_presetList, &QListWidget::itemDoubleClicked,
            this, &PresetManagerDialog::onLoadPreset);
    
    connect(m_newButton, &QPushButton::clicked, this, &PresetManagerDialog::onNewPreset);
    connect(m_editButton, &QPushButton::clicked, this, &PresetManagerDialog::onEditPreset);
    connect(m_deleteButton, &QPushButton::clicked, this, &PresetManagerDialog::onDeletePreset);
    connect(m_loadButton, &QPushButton::clicked, this, &PresetManagerDialog::onLoadPreset);
    connect(m_closeButton, &QPushButton::clicked, this, &QDialog::accept);
}

void PresetManagerDialog::loadPresets()
{
    m_presets.clear();
    m_presetList->clear();
    
    // Load built-in presets
    loadBuiltInPresets();
    
    // Load user presets from settings
    QStringList userPresetNames = getPresetNamesFromSettings();
    for (const QString& name : userPresetNames) {
        PresetInfo preset = loadPresetFromSettings(name);
        if (!preset.name.isEmpty()) {
            m_presets[name] = preset;
        }
    }
    
    // Populate list widget
    for (auto it = m_presets.constBegin(); it != m_presets.constEnd(); ++it) {
        const PresetInfo& preset = it.value();
        QListWidgetItem* item = new QListWidgetItem(m_presetList);
        
        if (preset.isBuiltIn) {
            item->setText(QString("🔒 %1").arg(preset.name));
            item->setForeground(QBrush(QColor(100, 100, 100)));
        } else {
            item->setText(QString("📝 %1").arg(preset.name));
        }
        
        item->setData(Qt::UserRole, preset.name);
        item->setToolTip(preset.description);
    }
}

void PresetManagerDialog::loadBuiltInPresets()
{
    // Downloads preset
    PresetInfo downloads;
    downloads.name = "Downloads";
    downloads.description = "Scan Downloads folder for duplicates";
    downloads.isBuiltIn = true;
    downloads.config.targetPaths << QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
    downloads.config.detectionMode = ScanSetupDialog::DetectionMode::Smart;
    downloads.config.minimumFileSize = 0;
    downloads.config.maximumDepth = -1;
    downloads.config.includeHidden = false;
    downloads.config.includeSystem = false;
    downloads.config.followSymlinks = true;
    downloads.config.scanArchives = false;
    downloads.config.excludePatterns << "*.tmp" << "*.log";
    m_presets["Downloads"] = downloads;
    
    // Photos preset
    PresetInfo photos;
    photos.name = "Photos";
    photos.description = "Scan Pictures folder for duplicate images";
    photos.isBuiltIn = true;
    photos.config.targetPaths << QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    photos.config.detectionMode = ScanSetupDialog::DetectionMode::Deep;
    photos.config.minimumFileSize = 10 * 1024; // 10KB
    photos.config.maximumDepth = -1;
    photos.config.includeHidden = false;
    photos.config.includeSystem = false;
    photos.config.followSymlinks = true;
    photos.config.scanArchives = false;
    photos.config.excludePatterns << "*.tmp" << "Thumbs.db";
    m_presets["Photos"] = photos;
    
    // Documents preset
    PresetInfo documents;
    documents.name = "Documents";
    documents.description = "Scan Documents folder for duplicates";
    documents.isBuiltIn = true;
    documents.config.targetPaths << QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    documents.config.detectionMode = ScanSetupDialog::DetectionMode::Smart;
    documents.config.minimumFileSize = 0;
    documents.config.maximumDepth = -1;
    documents.config.includeHidden = false;
    documents.config.includeSystem = false;
    documents.config.followSymlinks = true;
    documents.config.scanArchives = false;
    documents.config.excludePatterns << "*.tmp" << "~*";
    m_presets["Documents"] = documents;
    
    // Media preset
    PresetInfo media;
    media.name = "Media";
    media.description = "Scan Music and Videos folders for duplicate media files";
    media.isBuiltIn = true;
    media.config.targetPaths << QStandardPaths::writableLocation(QStandardPaths::MusicLocation)
                             << QStandardPaths::writableLocation(QStandardPaths::MoviesLocation);
    media.config.detectionMode = ScanSetupDialog::DetectionMode::Media;
    media.config.minimumFileSize = 100 * 1024; // 100KB
    media.config.maximumDepth = -1;
    media.config.includeHidden = false;
    media.config.includeSystem = false;
    media.config.followSymlinks = true;
    media.config.scanArchives = false;
    media.config.excludePatterns << "*.tmp";
    m_presets["Media"] = media;
}

void PresetManagerDialog::savePreset(const PresetInfo& preset)
{
    if (preset.name.isEmpty()) {
        return;
    }
    
    // Don't allow overwriting built-in presets
    if (m_presets.contains(preset.name) && m_presets[preset.name].isBuiltIn) {
        QMessageBox::warning(this, tr("Cannot Save"),
                           tr("Cannot overwrite built-in preset '%1'.").arg(preset.name));
        return;
    }
    
    // Save to memory
    m_presets[preset.name] = preset;
    
    // Save to settings
    savePresetToSettings(preset);
    
    // Reload list
    loadPresets();
    
    // Select the new/updated preset
    for (int i = 0; i < m_presetList->count(); ++i) {
        QListWidgetItem* item = m_presetList->item(i);
        if (item->data(Qt::UserRole).toString() == preset.name) {
            m_presetList->setCurrentItem(item);
            break;
        }
    }
    
    emit presetSaved(preset.name);
}

void PresetManagerDialog::deletePreset(const QString& name)
{
    if (!m_presets.contains(name)) {
        return;
    }
    
    const PresetInfo& preset = m_presets[name];
    
    // Don't allow deleting built-in presets
    if (preset.isBuiltIn) {
        QMessageBox::warning(this, tr("Cannot Delete"),
                           tr("Cannot delete built-in preset '%1'.").arg(name));
        return;
    }
    
    // Confirm deletion
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        tr("Delete Preset"),
        tr("Are you sure you want to delete preset '%1'?").arg(name),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No
    );
    
    if (reply != QMessageBox::Yes) {
        return;
    }
    
    // Remove from memory
    m_presets.remove(name);
    
    // Remove from settings
    m_settings->beginGroup("presets/scan");
    m_settings->remove(name);
    m_settings->endGroup();
    
    // Reload list
    loadPresets();
    
    emit presetDeleted(name);
}

QList<PresetManagerDialog::PresetInfo> PresetManagerDialog::getUserPresets() const
{
    QList<PresetInfo> userPresets;
    
    for (auto it = m_presets.constBegin(); it != m_presets.constEnd(); ++it) {
        if (!it.value().isBuiltIn) {
            userPresets.append(it.value());
        }
    }
    
    return userPresets;
}

PresetManagerDialog::PresetInfo PresetManagerDialog::getPreset(const QString& name) const
{
    return m_presets.value(name, PresetInfo());
}

QString PresetManagerDialog::getSelectedPresetName() const
{
    QListWidgetItem* item = m_presetList->currentItem();
    if (!item) {
        return QString();
    }
    
    return item->data(Qt::UserRole).toString();
}

PresetManagerDialog::PresetInfo PresetManagerDialog::getSelectedPreset() const
{
    QString name = getSelectedPresetName();
    return getPreset(name);
}

void PresetManagerDialog::onPresetSelectionChanged()
{
    updatePresetDetails();
    updateButtonStates();
}

void PresetManagerDialog::onEditPreset()
{
    QString name = getSelectedPresetName();
    if (name.isEmpty()) {
        return;
    }
    
    PresetInfo preset = getPreset(name);
    
    if (preset.isBuiltIn) {
        QMessageBox::information(this, tr("Cannot Edit"),
                                tr("Cannot edit built-in preset. Create a new preset instead."));
        return;
    }
    
    // Show dialog to edit description
    bool ok;
    QString newDescription = QInputDialog::getMultiLineText(
        this,
        tr("Edit Preset Description"),
        tr("Description for '%1':").arg(name),
        preset.description,
        &ok
    );
    
    if (ok) {
        preset.description = newDescription;
        savePreset(preset);
    }
}

void PresetManagerDialog::onDeletePreset()
{
    QString name = getSelectedPresetName();
    if (!name.isEmpty()) {
        deletePreset(name);
    }
}

void PresetManagerDialog::onLoadPreset()
{
    QString name = getSelectedPresetName();
    if (!name.isEmpty()) {
        emit presetSelected(name);
        accept();
    }
}

void PresetManagerDialog::onNewPreset()
{
    // Show dialog to get preset name and description
    bool ok;
    QString name = QInputDialog::getText(
        this,
        tr("New Preset"),
        tr("Preset name:"),
        QLineEdit::Normal,
        QString(),
        &ok
    );
    
    if (!ok || name.isEmpty()) {
        return;
    }
    
    // Check if name already exists
    if (m_presets.contains(name)) {
        QMessageBox::warning(this, tr("Name Exists"),
                           tr("A preset with name '%1' already exists.").arg(name));
        return;
    }
    
    QString description = QInputDialog::getMultiLineText(
        this,
        tr("New Preset"),
        tr("Description (optional):"),
        QString(),
        &ok
    );
    
    if (!ok) {
        return;
    }
    
    // Create new preset with default configuration
    PresetInfo preset;
    preset.name = name;
    preset.description = description;
    preset.isBuiltIn = false;
    
    // Set default configuration
    preset.config.detectionMode = ScanSetupDialog::DetectionMode::Smart;
    preset.config.minimumFileSize = 0;
    preset.config.maximumDepth = -1;
    preset.config.includeHidden = false;
    preset.config.includeSystem = false;
    preset.config.followSymlinks = true;
    preset.config.scanArchives = false;
    preset.config.excludePatterns << "*.tmp" << "*.log";
    
    savePreset(preset);
}

void PresetManagerDialog::updatePresetDetails()
{
    QString name = getSelectedPresetName();
    if (name.isEmpty()) {
        m_presetDetails->clear();
        return;
    }
    
    PresetInfo preset = getPreset(name);
    QString details = formatConfiguration(preset.config);
    
    QString fullDetails = QString("<h3>%1</h3>").arg(preset.name);
    
    if (!preset.description.isEmpty()) {
        fullDetails += QString("<p><i>%1</i></p>").arg(preset.description);
    }
    
    if (preset.isBuiltIn) {
        fullDetails += "<p><b>Type:</b> Built-in (read-only)</p>";
    } else {
        fullDetails += "<p><b>Type:</b> User-defined</p>";
    }
    
    fullDetails += "<hr>";
    fullDetails += details;
    
    m_presetDetails->setHtml(fullDetails);
}

void PresetManagerDialog::updateButtonStates()
{
    QString name = getSelectedPresetName();
    bool hasSelection = !name.isEmpty();
    bool isBuiltIn = hasSelection && m_presets.value(name).isBuiltIn;
    
    m_editButton->setEnabled(hasSelection && !isBuiltIn);
    m_deleteButton->setEnabled(hasSelection && !isBuiltIn);
    m_loadButton->setEnabled(hasSelection);
}

void PresetManagerDialog::savePresetToSettings(const PresetInfo& preset)
{
    m_settings->beginGroup("presets/scan");
    m_settings->beginGroup(preset.name);
    
    m_settings->setValue("description", preset.description);
    m_settings->setValue("isBuiltIn", preset.isBuiltIn);
    
    // Save configuration
    m_settings->setValue("targetPaths", preset.config.targetPaths);
    m_settings->setValue("detectionMode", static_cast<int>(preset.config.detectionMode));
    m_settings->setValue("minimumFileSize", preset.config.minimumFileSize);
    m_settings->setValue("maximumDepth", preset.config.maximumDepth);
    m_settings->setValue("excludePatterns", preset.config.excludePatterns);
    m_settings->setValue("excludeFolders", preset.config.excludeFolders);
    m_settings->setValue("includeHidden", preset.config.includeHidden);
    m_settings->setValue("includeSystem", preset.config.includeSystem);
    m_settings->setValue("followSymlinks", preset.config.followSymlinks);
    m_settings->setValue("scanArchives", preset.config.scanArchives);
    m_settings->setValue("fileTypeFilter", static_cast<int>(preset.config.fileTypeFilter));
    
    m_settings->endGroup();
    m_settings->endGroup();
}

PresetManagerDialog::PresetInfo PresetManagerDialog::loadPresetFromSettings(const QString& name) const
{
    PresetInfo preset;
    
    m_settings->beginGroup("presets/scan");
    m_settings->beginGroup(name);
    
    preset.name = name;
    preset.description = m_settings->value("description").toString();
    preset.isBuiltIn = m_settings->value("isBuiltIn", false).toBool();
    
    // Load configuration
    preset.config.targetPaths = m_settings->value("targetPaths").toStringList();
    preset.config.detectionMode = static_cast<ScanSetupDialog::DetectionMode>(
        m_settings->value("detectionMode", static_cast<int>(ScanSetupDialog::DetectionMode::Smart)).toInt()
    );
    preset.config.minimumFileSize = m_settings->value("minimumFileSize", 0).toLongLong();
    preset.config.maximumDepth = m_settings->value("maximumDepth", -1).toInt();
    preset.config.excludePatterns = m_settings->value("excludePatterns").toStringList();
    preset.config.excludeFolders = m_settings->value("excludeFolders").toStringList();
    preset.config.includeHidden = m_settings->value("includeHidden", false).toBool();
    preset.config.includeSystem = m_settings->value("includeSystem", false).toBool();
    preset.config.followSymlinks = m_settings->value("followSymlinks", true).toBool();
    preset.config.scanArchives = m_settings->value("scanArchives", false).toBool();
    preset.config.fileTypeFilter = static_cast<ScanSetupDialog::FileTypeFilter>(
        m_settings->value("fileTypeFilter", 0).toInt()
    );
    
    m_settings->endGroup();
    m_settings->endGroup();
    
    return preset;
}

QStringList PresetManagerDialog::getPresetNamesFromSettings() const
{
    m_settings->beginGroup("presets/scan");
    QStringList names = m_settings->childGroups();
    m_settings->endGroup();
    
    return names;
}

QString PresetManagerDialog::formatConfiguration(const ScanSetupDialog::ScanConfiguration& config) const
{
    QString html;
    
    html += "<p><b>Target Paths:</b></p><ul>";
    if (config.targetPaths.isEmpty()) {
        html += "<li><i>None selected</i></li>";
    } else {
        for (const QString& path : config.targetPaths) {
            html += QString("<li>%1</li>").arg(path);
        }
    }
    html += "</ul>";
    
    html += QString("<p><b>Detection Mode:</b> %1</p>").arg(formatDetectionMode(config.detectionMode));
    html += QString("<p><b>Minimum File Size:</b> %1</p>").arg(formatFileSize(config.minimumFileSize));
    html += QString("<p><b>Maximum Depth:</b> %1</p>").arg(config.maximumDepth < 0 ? "Unlimited" : QString::number(config.maximumDepth));
    
    html += "<p><b>Options:</b></p><ul>";
    html += QString("<li>Include Hidden: %1</li>").arg(config.includeHidden ? "Yes" : "No");
    html += QString("<li>Include System: %1</li>").arg(config.includeSystem ? "Yes" : "No");
    html += QString("<li>Follow Symlinks: %1</li>").arg(config.followSymlinks ? "Yes" : "No");
    html += QString("<li>Scan Archives: %1</li>").arg(config.scanArchives ? "Yes" : "No");
    html += "</ul>";
    
    if (!config.excludePatterns.isEmpty()) {
        html += "<p><b>Exclude Patterns:</b></p><ul>";
        for (const QString& pattern : config.excludePatterns) {
            html += QString("<li>%1</li>").arg(pattern);
        }
        html += "</ul>";
    }
    
    if (!config.excludeFolders.isEmpty()) {
        html += "<p><b>Exclude Folders:</b></p><ul>";
        for (const QString& folder : config.excludeFolders) {
            html += QString("<li>%1</li>").arg(folder);
        }
        html += "</ul>";
    }
    
    return html;
}

QString PresetManagerDialog::formatDetectionMode(ScanSetupDialog::DetectionMode mode) const
{
    switch (mode) {
        case ScanSetupDialog::DetectionMode::Quick:
            return "Quick (Size + Name)";
        case ScanSetupDialog::DetectionMode::Smart:
            return "Smart (Recommended)";
        case ScanSetupDialog::DetectionMode::Deep:
            return "Deep (Hash-based)";
        case ScanSetupDialog::DetectionMode::Media:
            return "Media (With Metadata)";
        default:
            return "Unknown";
    }
}

QString PresetManagerDialog::formatFileSize(qint64 bytes) const
{
    if (bytes == 0) {
        return "0 bytes";
    }
    
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    
    if (bytes >= GB) {
        return QString("%1 GB").arg(bytes / (double)GB, 0, 'f', 2);
    } else if (bytes >= MB) {
        return QString("%1 MB").arg(bytes / (double)MB, 0, 'f', 2);
    } else if (bytes >= KB) {
        return QString("%1 KB").arg(bytes / (double)KB, 0, 'f', 2);
    } else {
        return QString("%1 bytes").arg(bytes);
    }
}
