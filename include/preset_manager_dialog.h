#ifndef PRESET_MANAGER_DIALOG_H
#define PRESET_MANAGER_DIALOG_H

#include <QtWidgets/QDialog>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtCore/QSettings>
#include "scan_dialog.h"

class PresetManagerDialog : public QDialog
{
    Q_OBJECT

public:
    struct PresetInfo {
        QString name;
        QString description;
        ScanSetupDialog::ScanConfiguration config;
        bool isBuiltIn;
        
        PresetInfo() : isBuiltIn(false) {}
    };

    explicit PresetManagerDialog(QWidget* parent = nullptr);
    ~PresetManagerDialog();

    // Preset management
    void loadPresets();
    void savePreset(const PresetInfo& preset);
    void deletePreset(const QString& name);
    QList<PresetInfo> getUserPresets() const;
    PresetInfo getPreset(const QString& name) const;
    
    // Get selected preset
    QString getSelectedPresetName() const;
    PresetInfo getSelectedPreset() const;

signals:
    void presetSelected(const QString& name);
    void presetDeleted(const QString& name);
    void presetSaved(const QString& name);

private slots:
    void onPresetSelectionChanged();
    void onEditPreset();
    void onDeletePreset();
    void onLoadPreset();
    void onNewPreset();

private:
    void setupUI();
    void setupConnections();
    void updatePresetDetails();
    void updateButtonStates();
    
    // Persistence
    void savePresetToSettings(const PresetInfo& preset);
    PresetInfo loadPresetFromSettings(const QString& name) const;
    QStringList getPresetNamesFromSettings() const;
    
    // Built-in presets
    void loadBuiltInPresets();
    PresetInfo getBuiltInPreset(const QString& name) const;
    
    // UI Components
    QListWidget* m_presetList;
    QTextEdit* m_presetDetails;
    QPushButton* m_newButton;
    QPushButton* m_editButton;
    QPushButton* m_deleteButton;
    QPushButton* m_loadButton;
    QPushButton* m_closeButton;
    
    // Data
    QMap<QString, PresetInfo> m_presets;
    QSettings* m_settings;
    
    // Helper methods
    QString formatConfiguration(const ScanSetupDialog::ScanConfiguration& config) const;
    QString formatDetectionMode(ScanSetupDialog::DetectionMode mode) const;
    QString formatFileSize(qint64 bytes) const;
};

#endif // PRESET_MANAGER_DIALOG_H
