#pragma once

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QDateEdit>
#include <QSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>
#include <QListWidget>
#include <QDateTime>
#include <QSettings>

/**
 * @brief Advanced Filter Dialog for filtering duplicate results (Task 11)
 * 
 * This dialog provides advanced filtering options:
 * - Date range filter (created/modified dates)
 * - File extension filter
 * - Path pattern filter
 * - Size range filter
 * - Multiple criteria combination
 */
class AdvancedFilterDialog : public QDialog {
    Q_OBJECT

public:
    /**
     * @brief Filter criteria structure
     */
    struct FilterCriteria {
        // Date filters
        bool enableDateFilter = false;
        QDateTime dateFrom;
        QDateTime dateTo;
        enum DateType { CreatedDate, ModifiedDate } dateType = ModifiedDate;
        
        // Extension filters
        bool enableExtensionFilter = false;
        QStringList includedExtensions;
        QStringList excludedExtensions;
        
        // Path filters
        bool enablePathFilter = false;
        QStringList pathPatterns;
        bool pathCaseSensitive = false;
        
        // Size filters
        bool enableSizeFilter = false;
        qint64 minSize = 0;
        qint64 maxSize = 0;
        enum SizeUnit { Bytes, KB, MB, GB } sizeUnit = MB;
        
        // Combination logic
        enum CombineMode { AND, OR } combineMode = AND;
    };

    explicit AdvancedFilterDialog(QWidget* parent = nullptr);
    ~AdvancedFilterDialog() override = default;

    /**
     * @brief Get the current filter criteria
     */
    FilterCriteria getFilterCriteria() const;

    /**
     * @brief Set the filter criteria
     */
    void setFilterCriteria(const FilterCriteria& criteria);

    /**
     * @brief Reset all filters to default
     */
    void resetFilters();

    /**
     * @brief Save current criteria as a preset (Task 12)
     */
    void savePreset(const QString& name);

    /**
     * @brief Load a preset by name (Task 12)
     */
    bool loadPreset(const QString& name);

    /**
     * @brief Get list of available presets (Task 12)
     */
    QStringList getAvailablePresets() const;

    /**
     * @brief Delete a preset (Task 12)
     */
    bool deletePreset(const QString& name);

signals:
    /**
     * @brief Emitted when filters should be applied
     */
    void filtersChanged(const FilterCriteria& criteria);

private slots:
    void onApplyClicked();
    void onResetClicked();
    void onAddExtensionClicked();
    void onRemoveExtensionClicked();
    void onAddPathPatternClicked();
    void onRemovePathPatternClicked();
    void updateSizeUnits();
    
    // Preset management slots (Task 12)
    void onSavePresetClicked();
    void onLoadPresetClicked();
    void onDeletePresetClicked();
    void onPresetSelectionChanged();

private:
    // UI Components
    
    // Date filter section
    QGroupBox* m_dateGroup;
    QCheckBox* m_enableDateFilter;
    QComboBox* m_dateTypeCombo;
    QDateEdit* m_dateFromEdit;
    QDateEdit* m_dateToEdit;
    
    // Extension filter section
    QGroupBox* m_extensionGroup;
    QCheckBox* m_enableExtensionFilter;
    QLineEdit* m_extensionInput;
    QPushButton* m_addExtensionButton;
    QListWidget* m_extensionList;
    QPushButton* m_removeExtensionButton;
    
    // Path filter section
    QGroupBox* m_pathGroup;
    QCheckBox* m_enablePathFilter;
    QLineEdit* m_pathPatternInput;
    QPushButton* m_addPathButton;
    QListWidget* m_pathPatternList;
    QPushButton* m_removePathButton;
    QCheckBox* m_pathCaseSensitive;
    
    // Size filter section
    QGroupBox* m_sizeGroup;
    QCheckBox* m_enableSizeFilter;
    QSpinBox* m_minSizeSpinBox;
    QSpinBox* m_maxSizeSpinBox;
    QComboBox* m_sizeUnitCombo;
    
    // Combination section
    QGroupBox* m_combineGroup;
    QComboBox* m_combineModeCombo;
    
    // Buttons
    QPushButton* m_applyButton;
    QPushButton* m_resetButton;
    QPushButton* m_closeButton;
    
    // Preset management (Task 12)
    QGroupBox* m_presetGroup;
    QComboBox* m_presetCombo;
    QPushButton* m_savePresetButton;
    QPushButton* m_loadPresetButton;
    QPushButton* m_deletePresetButton;

    // Helper methods
    void setupUI();
    void setupDateSection();
    void setupExtensionSection();
    void setupPathSection();
    void setupSizeSection();
    void setupCombineSection();
    void setupPresetSection();  // Task 12
    QHBoxLayout* setupButtons();
    void connectSignals();
    void loadPresetList();      // Task 12
    
    qint64 convertSizeToBytes(int value, FilterCriteria::SizeUnit unit) const;
    int convertSizeFromBytes(qint64 bytes, FilterCriteria::SizeUnit unit) const;
};