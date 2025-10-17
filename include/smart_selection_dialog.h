#ifndef SMART_SELECTION_DIALOG_H
#define SMART_SELECTION_DIALOG_H

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QSpinBox>
#include <QDateTimeEdit>
#include <QCheckBox>
#include <QRadioButton>
#include <QButtonGroup>
#include <QGroupBox>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QTextEdit>
#include <QSlider>
#include <QProgressBar>
#include <QDateTime>
#include <QStringList>

/**
 * @brief Dialog for smart file selection based on various criteria
 * 
 * This dialog allows users to select files using intelligent criteria
 * such as file age, size, location patterns, and combinations thereof.
 * 
 * Features:
 * - Multiple selection modes (oldest, newest, largest, smallest, by path)
 * - Date range filtering
 * - Size range filtering
 * - Path pattern matching
 * - Criteria combination (AND/OR logic)
 * - Preview of selection results
 * - Save/load selection presets
 * 
 * Requirements: 4.1, 4.8
 */
class SmartSelectionDialog : public QDialog {
    Q_OBJECT

public:
    enum SelectionMode {
        OldestFiles,
        NewestFiles,
        LargestFiles,
        SmallestFiles,
        ByPath,
        ByCriteria,
        ByFileType,
        ByLocation
    };

    struct SelectionCriteria {
        SelectionMode mode;
        QString pathPattern;
        QDateTime dateFrom;
        QDateTime dateTo;
        qint64 minSize;
        qint64 maxSize;
        QStringList fileTypes;
        QStringList locationPatterns;
        bool useAnd; // true = AND, false = OR
        bool useDateRange;
        bool useSizeRange;
        bool useFileTypes;
        bool useLocationPatterns;
        int maxFiles; // Limit number of selected files
        double percentage; // Select percentage of files
        
        SelectionCriteria() 
            : mode(OldestFiles)
            , dateFrom(QDateTime::currentDateTime().addDays(-30))
            , dateTo(QDateTime::currentDateTime())
            , minSize(0)
            , maxSize(1024 * 1024 * 1024) // 1GB
            , useAnd(true)
            , useDateRange(false)
            , useSizeRange(false)
            , useFileTypes(false)
            , useLocationPatterns(false)
            , maxFiles(100)
            , percentage(50.0)
        {}
    };

    explicit SmartSelectionDialog(QWidget* parent = nullptr);
    ~SmartSelectionDialog();

    // Main interface
    SelectionCriteria getCriteria() const;
    void setCriteria(const SelectionCriteria& criteria);
    void setAvailableFiles(const QStringList& filePaths, const QList<qint64>& fileSizes, 
                          const QList<QDateTime>& fileDates);
    
    // Preview functionality
    void updatePreview();
    QStringList getPreviewResults() const;
    int getEstimatedSelectionCount() const;

    // Preset management
    void savePreset(const QString& name);
    void loadPreset(const QString& name);
    QStringList getAvailablePresets() const;

public slots:
    void onModeChanged();
    void onCriteriaChanged();
    void onPreviewRequested();
    void onResetToDefaults();

signals:
    void selectionRequested(const SelectionCriteria& criteria);
    void previewUpdated(const QStringList& selectedFiles);
    void criteriaChanged();

protected:
    void showEvent(QShowEvent* event) override;

private slots:
    void onModeComboChanged(int index);
    void onLogicChanged();
    void onDateRangeToggled(bool enabled);
    void onSizeRangeToggled(bool enabled);
    void onFileTypesToggled(bool enabled);
    void onLocationPatternsToggled(bool enabled);
    void onMaxFilesChanged(int value);
    void onPercentageChanged(double value);
    void onPresetComboChanged(const QString& presetName);
    void onSavePresetClicked();
    void onDeletePresetClicked();

private:
    void setupUI();
    void createModeSection();
    void createCriteriaSection();
    void createPreviewSection();
    void createPresetSection();
    void createButtonBox();
    void setupConnections();
    void updateUIFromCriteria();
    void updateCriteriaFromUI();
    void enableCriteriaControls();
    void calculatePreview();
    void loadPresets();
    void savePresets();
    
    // Selection logic helpers
    QStringList selectOldestFiles(int count) const;
    QStringList selectNewestFiles(int count) const;
    QStringList selectLargestFiles(int count) const;
    QStringList selectSmallestFiles(int count) const;
    QStringList selectByPath(const QString& pattern) const;
    QStringList selectByCriteria() const;
    QStringList applyFilters(const QStringList& files) const;
    bool matchesDateRange(const QDateTime& date) const;
    bool matchesSizeRange(qint64 size) const;
    bool matchesFileType(const QString& filePath) const;
    bool matchesLocationPattern(const QString& filePath) const;

    // UI Components
    QVBoxLayout* m_mainLayout;
    
    // Mode section
    QGroupBox* m_modeGroup;
    QVBoxLayout* m_modeLayout;
    QComboBox* m_modeCombo;
    QLabel* m_modeDescription;
    
    // Criteria section
    QGroupBox* m_criteriaGroup;
    QGridLayout* m_criteriaLayout;
    
    // Date range
    QCheckBox* m_useDateRangeCheck;
    QDateTimeEdit* m_dateFromEdit;
    QDateTimeEdit* m_dateToEdit;
    QLabel* m_dateRangeLabel;
    
    // Size range
    QCheckBox* m_useSizeRangeCheck;
    QSpinBox* m_minSizeSpin;
    QSpinBox* m_maxSizeSpin;
    QComboBox* m_sizeUnitCombo;
    QLabel* m_sizeRangeLabel;
    
    // File types
    QCheckBox* m_useFileTypesCheck;
    QLineEdit* m_fileTypesEdit;
    QLabel* m_fileTypesLabel;
    
    // Location patterns
    QCheckBox* m_useLocationPatternsCheck;
    QLineEdit* m_locationPatternsEdit;
    QLabel* m_locationPatternsLabel;
    
    // Path pattern (for ByPath mode)
    QLineEdit* m_pathPatternEdit;
    QLabel* m_pathPatternLabel;
    
    // Logic
    QLabel* m_logicLabel;
    QRadioButton* m_andRadio;
    QRadioButton* m_orRadio;
    QButtonGroup* m_logicGroup;
    
    // Limits
    QLabel* m_maxFilesLabel;
    QSpinBox* m_maxFilesSpin;
    QLabel* m_percentageLabel;
    QSlider* m_percentageSlider;
    QLabel* m_percentageValueLabel;
    
    // Preview section
    QGroupBox* m_previewGroup;
    QVBoxLayout* m_previewLayout;
    QLabel* m_previewCountLabel;
    QTextEdit* m_previewList;
    QPushButton* m_previewButton;
    QProgressBar* m_previewProgress;
    
    // Preset section
    QGroupBox* m_presetGroup;
    QHBoxLayout* m_presetLayout;
    QComboBox* m_presetCombo;
    QPushButton* m_savePresetButton;
    QPushButton* m_deletePresetButton;
    
    // Button box
    QDialogButtonBox* m_buttonBox;
    QPushButton* m_selectButton;
    QPushButton* m_cancelButton;
    QPushButton* m_resetButton;
    
    // Data
    SelectionCriteria m_criteria;
    QStringList m_availableFiles;
    QList<qint64> m_fileSizes;
    QList<QDateTime> m_fileDates;
    QStringList m_previewResults;
    QMap<QString, SelectionCriteria> m_presets;
    
    // State
    bool m_updating;
    
    // Constants
    static const int MAX_PREVIEW_FILES = 100;
    static const QStringList DEFAULT_FILE_TYPES;
    static const QStringList COMMON_LOCATION_PATTERNS;
};

Q_DECLARE_METATYPE(SmartSelectionDialog::SelectionCriteria)

#endif // SMART_SELECTION_DIALOG_H