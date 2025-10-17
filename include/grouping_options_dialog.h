#pragma once

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QRadioButton>
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>

/**
 * @brief Dialog for configuring how duplicate files are grouped (Task 13)
 * 
 * This dialog allows users to choose different grouping criteria:
 * - By hash (existing default)
 * - By size
 * - By type/extension
 * - By date (creation or modification)
 * - By location/directory
 * - Multiple criteria combinations
 */
class GroupingOptionsDialog : public QDialog {
    Q_OBJECT

public:
    enum class GroupingCriteria {
        Hash,           // Group by file hash (default)
        Size,           // Group by file size
        Type,           // Group by file extension/type
        CreationDate,   // Group by creation date
        ModificationDate, // Group by modification date
        Location        // Group by directory/location
    };

    enum class DateGrouping {
        ExactDate,      // Group by exact date
        SameDay,        // Group by same day
        SameWeek,       // Group by same week
        SameMonth,      // Group by same month
        SameYear        // Group by same year
    };

    struct GroupingOptions {
        GroupingCriteria primaryCriteria;
        GroupingCriteria secondaryCriteria;
        bool useSecondaryCriteria;
        DateGrouping dateGrouping;
        bool caseSensitiveTypes;
        bool groupByParentDirectory;
        
        GroupingOptions() 
            : primaryCriteria(GroupingCriteria::Hash)
            , secondaryCriteria(GroupingCriteria::Size)
            , useSecondaryCriteria(false)
            , dateGrouping(DateGrouping::SameDay)
            , caseSensitiveTypes(false)
            , groupByParentDirectory(false) {}
    };

    explicit GroupingOptionsDialog(QWidget* parent = nullptr);
    ~GroupingOptionsDialog() override = default;

    /**
     * @brief Get the current grouping options
     */
    GroupingOptions getGroupingOptions() const;

    /**
     * @brief Set the grouping options
     */
    void setGroupingOptions(const GroupingOptions& options);

    /**
     * @brief Get a user-friendly description of the grouping criteria
     */
    static QString getGroupingDescription(const GroupingOptions& options);

signals:
    /**
     * @brief Emitted when user wants to apply new grouping
     */
    void groupingChanged(const GroupingOptions& options);

private slots:
    void onPrimaryCriteriaChanged();
    void onSecondaryCriteriaToggled(bool enabled);
    void onApplyClicked();
    void onResetClicked();
    void updatePreview();

private:
    // UI Components
    QGroupBox* m_primaryGroup;
    QVBoxLayout* m_primaryLayout;
    QButtonGroup* m_primaryButtonGroup;
    QRadioButton* m_hashRadio;
    QRadioButton* m_sizeRadio;
    QRadioButton* m_typeRadio;
    QRadioButton* m_creationDateRadio;
    QRadioButton* m_modificationDateRadio;
    QRadioButton* m_locationRadio;

    QGroupBox* m_secondaryGroup;
    QVBoxLayout* m_secondaryLayout;
    QCheckBox* m_useSecondaryCheckBox;
    QComboBox* m_secondaryCombo;

    QGroupBox* m_optionsGroup;
    QGridLayout* m_optionsLayout;
    QComboBox* m_dateGroupingCombo;
    QCheckBox* m_caseSensitiveCheckBox;
    QCheckBox* m_groupByParentDirCheckBox;

    QLabel* m_previewLabel;
    
    QPushButton* m_applyButton;
    QPushButton* m_resetButton;
    QPushButton* m_cancelButton;

    // Helper methods
    void setupUI();
    void setupConnections();
    void populateSecondaryCombo();
    void updateSecondaryOptions();
    GroupingCriteria getSelectedPrimaryCriteria() const;
    GroupingCriteria getSelectedSecondaryCriteria() const;
    static QString formatCriteriaName(GroupingCriteria criteria);
};