#include "exclude_pattern_widget.h"
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QMenu>
#include <QtCore/QSettings>
#include <QtCore/QRegularExpression>
#include <QtGui/QFont>

ExcludePatternWidget::ExcludePatternWidget(QWidget* parent)
    : QWidget(parent)
    , m_mainLayout(nullptr)
    , m_titleLabel(nullptr)
    , m_patternList(nullptr)
    , m_patternInput(nullptr)
    , m_addButton(nullptr)
    , m_removeButton(nullptr)
    , m_testButton(nullptr)
    , m_addCommonButton(nullptr)
    , m_validationLabel(nullptr)
    , m_inputLayout(nullptr)
    , m_buttonLayout(nullptr)
{
    setupUI();
    setupConnections();
    applyStyles();
    updateButtonStates();
}

ExcludePatternWidget::~ExcludePatternWidget()
{
    // Qt handles cleanup
}

void ExcludePatternWidget::setupUI()
{
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(0, 0, 0, 0);
    m_mainLayout->setSpacing(8);
    
    // Title label
    m_titleLabel = new QLabel(tr("Exclude Patterns"), this);
    QFont titleFont = m_titleLabel->font();
    titleFont.setBold(true);
    titleFont.setPointSize(titleFont.pointSize() + 1);
    m_titleLabel->setFont(titleFont);
    
    // Pattern list
    m_patternList = new QListWidget(this);
    m_patternList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_patternList->setAlternatingRowColors(true);
    m_patternList->setMinimumHeight(120);
    m_patternList->setMaximumHeight(180);
    m_patternList->setToolTip(tr("List of file patterns to exclude from scanning"));
    
    // Input layout
    m_inputLayout = new QHBoxLayout();
    m_inputLayout->setSpacing(6);
    
    m_patternInput = new QLineEdit(this);
    m_patternInput->setPlaceholderText(tr("Enter pattern (e.g., *.tmp, *.log)"));
    m_patternInput->setToolTip(tr("Enter a wildcard pattern or filename to exclude"));
    
    m_addButton = new QPushButton(tr("Add"), this);
    m_addButton->setToolTip(tr("Add pattern to exclusion list"));
    m_addButton->setEnabled(false);
    
    m_inputLayout->addWidget(m_patternInput);
    m_inputLayout->addWidget(m_addButton);
    
    // Validation label
    m_validationLabel = new QLabel(this);
    m_validationLabel->setWordWrap(true);
    m_validationLabel->setVisible(false);
    m_validationLabel->setMinimumHeight(20);
    
    // Button layout
    m_buttonLayout = new QHBoxLayout();
    m_buttonLayout->setSpacing(6);
    
    m_removeButton = new QPushButton(tr("Remove"), this);
    m_removeButton->setToolTip(tr("Remove selected pattern"));
    m_removeButton->setEnabled(false);
    
    m_testButton = new QPushButton(tr("Test Pattern"), this);
    m_testButton->setToolTip(tr("Test patterns against a filename"));
    
    m_addCommonButton = new QPushButton(tr("Add Common"), this);
    m_addCommonButton->setToolTip(tr("Add commonly used exclusion patterns"));
    
    m_buttonLayout->addWidget(m_removeButton);
    m_buttonLayout->addWidget(m_testButton);
    m_buttonLayout->addWidget(m_addCommonButton);
    m_buttonLayout->addStretch();
    
    // Add to main layout
    m_mainLayout->addWidget(m_titleLabel);
    m_mainLayout->addWidget(m_patternList);
    m_mainLayout->addLayout(m_inputLayout);
    m_mainLayout->addWidget(m_validationLabel);
    m_mainLayout->addLayout(m_buttonLayout);
}

void ExcludePatternWidget::setupConnections()
{
    connect(m_addButton, &QPushButton::clicked, this, &ExcludePatternWidget::onAddButtonClicked);
    connect(m_removeButton, &QPushButton::clicked, this, &ExcludePatternWidget::onRemoveButtonClicked);
    connect(m_testButton, &QPushButton::clicked, this, &ExcludePatternWidget::onTestButtonClicked);
    connect(m_addCommonButton, &QPushButton::clicked, this, &ExcludePatternWidget::onAddCommonButtonClicked);
    
    connect(m_patternInput, &QLineEdit::textChanged, this, &ExcludePatternWidget::onPatternInputChanged);
    connect(m_patternInput, &QLineEdit::returnPressed, this, &ExcludePatternWidget::onPatternInputReturnPressed);
    
    connect(m_patternList, &QListWidget::itemSelectionChanged, this, &ExcludePatternWidget::onPatternListSelectionChanged);
}

void ExcludePatternWidget::applyStyles()
{
    // Style the pattern list
    m_patternList->setStyleSheet(
        "QListWidget {"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    padding: 4px;"
        "    background: palette(base);"
        "}"
        "QListWidget::item {"
        "    padding: 4px;"
        "    margin: 1px;"
        "}"
        "QListWidget::item:selected {"
        "    background: palette(highlight);"
        "    color: palette(highlighted-text);"
        "}"
    );
    
    // Style the input field
    m_patternInput->setStyleSheet(
        "QLineEdit {"
        "    padding: 6px;"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(base);"
        "}"
        "QLineEdit:focus {"
        "    border-color: palette(highlight);"
        "}"
    );
    
    // Style buttons
    QString buttonStyle = 
        "QPushButton {"
        "    padding: 6px 12px;"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(button);"
        "    color: palette(button-text);"
        "}"
        "QPushButton:hover {"
        "    background: palette(light);"
        "    border-color: palette(highlight);"
        "}"
        "QPushButton:pressed {"
        "    background: palette(dark);"
        "}"
        "QPushButton:disabled {"
        "    color: palette(mid);"
        "    background: palette(window);"
        "}"
    ;
    
    m_addButton->setStyleSheet(buttonStyle);
    m_removeButton->setStyleSheet(buttonStyle);
    m_testButton->setStyleSheet(buttonStyle);
    m_addCommonButton->setStyleSheet(buttonStyle);
}

QStringList ExcludePatternWidget::getPatterns() const
{
    return m_patterns;
}

void ExcludePatternWidget::setPatterns(const QStringList& patterns)
{
    m_patterns.clear();
    m_patternList->clear();
    
    for (const QString& pattern : patterns) {
        if (!pattern.trimmed().isEmpty()) {
            m_patterns.append(pattern.trimmed());
            m_patternList->addItem(pattern.trimmed());
        }
    }
    
    emit patternsChanged(m_patterns);
}

bool ExcludePatternWidget::addPattern(const QString& pattern)
{
    QString trimmedPattern = pattern.trimmed();
    
    if (trimmedPattern.isEmpty()) {
        return false;
    }
    
    // Validate pattern
    QString errorMessage;
    if (!validatePattern(trimmedPattern, &errorMessage)) {
        showValidationFeedback(false, errorMessage);
        return false;
    }
    
    // Check for duplicates
    if (m_patterns.contains(trimmedPattern)) {
        showValidationFeedback(false, tr("Pattern already exists"));
        return false;
    }
    
    // Add pattern
    m_patterns.append(trimmedPattern);
    m_patternList->addItem(trimmedPattern);
    
    showValidationFeedback(true, tr("Pattern added successfully"));
    
    emit patternAdded(trimmedPattern);
    emit patternsChanged(m_patterns);
    
    return true;
}

void ExcludePatternWidget::removePattern(const QString& pattern)
{
    qsizetype index = m_patterns.indexOf(pattern);
    if (index >= 0) {
        m_patterns.removeAt(index);
        delete m_patternList->takeItem(static_cast<int>(index));
        
        emit patternRemoved(pattern);
        emit patternsChanged(m_patterns);
    }
}

void ExcludePatternWidget::clearPatterns()
{
    m_patterns.clear();
    m_patternList->clear();
    emit patternsChanged(m_patterns);
}

bool ExcludePatternWidget::validatePattern(const QString& pattern, QString* errorMessage)
{
    QString trimmedPattern = pattern.trimmed();
    
    // Check if empty
    if (trimmedPattern.isEmpty()) {
        if (errorMessage) {
            *errorMessage = tr("Pattern cannot be empty");
        }
        return false;
    }
    
    // Check for invalid characters in wildcards
    // Allow: alphanumeric, *, ?, ., -, _, /, \, space
    QRegularExpression validChars("^[a-zA-Z0-9*?.\\-_/\\\\ ]+$");
    if (!validChars.match(trimmedPattern).hasMatch()) {
        if (errorMessage) {
            *errorMessage = tr("Pattern contains invalid characters");
        }
        return false;
    }
    
    // If pattern looks like a regex (contains special regex chars), try to compile it
    if (trimmedPattern.contains(QRegularExpression("[\\[\\]\\(\\)\\{\\}\\^\\$\\+\\|]"))) {
        QRegularExpression regex(trimmedPattern);
        if (!regex.isValid()) {
            if (errorMessage) {
                *errorMessage = tr("Invalid regular expression: %1").arg(regex.errorString());
            }
            return false;
        }
    }
    
    // Pattern is valid
    if (errorMessage) {
        *errorMessage = tr("Valid pattern");
    }
    return true;
}

bool ExcludePatternWidget::matchesAnyPattern(const QString& filename) const
{
    for (const QString& pattern : m_patterns) {
        // Convert wildcard pattern to regex
        QString regexPattern = QRegularExpression::wildcardToRegularExpression(pattern);
        QRegularExpression regex(regexPattern, QRegularExpression::CaseInsensitiveOption);
        
        if (regex.match(filename).hasMatch()) {
            return true;
        }
    }
    
    return false;
}

void ExcludePatternWidget::loadFromSettings(const QString& settingsKey)
{
    QSettings settings;
    QStringList patterns = settings.value(settingsKey, QStringList()).toStringList();
    setPatterns(patterns);
}

void ExcludePatternWidget::saveToSettings(const QString& settingsKey)
{
    QSettings settings;
    settings.setValue(settingsKey, m_patterns);
}

void ExcludePatternWidget::onAddButtonClicked()
{
    QString pattern = m_patternInput->text().trimmed();
    if (addPattern(pattern)) {
        m_patternInput->clear();
        m_validationLabel->setVisible(false);
    }
}

void ExcludePatternWidget::onRemoveButtonClicked()
{
    QListWidgetItem* currentItem = m_patternList->currentItem();
    if (currentItem) {
        QString pattern = currentItem->text();
        removePattern(pattern);
    }
}

void ExcludePatternWidget::onPatternInputChanged(const QString& text)
{
    QString trimmedText = text.trimmed();
    
    if (trimmedText.isEmpty()) {
        m_addButton->setEnabled(false);
        m_validationLabel->setVisible(false);
        return;
    }
    
    // Validate pattern
    QString errorMessage;
    bool valid = validatePattern(trimmedText, &errorMessage);
    
    m_addButton->setEnabled(valid);
    
    // Show validation feedback as user types
    if (!trimmedText.isEmpty()) {
        showValidationFeedback(valid, errorMessage);
    }
}

void ExcludePatternWidget::onPatternInputReturnPressed()
{
    if (m_addButton->isEnabled()) {
        onAddButtonClicked();
    }
}

void ExcludePatternWidget::onPatternListSelectionChanged()
{
    updateButtonStates();
}

void ExcludePatternWidget::onTestButtonClicked()
{
    bool ok;
    QString filename = QInputDialog::getText(
        this,
        tr("Test Pattern"),
        tr("Enter a filename to test against current patterns:"),
        QLineEdit::Normal,
        QString(),
        &ok
    );
    
    if (ok && !filename.isEmpty()) {
        bool matches = matchesAnyPattern(filename);
        
        QString message;
        if (matches) {
            message = tr("✓ Filename '%1' MATCHES one or more patterns and will be excluded.")
                     .arg(filename);
        } else {
            message = tr("✗ Filename '%1' does NOT match any patterns and will be included.")
                     .arg(filename);
        }
        
        QMessageBox::information(this, tr("Pattern Test Result"), message);
    }
}

void ExcludePatternWidget::onAddCommonButtonClicked()
{
    QStringList commonPatterns = getCommonPatterns();
    
    // Show menu with common patterns
    QMenu menu(this);
    menu.setTitle(tr("Common Patterns"));
    
    for (const QString& pattern : commonPatterns) {
        QAction* action = menu.addAction(pattern);
        connect(action, &QAction::triggered, this, [this, pattern]() {
            if (addPattern(pattern)) {
                m_validationLabel->setVisible(false);
            }
        });
    }
    
    menu.addSeparator();
    QAction* addAllAction = menu.addAction(tr("Add All Common Patterns"));
    connect(addAllAction, &QAction::triggered, this, [this, commonPatterns]() {
        int added = 0;
        for (const QString& pattern : commonPatterns) {
            if (!m_patterns.contains(pattern)) {
                if (addPattern(pattern)) {
                    added++;
                }
            }
        }
        
        if (added > 0) {
            showValidationFeedback(true, tr("Added %1 common patterns").arg(added));
        } else {
            showValidationFeedback(false, tr("All common patterns already added"));
        }
    });
    
    menu.exec(QCursor::pos());
}

void ExcludePatternWidget::updateButtonStates()
{
    m_removeButton->setEnabled(m_patternList->currentItem() != nullptr);
}

void ExcludePatternWidget::showValidationFeedback(bool valid, const QString& message)
{
    if (message.isEmpty()) {
        m_validationLabel->setVisible(false);
        return;
    }
    
    m_validationLabel->setText(message);
    m_validationLabel->setVisible(true);
    
    // Style based on validity
    if (valid) {
        m_validationLabel->setStyleSheet(
            "QLabel {"
            "    color: green;"
            "    padding: 4px;"
            "    background: #e8f5e9;"
            "    border: 1px solid #4caf50;"
            "    border-radius: 3px;"
            "}"
        );
    } else {
        m_validationLabel->setStyleSheet(
            "QLabel {"
            "    color: #d32f2f;"
            "    padding: 4px;"
            "    background: #ffebee;"
            "    border: 1px solid #f44336;"
            "    border-radius: 3px;"
            "}"
        );
    }
}

QStringList ExcludePatternWidget::getCommonPatterns() const
{
    return QStringList{
        "*.tmp",
        "*.log",
        "*.bak",
        "*.cache",
        "*.swp",
        "Thumbs.db",
        ".DS_Store",
        "desktop.ini",
        "*.temp",
        "~*",
        "*.old"
    };
}
