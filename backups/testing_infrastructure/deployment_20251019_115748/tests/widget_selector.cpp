#include "widget_selector.h"
#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QTextEdit>
#include <QComboBox>
#include <QCheckBox>
#include <QRadioButton>
#include <QDebug>
#include <QMetaObject>
#include <QRegularExpression>

WidgetSelector::WidgetSelector()
    : m_type(Type::ObjectName)
    , m_matchMode(MatchMode::Exact)
    , m_parentWidget(nullptr)
    , m_requireVisible(false)
    , m_requireEnabled(false)
    , m_requireFocusable(false)
    , m_index(-1)
    , m_timeoutMs(5000)
    , m_retryIntervalMs(100)
    , m_maxRetries(3)
{
}

WidgetSelector::WidgetSelector(Type type, const QString& value)
    : WidgetSelector()
{
    m_type = type;
    m_value = value;
}

WidgetSelector::WidgetSelector(const QString& cssSelector)
    : WidgetSelector()
{
    m_type = Type::CSS;
    parseCssSelector(cssSelector);
}

WidgetSelector& WidgetSelector::setType(Type type) {
    m_type = type;
    return *this;
}

WidgetSelector& WidgetSelector::setValue(const QString& value) {
    m_value = value;
    return *this;
}

WidgetSelector& WidgetSelector::setMatchMode(MatchMode mode) {
    m_matchMode = mode;
    return *this;
}

WidgetSelector& WidgetSelector::setParent(const QString& parentSelector) {
    m_parentSelector = parentSelector;
    return *this;
}

WidgetSelector& WidgetSelector::setParent(QWidget* parentWidget) {
    m_parentWidget = parentWidget;
    return *this;
}

WidgetSelector& WidgetSelector::addConstraint(const QString& property, const QVariant& value, MatchMode mode) {
    Constraint constraint;
    constraint.property = property;
    constraint.value = value;
    constraint.matchMode = mode;
    constraint.required = true;
    
    m_constraints.append(constraint);
    return *this;
}

WidgetSelector& WidgetSelector::requireVisible(bool visible) {
    m_requireVisible = visible;
    return *this;
}

WidgetSelector& WidgetSelector::requireEnabled(bool enabled) {
    m_requireEnabled = enabled;
    return *this;
}

WidgetSelector& WidgetSelector::setIndex(int index) {
    m_index = index;
    return *this;
}

WidgetSelector& WidgetSelector::setTimeout(int timeoutMs) {
    m_timeoutMs = timeoutMs;
    return *this;
}

WidgetSelector& WidgetSelector::setCustomPredicate(std::function<bool(QWidget*)> predicate) {
    m_customPredicate = predicate;
    m_type = Type::Custom;
    return *this;
}

QWidget* WidgetSelector::findWidget(QWidget* root) const {
    QList<QWidget*> widgets = findAllWidgets(root);
    
    if (widgets.isEmpty()) {
        return nullptr;
    }
    
    // Apply index selection
    if (m_index >= 0 && m_index < widgets.size()) {
        return widgets[m_index];
    }
    
    // Return first widget if no specific index
    return widgets.first();
}

QList<QWidget*> WidgetSelector::findAllWidgets(QWidget* root) const {
    QWidget* searchRoot = root;
    if (!searchRoot) {
        if (m_parentWidget) {
            searchRoot = m_parentWidget;
        } else {
            searchRoot = QApplication::activeWindow();
        }
    }
    
    if (!searchRoot) {
        // Search all top-level widgets
        QList<QWidget*> allResults;
        QWidgetList topLevelWidgets = QApplication::topLevelWidgets();
        for (QWidget* widget : topLevelWidgets) {
            if (widget->isVisible()) {
                QList<QWidget*> results = findWidgetsRecursive(widget);
                allResults.append(results);
            }
        }
        return filterByConstraints(allResults);
    }
    
    QList<QWidget*> results = findWidgetsRecursive(searchRoot);
    return filterByConstraints(results);
}

bool WidgetSelector::exists(QWidget* root) const {
    return !findAllWidgets(root).isEmpty();
}

int WidgetSelector::count(QWidget* root) const {
    return findAllWidgets(root).size();
}

bool WidgetSelector::matches(QWidget* widget) const {
    if (!widget) {
        return false;
    }
    
    // Check basic type matching
    bool typeMatches = false;
    
    switch (m_type) {
        case Type::ObjectName:
            typeMatches = matchesTextContent(widget, m_value, m_matchMode) && 
                         widget->objectName() == m_value;
            break;
            
        case Type::Text: {
            QString text;
            if (auto* label = qobject_cast<QLabel*>(widget)) {
                text = label->text();
            } else if (auto* button = qobject_cast<QPushButton*>(widget)) {
                text = button->text();
            } else if (auto* lineEdit = qobject_cast<QLineEdit*>(widget)) {
                text = lineEdit->text();
            } else {
                text = widget->property("text").toString();
            }
            typeMatches = matchesTextContent(widget, text, m_matchMode);
            break;
        }
        
        case Type::ClassName:
            typeMatches = widget->metaObject()->className() == m_value;
            break;
            
        case Type::Property:
            typeMatches = matchesProperty(widget, m_value, 
                                        m_constraints.isEmpty() ? QVariant() : m_constraints.first().value, 
                                        m_matchMode);
            break;
            
        case Type::Custom:
            typeMatches = m_customPredicate && m_customPredicate(widget);
            break;
            
        default:
            typeMatches = true;
            break;
    }
    
    if (!typeMatches) {
        return false;
    }
    
    // Check constraints
    if (!matchesConstraints(widget)) {
        return false;
    }
    
    // Check state requirements
    if (!matchesStateRequirements(widget)) {
        return false;
    }
    
    return true;
}

bool WidgetSelector::isValid() const {
    return !m_value.isEmpty() || m_type == Type::Custom || !m_constraints.isEmpty();
}

QString WidgetSelector::toString() const {
    QString result;
    
    switch (m_type) {
        case Type::ObjectName:
            result = QString("ObjectName('%1')").arg(m_value);
            break;
        case Type::Text:
            result = QString("Text('%1')").arg(m_value);
            break;
        case Type::ClassName:
            result = QString("ClassName('%1')").arg(m_value);
            break;
        case Type::Property:
            result = QString("Property('%1')").arg(m_value);
            break;
        case Type::Custom:
            result = "Custom(predicate)";
            break;
        default:
            result = "Unknown";
            break;
    }
    
    if (m_index >= 0) {
        result += QString("[%1]").arg(m_index);
    }
    
    return result;
}

// Static factory methods
WidgetSelector WidgetSelector::byObjectName(const QString& name) {
    return WidgetSelector(Type::ObjectName, name);
}

WidgetSelector WidgetSelector::byText(const QString& text, MatchMode mode) {
    WidgetSelector selector(Type::Text, text);
    selector.setMatchMode(mode);
    return selector;
}

WidgetSelector WidgetSelector::byClassName(const QString& className) {
    return WidgetSelector(Type::ClassName, className);
}

WidgetSelector WidgetSelector::byProperty(const QString& property, const QVariant& value) {
    WidgetSelector selector(Type::Property, property);
    selector.addConstraint(property, value);
    return selector;
}

WidgetSelector WidgetSelector::byCustom(std::function<bool(QWidget*)> predicate) {
    WidgetSelector selector;
    selector.setCustomPredicate(predicate);
    return selector;
}

// Private helper methods
bool WidgetSelector::matchesConstraints(QWidget* widget) const {
    for (const Constraint& constraint : m_constraints) {
        bool matches = matchesProperty(widget, constraint.property, constraint.value, constraint.matchMode);
        if (constraint.required && !matches) {
            return false;
        }
    }
    return true;
}

bool WidgetSelector::matchesStateRequirements(QWidget* widget) const {
    if (m_requireVisible && !widget->isVisible()) {
        return false;
    }
    
    if (m_requireEnabled && !widget->isEnabled()) {
        return false;
    }
    
    if (m_requireFocusable && !widget->focusPolicy() != Qt::NoFocus) {
        return false;
    }
    
    return true;
}

bool WidgetSelector::matchesTextContent(QWidget* widget, const QString& expectedText, MatchMode mode) const {
    Q_UNUSED(widget)
    
    QString actualText = m_value; // For now, use the selector value
    
    switch (mode) {
        case MatchMode::Exact:
            return actualText == expectedText;
        case MatchMode::Contains:
            return actualText.contains(expectedText);
        case MatchMode::StartsWith:
            return actualText.startsWith(expectedText);
        case MatchMode::EndsWith:
            return actualText.endsWith(expectedText);
        case MatchMode::CaseInsensitive:
            return actualText.compare(expectedText, Qt::CaseInsensitive) == 0;
        case MatchMode::Regex: {
            QRegularExpression regex(expectedText);
            return regex.match(actualText).hasMatch();
        }
        case MatchMode::Wildcard: {
            QRegularExpression regex = createWildcardRegex(expectedText);
            return regex.match(actualText).hasMatch();
        }
    }
    
    return false;
}

bool WidgetSelector::matchesProperty(QWidget* widget, const QString& property, const QVariant& expectedValue, MatchMode mode) const {
    QVariant actualValue = widget->property(property.toUtf8().constData());
    
    if (mode == MatchMode::Exact) {
        return actualValue == expectedValue;
    }
    
    // For other modes, convert to string and compare
    QString actualStr = actualValue.toString();
    QString expectedStr = expectedValue.toString();
    
    return matchesTextContent(widget, expectedStr, mode);
}

QList<QWidget*> WidgetSelector::findWidgetsRecursive(QWidget* root) const {
    QList<QWidget*> results;
    
    if (!root) {
        return results;
    }
    
    // Check if root matches
    if (matches(root)) {
        results.append(root);
    }
    
    // Check children recursively
    QList<QWidget*> children = root->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (QWidget* child : children) {
        QList<QWidget*> childResults = findWidgetsRecursive(child);
        results.append(childResults);
    }
    
    return results;
}

QList<QWidget*> WidgetSelector::filterByConstraints(const QList<QWidget*>& widgets) const {
    QList<QWidget*> filtered;
    
    for (QWidget* widget : widgets) {
        if (matchesConstraints(widget) && matchesStateRequirements(widget)) {
            filtered.append(widget);
        }
    }
    
    return filtered;
}

void WidgetSelector::parseCssSelector(const QString& cssSelector) {
    // Simplified CSS selector parsing
    // For now, just treat it as an object name selector
    m_type = Type::ObjectName;
    m_value = cssSelector;
}

QRegularExpression WidgetSelector::createWildcardRegex(const QString& pattern) const {
    QString regexPattern = QRegularExpression::escape(pattern);
    regexPattern.replace("\\*", ".*");
    regexPattern.replace("\\?", ".");
    return QRegularExpression("^" + regexPattern + "$");
}

// WidgetSelectorBuilder implementation
WidgetSelectorBuilder::WidgetSelectorBuilder() = default;

WidgetSelectorBuilder& WidgetSelectorBuilder::objectName(const QString& name) {
    m_selector.setType(WidgetSelector::Type::ObjectName).setValue(name);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::text(const QString& text, WidgetSelector::MatchMode mode) {
    m_selector.setType(WidgetSelector::Type::Text).setValue(text).setMatchMode(mode);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::className(const QString& className) {
    m_selector.setType(WidgetSelector::Type::ClassName).setValue(className);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::property(const QString& property, const QVariant& value) {
    m_selector.addConstraint(property, value);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::visible(bool visible) {
    m_selector.requireVisible(visible);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::enabled(bool enabled) {
    m_selector.requireEnabled(enabled);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::first() {
    m_selector.setIndex(0);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::last() {
    m_selector.setIndex(-1); // Special value for last
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::nth(int n) {
    m_selector.setIndex(n);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::index(int index) {
    m_selector.setIndex(index);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::parent(const QString& parentSelector) {
    m_selector.setParent(parentSelector);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::parent(QWidget* parentWidget) {
    m_selector.setParent(parentWidget);
    return *this;
}

WidgetSelectorBuilder& WidgetSelectorBuilder::custom(std::function<bool(QWidget*)> predicate) {
    m_selector.setCustomPredicate(predicate);
    return *this;
}

WidgetSelector WidgetSelectorBuilder::build() const {
    return m_selector;
}