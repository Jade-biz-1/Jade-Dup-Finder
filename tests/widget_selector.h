#pragma once

#include <QWidget>
#include <QString>
#include <QStringList>
#include <QVariant>
#include <QPoint>
#include <QRect>
#include <QRegularExpression>
#include <functional>

/**
 * @brief Advanced widget selector system for reliable element identification
 * 
 * Provides sophisticated widget selection capabilities including CSS-like selectors,
 * XPath-style navigation, and advanced matching algorithms for robust UI testing.
 */
class WidgetSelector {
public:
    /**
     * @brief Selector types for different matching strategies
     */
    enum class Type {
        ObjectName,         ///< Match by objectName property
        Text,              ///< Match by displayed text content
        ClassName,         ///< Match by widget class name
        Property,          ///< Match by custom property value
        Attribute,         ///< Match by Qt attribute
        Position,          ///< Match by screen position
        Size,              ///< Match by widget size
        Index,             ///< Match by child index
        CSS,               ///< CSS-like selector syntax
        XPath,             ///< XPath-like navigation
        Accessibility,     ///< Match by accessibility properties
        Custom,            ///< Custom predicate function
        Composite          ///< Combination of multiple selectors
    };

    /**
     * @brief Matching modes for text and property comparisons
     */
    enum class MatchMode {
        Exact,             ///< Exact string match
        Contains,          ///< String contains substring
        StartsWith,        ///< String starts with prefix
        EndsWith,          ///< String ends with suffix
        Regex,             ///< Regular expression match
        CaseInsensitive,   ///< Case-insensitive exact match
        Wildcard           ///< Wildcard pattern match (* and ?)
    };

    /**
     * @brief Selector constraint for filtering results
     */
    struct Constraint {
        QString property;               ///< Property name to check
        QVariant value;                ///< Expected value
        MatchMode matchMode = MatchMode::Exact;
        bool required = true;           ///< Whether constraint must be satisfied
    };

    /**
     * @brief Spatial relationship for position-based selection
     */
    enum class SpatialRelation {
        Above,             ///< Widget is above reference
        Below,             ///< Widget is below reference
        LeftOf,            ///< Widget is to the left of reference
        RightOf,           ///< Widget is to the right of reference
        Inside,            ///< Widget is inside reference
        Outside,           ///< Widget is outside reference
        Overlapping,       ///< Widget overlaps with reference
        Adjacent,          ///< Widget is adjacent to reference
        Near               ///< Widget is near reference (within distance)
    };

    /**
     * @brief Spatial constraint for position-based matching
     */
    struct SpatialConstraint {
        SpatialRelation relation;
        QWidget* referenceWidget = nullptr;
        QString referenceSelector;      ///< Alternative to referenceWidget
        int tolerance = 10;             ///< Tolerance in pixels for "near" relation
        QRect boundingBox;             ///< Bounding box for area-based relations
    };

    WidgetSelector();
    explicit WidgetSelector(Type type, const QString& value = "");
    WidgetSelector(const QString& cssSelector);  // CSS-like selector constructor

    // Basic selector configuration
    WidgetSelector& setType(Type type);
    WidgetSelector& setValue(const QString& value);
    WidgetSelector& setMatchMode(MatchMode mode);
    WidgetSelector& setParent(const QString& parentSelector);
    WidgetSelector& setParent(QWidget* parentWidget);

    // Constraint management
    WidgetSelector& addConstraint(const QString& property, const QVariant& value, MatchMode mode = MatchMode::Exact);
    WidgetSelector& addConstraint(const Constraint& constraint);
    WidgetSelector& removeConstraint(const QString& property);
    WidgetSelector& clearConstraints();

    // Spatial constraints
    WidgetSelector& addSpatialConstraint(SpatialRelation relation, QWidget* reference, int tolerance = 10);
    WidgetSelector& addSpatialConstraint(SpatialRelation relation, const QString& referenceSelector, int tolerance = 10);
    WidgetSelector& addSpatialConstraint(const SpatialConstraint& constraint);

    // Visibility and state constraints
    WidgetSelector& requireVisible(bool visible = true);
    WidgetSelector& requireEnabled(bool enabled = true);
    WidgetSelector& requireFocusable(bool focusable = true);
    WidgetSelector& requireMinimumSize(const QSize& minSize);
    WidgetSelector& requireMaximumSize(const QSize& maxSize);

    // Index and ordering
    WidgetSelector& setIndex(int index);
    WidgetSelector& setFirst();
    WidgetSelector& setLast();
    WidgetSelector& setNth(int n);

    // Timeout and retry configuration
    WidgetSelector& setTimeout(int timeoutMs);
    WidgetSelector& setRetryInterval(int intervalMs);
    WidgetSelector& setMaxRetries(int retries);

    // Custom predicate
    WidgetSelector& setCustomPredicate(std::function<bool(QWidget*)> predicate);

    // CSS-like selector methods
    WidgetSelector& descendant(const QString& selector);     // space: "parent child"
    WidgetSelector& child(const QString& selector);          // >: "parent > child"
    WidgetSelector& sibling(const QString& selector);        // ~: "element ~ sibling"
    WidgetSelector& adjacent(const QString& selector);       // +: "element + adjacent"
    WidgetSelector& hasClass(const QString& className);      // .class
    WidgetSelector& hasId(const QString& id);               // #id
    WidgetSelector& hasAttribute(const QString& attr, const QString& value = ""); // [attr=value]
    WidgetSelector& contains(const QString& text);           // :contains(text)
    WidgetSelector& visible();                              // :visible
    WidgetSelector& hidden();                               // :hidden
    WidgetSelector& enabled();                              // :enabled
    WidgetSelector& disabled();                             // :disabled
    WidgetSelector& first();                                // :first
    WidgetSelector& last();                                 // :last
    WidgetSelector& nth(int n);                             // :nth(n)
    WidgetSelector& even();                                 // :even
    WidgetSelector& odd();                                  // :odd

    // Widget finding methods
    QWidget* findWidget(QWidget* root = nullptr) const;
    QList<QWidget*> findAllWidgets(QWidget* root = nullptr) const;
    bool exists(QWidget* root = nullptr) const;
    int count(QWidget* root = nullptr) const;

    // Validation and matching
    bool matches(QWidget* widget) const;
    bool isValid() const;
    QString toString() const;

    // Static factory methods
    static WidgetSelector byObjectName(const QString& name);
    static WidgetSelector byText(const QString& text, MatchMode mode = MatchMode::Exact);
    static WidgetSelector byClassName(const QString& className);
    static WidgetSelector byProperty(const QString& property, const QVariant& value);
    static WidgetSelector byPosition(const QPoint& position, int tolerance = 5);
    static WidgetSelector bySize(const QSize& size, int tolerance = 5);
    static WidgetSelector byIndex(int index, const QString& parentSelector = "");
    static WidgetSelector byCss(const QString& cssSelector);
    static WidgetSelector byXPath(const QString& xpathExpression);
    static WidgetSelector byAccessibility(const QString& accessibleName);
    static WidgetSelector byCustom(std::function<bool(QWidget*)> predicate);

    // Combination methods
    WidgetSelector operator&&(const WidgetSelector& other) const;  // AND combination
    WidgetSelector operator||(const WidgetSelector& other) const;  // OR combination
    WidgetSelector operator!() const;                              // NOT negation

private:
    Type m_type;
    QString m_value;
    MatchMode m_matchMode;
    QString m_parentSelector;
    QWidget* m_parentWidget;
    
    QList<Constraint> m_constraints;
    QList<SpatialConstraint> m_spatialConstraints;
    
    bool m_requireVisible;
    bool m_requireEnabled;
    bool m_requireFocusable;
    QSize m_minimumSize;
    QSize m_maximumSize;
    
    int m_index;
    int m_timeoutMs;
    int m_retryIntervalMs;
    int m_maxRetries;
    
    std::function<bool(QWidget*)> m_customPredicate;
    
    // CSS selector parsing
    struct CssRule {
        QString selector;
        QString combinator;  // " ", ">", "~", "+"
    };
    QList<CssRule> m_cssRules;
    
    // Helper methods
    bool matchesConstraints(QWidget* widget) const;
    bool matchesSpatialConstraints(QWidget* widget) const;
    bool matchesStateRequirements(QWidget* widget) const;
    bool matchesTextContent(QWidget* widget, const QString& expectedText, MatchMode mode) const;
    bool matchesProperty(QWidget* widget, const QString& property, const QVariant& expectedValue, MatchMode mode) const;
    bool matchesSpatialRelation(QWidget* widget, const SpatialConstraint& constraint) const;
    
    QList<QWidget*> findWidgetsRecursive(QWidget* root) const;
    QList<QWidget*> filterByConstraints(const QList<QWidget*>& widgets) const;
    QList<QWidget*> filterByIndex(const QList<QWidget*>& widgets) const;
    
    void parseCssSelector(const QString& cssSelector);
    QList<QWidget*> applyCssRules(QWidget* root) const;
    
    QString escapeRegex(const QString& text) const;
    QRegularExpression createWildcardRegex(const QString& pattern) const;
};

/**
 * @brief Widget selector builder for fluent API
 */
class WidgetSelectorBuilder {
public:
    WidgetSelectorBuilder();
    
    // Basic selectors
    WidgetSelectorBuilder& objectName(const QString& name);
    WidgetSelectorBuilder& text(const QString& text, WidgetSelector::MatchMode mode = WidgetSelector::MatchMode::Exact);
    WidgetSelectorBuilder& className(const QString& className);
    WidgetSelectorBuilder& property(const QString& property, const QVariant& value);
    
    // State selectors
    WidgetSelectorBuilder& visible(bool visible = true);
    WidgetSelectorBuilder& enabled(bool enabled = true);
    WidgetSelectorBuilder& focusable(bool focusable = true);
    
    // Position selectors
    WidgetSelectorBuilder& above(QWidget* reference);
    WidgetSelectorBuilder& below(QWidget* reference);
    WidgetSelectorBuilder& leftOf(QWidget* reference);
    WidgetSelectorBuilder& rightOf(QWidget* reference);
    WidgetSelectorBuilder& near(QWidget* reference, int tolerance = 10);
    
    // Index selectors
    WidgetSelectorBuilder& first();
    WidgetSelectorBuilder& last();
    WidgetSelectorBuilder& nth(int n);
    WidgetSelectorBuilder& index(int index);
    
    // Size constraints
    WidgetSelectorBuilder& minSize(const QSize& size);
    WidgetSelectorBuilder& maxSize(const QSize& size);
    WidgetSelectorBuilder& exactSize(const QSize& size);
    
    // Parent/child relationships
    WidgetSelectorBuilder& parent(const QString& parentSelector);
    WidgetSelectorBuilder& parent(QWidget* parentWidget);
    WidgetSelectorBuilder& child(const QString& childSelector);
    WidgetSelectorBuilder& descendant(const QString& descendantSelector);
    
    // Custom predicate
    WidgetSelectorBuilder& custom(std::function<bool(QWidget*)> predicate);
    
    // Build the selector
    WidgetSelector build() const;
    
    // Implicit conversion to WidgetSelector
    operator WidgetSelector() const { return build(); }

private:
    WidgetSelector m_selector;
};

/**
 * @brief Convenience macros for widget selection
 */
#define SELECT_BY_NAME(name) WidgetSelector::byObjectName(name)
#define SELECT_BY_TEXT(text) WidgetSelector::byText(text)
#define SELECT_BY_CLASS(className) WidgetSelector::byClassName(className)
#define SELECT_BY_CSS(css) WidgetSelector::byCss(css)

#define WIDGET_BUILDER() WidgetSelectorBuilder()

// Fluent API examples:
// auto selector = WIDGET_BUILDER().objectName("myButton").visible().enabled().build();
// auto selector = SELECT_BY_CSS("QPushButton[text='OK']:enabled");
// auto selector = SELECT_BY_NAME("dialog").child("QPushButton").text("Cancel");