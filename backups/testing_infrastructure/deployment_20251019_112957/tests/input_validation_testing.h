#ifndef INPUT_VALIDATION_TESTING_H
#define INPUT_VALIDATION_TESTING_H

#include <QObject>
#include <QTest>
#include <QRegularExpression>
#include <QUrl>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QXmlStreamReader>

/**
 * @brief Specialized input validation and sanitization testing
 * 
 * This class provides comprehensive testing for:
 * - Path traversal attack prevention
 * - Injection attack prevention (SQL, XSS, Command, etc.)
 * - File name validation and sanitization
 * - Configuration file security validation
 * - URL and URI validation
 * - Data format validation (JSON, XML, etc.)
 * 
 * Requirements: 10.3, 10.4
 */
class InputValidationTesting : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Input validation test categories
     */
    enum class ValidationCategory {
        PathTraversal,
        InjectionAttacks,
        FileNameValidation,
        ConfigurationSecurity,
        URLValidation,
        DataFormatValidation,
        EncodingValidation,
        LengthValidation
    };

    /**
     * @brief Validation test result
     */
    struct ValidationResult {
        QString testName;
        ValidationCategory category;
        QString input;
        QString sanitizedOutput;
        bool inputBlocked;
        bool sanitizationCorrect;
        QString errorMessage;
        QVariantMap metrics;
        
        ValidationResult() : inputBlocked(false), sanitizationCorrect(false) {}
    };

    /**
     * @brief Input validation configuration
     */
    struct ValidationConfig {
        bool enablePathTraversalTests = true;
        bool enableInjectionTests = true;
        bool enableFileNameTests = true;
        bool enableConfigTests = true;
        bool enableURLTests = true;
        bool enableDataFormatTests = true;
        bool enableEncodingTests = true;
        bool enableLengthTests = true;
        
        int maxFileNameLength = 255;
        int maxPathLength = 4096;
        int maxConfigValueLength = 1024;
        
        QStringList allowedFileExtensions;
        QStringList blockedFileExtensions;
        QStringList allowedProtocols;
    };

    explicit InputValidationTesting(QObject* parent = nullptr);
    ~InputValidationTesting();

    // Configuration
    void setValidationConfig(const ValidationConfig& config);
    ValidationConfig validationConfig() const;

    // Test execution
    QList<ValidationResult> runAllValidationTests();
    QList<ValidationResult> runPathTraversalTests();
    QList<ValidationResult> runInjectionTests();
    QList<ValidationResult> runFileNameTests();
    QList<ValidationResult> runConfigurationTests();
    QList<ValidationResult> runURLValidationTests();
    QList<ValidationResult> runDataFormatTests();

    // Individual validation methods
    ValidationResult validatePathTraversal(const QString& input);
    ValidationResult validateSQLInjection(const QString& input);
    ValidationResult validateXSSInjection(const QString& input);
    ValidationResult validateCommandInjection(const QString& input);
    ValidationResult validateLDAPInjection(const QString& input);
    ValidationResult validateFileName(const QString& input);
    ValidationResult validateFilePath(const QString& input);
    ValidationResult validateConfigurationValue(const QString& key, const QString& value);
    ValidationResult validateURL(const QString& input);
    ValidationResult validateJSONData(const QString& input);
    ValidationResult validateXMLData(const QString& input);

    // Sanitization methods
    static QString sanitizeFileName(const QString& fileName);
    static QString sanitizeFilePath(const QString& filePath);
    static QString sanitizeConfigValue(const QString& value);
    static QString sanitizeURL(const QString& url);
    static QString sanitizeForHTML(const QString& input);
    static QString sanitizeForSQL(const QString& input);
    static QString sanitizeForShell(const QString& input);

    // Validation utility methods
    static bool isValidFileName(const QString& fileName);
    static bool isValidFilePath(const QString& filePath);
    static bool containsPathTraversal(const QString& input);
    static bool containsInjectionPattern(const QString& input);
    static bool isValidURL(const QString& url);
    static bool isValidJSON(const QString& json);
    static bool isValidXML(const QString& xml);

    // Test data generation
    static QStringList generatePathTraversalPayloads();
    static QStringList generateSQLInjectionPayloads();
    static QStringList generateXSSPayloads();
    static QStringList generateCommandInjectionPayloads();
    static QStringList generateLDAPInjectionPayloads();
    static QStringList generateMaliciousFileNames();
    static QStringList generateMaliciousURLs();
    static QStringList generateMaliciousJSONPayloads();
    static QStringList generateMaliciousXMLPayloads();

signals:
    void validationTestStarted(const QString& testName, InputValidationTesting::ValidationCategory category);
    void validationTestCompleted(const InputValidationTesting::ValidationResult& result);
    void maliciousInputDetected(const QString& input, const QString& category);
    void sanitizationPerformed(const QString& original, const QString& sanitized);

private:
    // Internal validation methods
    bool testPathTraversalPrevention(const QString& input);
    bool testInjectionPrevention(const QString& input, const QString& injectionType);
    bool testFileNameSecurity(const QString& fileName);
    bool testConfigurationSecurity(const QString& key, const QString& value);
    bool testURLSecurity(const QString& url);
    bool testDataFormatSecurity(const QString& data, const QString& format);

    // Sanitization validation
    bool validateSanitization(const QString& original, const QString& sanitized, ValidationCategory category);

    // Pattern matching
    bool matchesPathTraversalPattern(const QString& input);
    bool matchesSQLInjectionPattern(const QString& input);
    bool matchesXSSPattern(const QString& input);
    bool matchesCommandInjectionPattern(const QString& input);
    bool matchesLDAPInjectionPattern(const QString& input);

    // Member variables
    ValidationConfig m_config;
    QList<ValidationResult> m_results;

    // Static patterns
    static const QStringList PATH_TRAVERSAL_PATTERNS;
    static const QStringList SQL_INJECTION_PATTERNS;
    static const QStringList XSS_PATTERNS;
    static const QStringList COMMAND_INJECTION_PATTERNS;
    static const QStringList LDAP_INJECTION_PATTERNS;
    static const QStringList MALICIOUS_FILE_NAMES;
    static const QStringList MALICIOUS_URLS;
    static const QStringList RESERVED_WINDOWS_NAMES;
    static const QStringList DANGEROUS_FILE_EXTENSIONS;
};

Q_DECLARE_METATYPE(InputValidationTesting::ValidationCategory)
Q_DECLARE_METATYPE(InputValidationTesting::ValidationResult)
Q_DECLARE_METATYPE(InputValidationTesting::ValidationConfig)

#endif // INPUT_VALIDATION_TESTING_H