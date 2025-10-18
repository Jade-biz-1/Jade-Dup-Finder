#ifndef SECURITY_SAFETY_TESTING_H
#define SECURITY_SAFETY_TESTING_H

#include <QObject>
#include <QTest>
#include <QTemporaryDir>
#include <QSignalSpy>
#include <QTimer>
#include <QEventLoop>
#include <QCryptographicHash>
#include <QFileSystemWatcher>
#include <QProcess>
#include <QRegularExpression>

#include "file_manager.h"
#include "../src/core/safety_manager.h"

/**
 * @brief Comprehensive security and safety testing framework
 * 
 * This class provides testing for:
 * - File operation safety and backup validation
 * - Input validation and sanitization
 * - Security audit and compliance
 * - Protection rule enforcement
 * - Data integrity verification
 * 
 * Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
 */
class SecuritySafetyTesting : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Security test categories
     */
    enum class SecurityTestCategory {
        FileOperationSafety,
        InputValidation,
        SecurityAudit,
        DataIntegrity,
        AccessControl,
        ProtectionRules
    };

    /**
     * @brief Test result structure
     */
    struct SecurityTestResult {
        QString testName;
        SecurityTestCategory category;
        bool passed;
        QString errorMessage;
        QStringList warnings;
        QDateTime timestamp;
        QVariantMap metrics;
        
        SecurityTestResult() : passed(false), timestamp(QDateTime::currentDateTime()) {}
    };

    /**
     * @brief File operation safety test configuration
     */
    struct SafetyTestConfig {
        bool enableBackupValidation = true;
        bool enableIntegrityChecking = true;
        bool enablePermissionTesting = true;
        bool enableProtectionRuleTesting = true;
        int maxTestFiles = 100;
        qint64 maxTestFileSize = 10 * 1024 * 1024; // 10MB
        QString testDataDirectory;
    };

    /**
     * @brief Input validation test configuration
     */
    struct ValidationTestConfig {
        bool enablePathTraversalTests = true;
        bool enableInjectionTests = true;
        bool enableFileNameValidation = true;
        bool enableConfigValidation = true;
        QStringList maliciousPatterns;
        QStringList validPatterns;
    };

    explicit SecuritySafetyTesting(QObject* parent = nullptr);
    ~SecuritySafetyTesting();

    // Configuration
    void setSafetyTestConfig(const SafetyTestConfig& config);
    void setValidationTestConfig(const ValidationTestConfig& config);
    void setTestEnvironment(const QString& testDirectory);

    // Test execution
    QList<SecurityTestResult> runAllSecurityTests();
    QList<SecurityTestResult> runFileOperationSafetyTests();
    QList<SecurityTestResult> runInputValidationTests();
    QList<SecurityTestResult> runSecurityAuditTests();

    // Individual test methods
    SecurityTestResult testBackupCreationAndValidation();
    SecurityTestResult testFilePermissionHandling();
    SecurityTestResult testDataIntegrityVerification();
    SecurityTestResult testProtectionRuleEnforcement();
    SecurityTestResult testPathTraversalPrevention();
    SecurityTestResult testInjectionAttackPrevention();
    SecurityTestResult testFileNameValidation();
    SecurityTestResult testConfigurationSecurity();
    SecurityTestResult testAuditTrailValidation();
    SecurityTestResult testEncryptionAndSecureStorage();

    // Utility methods
    static QString generateSecureTestData(int size);
    static QString calculateFileChecksum(const QString& filePath);
    static bool validateFileIntegrity(const QString& originalPath, const QString& backupPath);
    static QStringList generateMaliciousInputs();
    static bool isSecureFileName(const QString& fileName);

signals:
    void testStarted(const QString& testName, SecuritySafetyTesting::SecurityTestCategory category);
    void testCompleted(const SecuritySafetyTesting::SecurityTestResult& result);
    void securityViolationDetected(const QString& violation, const QString& context);
    void auditEventGenerated(const QString& event, const QVariantMap& details);

private:
    // Test helper methods
    QString createTestFile(const QString& relativePath, const QByteArray& content);
    QString createTestDirectory(const QString& relativePath);
    bool setupTestEnvironment();
    void cleanupTestEnvironment();
    
    // Safety testing helpers
    bool validateBackupIntegrity(const QString& originalPath, const QString& backupPath);
    bool testFilePermissions(const QString& filePath, QFile::Permissions expectedPermissions);
    bool simulateSystemFailure(const QString& context);
    
    // Input validation helpers
    bool testMaliciousInput(const QString& input, const QString& context);
    bool validateInputSanitization(const QString& input, const QString& sanitized);
    QStringList generatePathTraversalAttacks();
    QStringList generateInjectionAttacks();
    
    // Security audit helpers
    bool validateAuditTrail(const QString& operationId);
    bool checkSecurityCompliance();
    bool validateEncryption(const QString& filePath);
    
    // Member variables
    QTemporaryDir* m_testDir;
    QString m_testPath;
    SafetyManager* m_safetyManager;
    FileManager* m_fileManager;
    QFileSystemWatcher* m_fileWatcher;
    
    SafetyTestConfig m_safetyConfig;
    ValidationTestConfig m_validationConfig;
    
    QList<SecurityTestResult> m_testResults;
    QStringList m_createdFiles;
    QStringList m_createdDirectories;
    
    // Test data
    static const QStringList MALICIOUS_FILENAMES;
    static const QStringList PATH_TRAVERSAL_PATTERNS;
    static const QStringList INJECTION_PATTERNS;
    static const QStringList SECURE_FILENAME_PATTERNS;
};

Q_DECLARE_METATYPE(SecuritySafetyTesting::SecurityTestCategory)
Q_DECLARE_METATYPE(SecuritySafetyTesting::SecurityTestResult)
Q_DECLARE_METATYPE(SecuritySafetyTesting::SafetyTestConfig)
Q_DECLARE_METATYPE(SecuritySafetyTesting::ValidationTestConfig)

#endif // SECURITY_SAFETY_TESTING_H