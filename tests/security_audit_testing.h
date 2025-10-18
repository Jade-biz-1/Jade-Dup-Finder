#ifndef SECURITY_AUDIT_TESTING_H
#define SECURITY_AUDIT_TESTING_H

#include <QObject>
#include <QTest>
#include <QDateTime>
#include <QJsonObject>
#include <QJsonDocument>
#include <QCryptographicHash>
#include <QProcess>
#include <QFileSystemWatcher>
#include <QTimer>
#include <QNetworkAccessManager>
#include <QNetworkReply>

#include "file_manager.h"
#include "../src/core/safety_manager.h"

/**
 * @brief Security audit and compliance testing framework
 * 
 * This class provides comprehensive testing for:
 * - Automated security scanning integration
 * - Compliance validation for data protection requirements
 * - Audit trail validation for all file operations
 * - Encryption and secure storage validation
 * - Security policy enforcement
 * - Vulnerability assessment
 * 
 * Requirements: 10.1, 10.5
 */
class SecurityAuditTesting : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Security audit categories
     */
    enum class AuditCategory {
        SecurityScanning,
        ComplianceValidation,
        AuditTrailValidation,
        EncryptionValidation,
        PolicyEnforcement,
        VulnerabilityAssessment,
        AccessControl,
        DataProtection
    };

    /**
     * @brief Compliance standards
     */
    enum class ComplianceStandard {
        GDPR,           // General Data Protection Regulation
        HIPAA,          // Health Insurance Portability and Accountability Act
        SOX,            // Sarbanes-Oxley Act
        PCI_DSS,        // Payment Card Industry Data Security Standard
        ISO27001,       // ISO/IEC 27001
        NIST,           // NIST Cybersecurity Framework
        Custom          // Custom compliance requirements
    };

    /**
     * @brief Security audit result
     */
    struct AuditResult {
        QString auditName;
        AuditCategory category;
        ComplianceStandard standard;
        bool passed;
        QString errorMessage;
        QStringList warnings;
        QStringList recommendations;
        QDateTime timestamp;
        QVariantMap metrics;
        QVariantMap evidence;
        
        AuditResult() : category(AuditCategory::SecurityScanning), 
                       standard(ComplianceStandard::Custom), 
                       passed(false), 
                       timestamp(QDateTime::currentDateTime()) {}
    };

    /**
     * @brief Security policy configuration
     */
    struct SecurityPolicy {
        bool requireEncryption = true;
        bool requireAuditLogging = true;
        bool requireAccessControl = true;
        bool requireDataClassification = false;
        bool requireSecureDeletion = true;
        bool requireBackupEncryption = true;
        bool requireIntegrityChecking = true;
        
        int maxPasswordAge = 90;           // days
        int minPasswordLength = 8;
        int maxFailedLogins = 5;
        int sessionTimeout = 30;           // minutes
        int auditRetention = 365;          // days
        
        QStringList allowedCiphers;
        QStringList blockedExtensions;
        QStringList sensitiveDataPatterns;
    };

    /**
     * @brief Audit configuration
     */
    struct AuditConfig {
        bool enableSecurityScanning = true;
        bool enableComplianceChecks = true;
        bool enableAuditTrailValidation = true;
        bool enableEncryptionValidation = true;
        bool enablePolicyEnforcement = true;
        bool enableVulnerabilityAssessment = true;
        
        QList<ComplianceStandard> enabledStandards;
        QString auditLogPath;
        QString reportOutputPath;
        int maxAuditEntries = 10000;
        
        SecurityPolicy securityPolicy;
    };

    /**
     * @brief Vulnerability assessment result
     */
    struct VulnerabilityResult {
        QString vulnerabilityId;
        QString title;
        QString description;
        QString severity;      // Critical, High, Medium, Low
        QString category;
        QStringList affectedComponents;
        QStringList recommendations;
        bool exploitable;
        QDateTime discoveredAt;
        
        VulnerabilityResult() : exploitable(false), discoveredAt(QDateTime::currentDateTime()) {}
    };

    explicit SecurityAuditTesting(QObject* parent = nullptr);
    ~SecurityAuditTesting();

    // Configuration
    void setAuditConfig(const AuditConfig& config);
    AuditConfig auditConfig() const;
    void setSecurityPolicy(const SecurityPolicy& policy);
    SecurityPolicy securityPolicy() const;

    // Main audit execution
    QList<AuditResult> runFullSecurityAudit();
    QList<AuditResult> runComplianceAudit(ComplianceStandard standard);
    QList<AuditResult> runSecurityScanningTests();
    QList<AuditResult> runAuditTrailValidation();
    QList<AuditResult> runEncryptionValidation();
    QList<AuditResult> runPolicyEnforcementTests();
    QList<VulnerabilityResult> runVulnerabilityAssessment();

    // Individual audit methods
    AuditResult auditFileOperationSecurity();
    AuditResult auditDataProtectionCompliance();
    AuditResult auditAccessControlMechanisms();
    AuditResult auditEncryptionImplementation();
    AuditResult auditAuditTrailIntegrity();
    AuditResult auditBackupSecurity();
    AuditResult auditConfigurationSecurity();
    AuditResult auditNetworkSecurity();
    AuditResult auditPasswordPolicy();
    AuditResult auditSessionManagement();

    // Compliance-specific audits
    AuditResult auditGDPRCompliance();
    AuditResult auditHIPAACompliance();
    AuditResult auditSOXCompliance();
    AuditResult auditPCIDSSCompliance();
    AuditResult auditISO27001Compliance();
    AuditResult auditNISTCompliance();

    // Security scanning integration
    bool integrateSecurityScanner(const QString& scannerPath);
    QList<VulnerabilityResult> runExternalSecurityScan();
    bool validateSecurityScanResults(const QList<VulnerabilityResult>& results);

    // Audit trail methods
    bool validateAuditTrailCompleteness();
    bool validateAuditTrailIntegrity();
    bool validateAuditTrailRetention();
    QList<QJsonObject> extractAuditEvents(const QDateTime& startTime, const QDateTime& endTime);

    // Encryption validation
    bool validateEncryptionStrength();
    bool validateKeyManagement();
    bool validateDataAtRestEncryption();
    bool validateDataInTransitEncryption();

    // Report generation
    QString generateSecurityAuditReport(const QList<AuditResult>& results);
    QString generateComplianceReport(ComplianceStandard standard, const QList<AuditResult>& results);
    QString generateVulnerabilityReport(const QList<VulnerabilityResult>& vulnerabilities);
    bool exportAuditResults(const QString& filePath, const QList<AuditResult>& results);

    // Utility methods
    static QString calculateAuditHash(const QJsonObject& auditEntry);
    static bool verifyAuditSignature(const QJsonObject& auditEntry, const QString& signature);
    static QString classifyDataSensitivity(const QString& filePath);
    static QStringList extractSensitiveDataPatterns(const QString& content);

signals:
    void auditStarted(const QString& auditName, SecurityAuditTesting::AuditCategory category);
    void auditCompleted(const SecurityAuditTesting::AuditResult& result);
    void vulnerabilityDiscovered(const SecurityAuditTesting::VulnerabilityResult& vulnerability);
    void complianceViolationDetected(SecurityAuditTesting::ComplianceStandard standard, const QString& violation);
    void securityPolicyViolation(const QString& policy, const QString& violation);
    void auditTrailEvent(const QJsonObject& event);

private slots:
    void onFileSystemChanged(const QString& path);
    void onNetworkReplyFinished();
    void onSecurityScanCompleted();

private:
    // Internal audit methods
    bool performSecurityScan(const QString& targetPath);
    bool validateComplianceRequirement(ComplianceStandard standard, const QString& requirement);
    bool checkPolicyCompliance(const QString& policyName);
    bool assessVulnerability(const QString& component, const QString& vulnerabilityType);
    
    // Audit trail helpers
    QJsonObject createAuditEntry(const QString& eventType, const QVariantMap& details);
    bool storeAuditEntry(const QJsonObject& entry);
    bool validateAuditEntry(const QJsonObject& entry);
    QString generateAuditSignature(const QJsonObject& entry);
    
    // Encryption helpers
    bool testEncryptionAlgorithm(const QString& algorithm);
    bool validateCertificate(const QString& certificatePath);
    bool checkKeyStrength(const QString& keyPath);
    
    // Compliance helpers
    bool checkGDPRRequirement(const QString& requirement);
    bool checkHIPAARequirement(const QString& requirement);
    bool checkSOXRequirement(const QString& requirement);
    bool checkPCIDSSRequirement(const QString& requirement);
    bool checkISO27001Requirement(const QString& requirement);
    bool checkNISTRequirement(const QString& requirement);
    
    // Vulnerability assessment helpers
    VulnerabilityResult assessFilePermissionVulnerability();
    VulnerabilityResult assessConfigurationVulnerability();
    VulnerabilityResult assessNetworkVulnerability();
    VulnerabilityResult assessCryptographicVulnerability();
    VulnerabilityResult assessInputValidationVulnerability();
    
    // Security policy enforcement
    bool enforcePasswordPolicy();
    bool enforceAccessControlPolicy();
    bool enforceEncryptionPolicy();
    bool enforceAuditPolicy();
    
    // Member variables
    AuditConfig m_config;
    SecurityPolicy m_policy;
    SafetyManager* m_safetyManager;
    FileManager* m_fileManager;
    QFileSystemWatcher* m_fileWatcher;
    QNetworkAccessManager* m_networkManager;
    QProcess* m_securityScanner;
    
    QList<AuditResult> m_auditResults;
    QList<VulnerabilityResult> m_vulnerabilities;
    QList<QJsonObject> m_auditTrail;
    
    QString m_auditLogPath;
    QString m_reportOutputPath;
    QString m_securityScannerPath;
    
    // Static compliance requirements
    static const QStringList GDPR_REQUIREMENTS;
    static const QStringList HIPAA_REQUIREMENTS;
    static const QStringList SOX_REQUIREMENTS;
    static const QStringList PCI_DSS_REQUIREMENTS;
    static const QStringList ISO27001_REQUIREMENTS;
    static const QStringList NIST_REQUIREMENTS;
    
    // Static security patterns
    static const QStringList SENSITIVE_DATA_PATTERNS;
    static const QStringList WEAK_CIPHER_PATTERNS;
    static const QStringList VULNERABILITY_PATTERNS;
};

Q_DECLARE_METATYPE(SecurityAuditTesting::AuditCategory)
Q_DECLARE_METATYPE(SecurityAuditTesting::ComplianceStandard)
Q_DECLARE_METATYPE(SecurityAuditTesting::AuditResult)
Q_DECLARE_METATYPE(SecurityAuditTesting::SecurityPolicy)
Q_DECLARE_METATYPE(SecurityAuditTesting::AuditConfig)
Q_DECLARE_METATYPE(SecurityAuditTesting::VulnerabilityResult)

#endif // SECURITY_AUDIT_TESTING_H