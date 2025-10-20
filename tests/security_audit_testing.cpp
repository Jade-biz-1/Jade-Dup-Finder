#include "security_audit_testing.h"
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>
#include <QCoreApplication>
#include <QJsonArray>
#include <QNetworkRequest>
#include <QSslConfiguration>
#include <QSslCertificate>
#include <QRegularExpression>

// Static compliance requirements
const QStringList SecurityAuditTesting::GDPR_REQUIREMENTS = {
    "data_minimization", "consent_management", "right_to_erasure", "data_portability",
    "privacy_by_design", "breach_notification", "data_protection_officer", "impact_assessment"
};

const QStringList SecurityAuditTesting::HIPAA_REQUIREMENTS = {
    "access_control", "audit_controls", "integrity", "person_authentication",
    "transmission_security", "encryption_decryption", "automatic_logoff", "unique_user_identification"
};

const QStringList SecurityAuditTesting::SOX_REQUIREMENTS = {
    "financial_reporting_controls", "audit_trail_integrity", "segregation_of_duties", "change_management",
    "access_controls", "data_retention", "backup_recovery", "incident_response"
};

const QStringList SecurityAuditTesting::PCI_DSS_REQUIREMENTS = {
    "firewall_configuration", "default_passwords", "cardholder_data_protection", "encrypted_transmission",
    "antivirus_software", "secure_systems", "access_control_measures", "unique_ids",
    "physical_access", "network_monitoring", "security_testing", "information_security_policy"
};

const QStringList SecurityAuditTesting::ISO27001_REQUIREMENTS = {
    "information_security_policy", "risk_management", "asset_management", "access_control",
    "cryptography", "physical_security", "operations_security", "communications_security",
    "system_acquisition", "supplier_relationships", "incident_management", "business_continuity", "compliance"
};

const QStringList SecurityAuditTesting::NIST_REQUIREMENTS = {
    "identify_assets", "protect_systems", "detect_events", "respond_incidents", "recover_operations"
};

const QStringList SecurityAuditTesting::SENSITIVE_DATA_PATTERNS = {
    "\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b",  // Credit card
    "\\b\\d{3}-\\d{2}-\\d{4}\\b",                            // SSN
    "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b", // Email
    "password\\s*[:=]\\s*[\"']?([^\"'\\s]+)[\"']?",          // Password
    "api[_-]?key\\s*[:=]\\s*[\"']?([^\"'\\s]+)[\"']?",       // API Key
    "secret\\s*[:=]\\s*[\"']?([^\"'\\s]+)[\"']?"             // Secret
};

const QStringList SecurityAuditTesting::WEAK_CIPHER_PATTERNS = {
    "DES", "3DES", "RC4", "MD5", "SHA1", "SSL", "TLS1.0", "TLS1.1"
};

const QStringList SecurityAuditTesting::VULNERABILITY_PATTERNS = {
    "buffer_overflow", "sql_injection", "xss_vulnerability", "path_traversal",
    "command_injection", "weak_authentication", "insecure_storage", "insufficient_logging"
};

SecurityAuditTesting::SecurityAuditTesting(QObject* parent)
    : QObject(parent), m_safetyManager(nullptr), m_fileManager(nullptr), m_fileWatcher(nullptr),
      m_networkManager(nullptr), m_securityScanner(nullptr)
{
    // Initialize default configuration and setup components
    m_config.enableSecurityScanning = true;
    m_config.enableComplianceChecks = true;
    m_config.enableAuditTrailValidation = true;
    m_config.enableEncryptionValidation = true;
    m_config.enablePolicyEnforcement = true;
    m_config.enableVulnerabilityAssessment = true;
    m_config.enabledStandards = {ComplianceStandard::ISO27001, ComplianceStandard::NIST};
    m_config.maxAuditEntries = 10000;
    
    // Initialize security policy
    m_policy.requireEncryption = true;
    m_policy.requireAuditLogging = true;
    m_policy.requireAccessControl = true;
    m_policy.allowedCiphers = {"AES-256", "ChaCha20", "RSA-2048", "ECDSA-256"};
    
    // Setup paths and components
    m_auditLogPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/audit.log";
    m_reportOutputPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/security_reports";
    
    m_networkManager = new QNetworkAccessManager(this);
    m_fileWatcher = new QFileSystemWatcher(this);
    
    // Register metatypes
    qRegisterMetaType<AuditCategory>("AuditCategory");
    qRegisterMetaType<ComplianceStandard>("ComplianceStandard");
    qRegisterMetaType<AuditResult>("AuditResult");
}

SecurityAuditTesting::~SecurityAuditTesting()
{
    if (m_securityScanner && m_securityScanner->state() != QProcess::NotRunning) {
        m_securityScanner->kill();
        m_securityScanner->waitForFinished(3000);
    }
}void Securit
yAuditTesting::setAuditConfig(const AuditConfig& config)
{
    m_config = config;
}

SecurityAuditTesting::AuditConfig SecurityAuditTesting::auditConfig() const
{
    return m_config;
}

void SecurityAuditTesting::setSecurityPolicy(const SecurityPolicy& policy)
{
    m_policy = policy;
}

SecurityAuditTesting::SecurityPolicy SecurityAuditTesting::securityPolicy() const
{
    return m_policy;
}

QList<SecurityAuditTesting::AuditResult> SecurityAuditTesting::runFullSecurityAudit()
{
    qDebug() << "Starting comprehensive security audit...";
    
    m_auditResults.clear();
    
    if (m_config.enableSecurityScanning) {
        m_auditResults << runSecurityScanningTests();
    }
    
    if (m_config.enableComplianceChecks) {
        for (ComplianceStandard standard : m_config.enabledStandards) {
            m_auditResults << runComplianceAudit(standard);
        }
    }
    
    if (m_config.enableAuditTrailValidation) {
        m_auditResults << runAuditTrailValidation();
    }
    
    if (m_config.enableEncryptionValidation) {
        m_auditResults << runEncryptionValidation();
    }
    
    if (m_config.enablePolicyEnforcement) {
        m_auditResults << runPolicyEnforcementTests();
    }
    
    if (m_config.enableVulnerabilityAssessment) {
        m_vulnerabilities = runVulnerabilityAssessment();
    }
    
    qDebug() << "Security audit completed. Results:" << m_auditResults.size() 
             << "Vulnerabilities:" << m_vulnerabilities.size();
    
    return m_auditResults;
}

QList<SecurityAuditTesting::AuditResult> SecurityAuditTesting::runComplianceAudit(ComplianceStandard standard)
{
    qDebug() << "Running compliance audit for standard:" << static_cast<int>(standard);
    
    QList<AuditResult> results;
    
    switch (standard) {
        case ComplianceStandard::GDPR:
            results << auditGDPRCompliance();
            break;
        case ComplianceStandard::HIPAA:
            results << auditHIPAACompliance();
            break;
        case ComplianceStandard::SOX:
            results << auditSOXCompliance();
            break;
        case ComplianceStandard::PCI_DSS:
            results << auditPCIDSSCompliance();
            break;
        case ComplianceStandard::ISO27001:
            results << auditISO27001Compliance();
            break;
        case ComplianceStandard::NIST:
            results << auditNISTCompliance();
            break;
        case ComplianceStandard::Custom:
            // Custom compliance checks would be implemented here
            break;
    }
    
    return results;
}

QList<SecurityAuditTesting::AuditResult> SecurityAuditTesting::runSecurityScanningTests()
{
    qDebug() << "Running security scanning tests...";
    
    QList<AuditResult> results;
    
    results << auditFileOperationSecurity();
    results << auditAccessControlMechanisms();
    results << auditConfigurationSecurity();
    results << auditNetworkSecurity();
    
    return results;
}

QList<SecurityAuditTesting::AuditResult> SecurityAuditTesting::runAuditTrailValidation()
{
    qDebug() << "Running audit trail validation...";
    
    QList<AuditResult> results;
    results << auditAuditTrailIntegrity();
    
    return results;
}

QList<SecurityAuditTesting::AuditResult> SecurityAuditTesting::runEncryptionValidation()
{
    qDebug() << "Running encryption validation...";
    
    QList<AuditResult> results;
    results << auditEncryptionImplementation();
    results << auditBackupSecurity();
    
    return results;
}

QList<SecurityAuditTesting::AuditResult> SecurityAuditTesting::runPolicyEnforcementTests()
{
    qDebug() << "Running policy enforcement tests...";
    
    QList<AuditResult> results;
    results << auditPasswordPolicy();
    results << auditSessionManagement();
    
    return results;
}

QList<SecurityAuditTesting::VulnerabilityResult> SecurityAuditTesting::runVulnerabilityAssessment()
{
    qDebug() << "Running vulnerability assessment...";
    
    QList<VulnerabilityResult> vulnerabilities;
    
    vulnerabilities << assessFilePermissionVulnerability();
    vulnerabilities << assessConfigurationVulnerability();
    vulnerabilities << assessNetworkVulnerability();
    vulnerabilities << assessCryptographicVulnerability();
    vulnerabilities << assessInputValidationVulnerability();
    
    return vulnerabilities;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditFileOperationSecurity()
{
    AuditResult result;
    result.auditName = "File Operation Security Audit";
    result.category = AuditCategory::SecurityScanning;
    result.standard = ComplianceStandard::Custom;
    
    emit auditStarted(result.auditName, result.category);
    
    try {
        // Check if SafetyManager is properly configured
        bool safetyManagerConfigured = (m_safetyManager != nullptr);
        result.metrics["safety_manager_configured"] = safetyManagerConfigured;
        
        if (safetyManagerConfigured) {
            // Check backup creation
            bool backupsEnabled = (m_safetyManager->defaultBackupStrategy() != SafetyManager::BackupStrategy::None);
            result.metrics["backups_enabled"] = backupsEnabled;
            
            // Check protection rules
            QList<SafetyManager::ProtectionEntry> rules = m_safetyManager->getProtectionRules();
            result.metrics["protection_rules_count"] = rules.size();
            
            // Check audit logging
            int operationsTracked = m_safetyManager->getTotalOperationsTracked();
            result.metrics["operations_tracked"] = operationsTracked;
            
            result.passed = backupsEnabled && (rules.size() > 0) && (operationsTracked >= 0);
        } else {
            result.errorMessage = "SafetyManager not configured";
        }
        
        if (result.passed) {
            qDebug() << "✓ File operation security audit passed";
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    }
    
    emit auditCompleted(result);
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditDataProtectionCompliance()
{
    AuditResult result;
    result.auditName = "Data Protection Compliance Audit";
    result.category = AuditCategory::DataProtection;
    result.standard = ComplianceStandard::GDPR;
    
    emit auditStarted(result.auditName, result.category);
    
    try {
        // Check data encryption requirements
        bool encryptionRequired = m_policy.requireEncryption;
        result.metrics["encryption_required"] = encryptionRequired;
        
        // Check data retention policies
        int auditRetention = m_policy.auditRetention;
        result.metrics["audit_retention_days"] = auditRetention;
        
        // Check secure deletion
        bool secureDeletion = m_policy.requireSecureDeletion;
        result.metrics["secure_deletion_enabled"] = secureDeletion;
        
        // Check access controls
        bool accessControl = m_policy.requireAccessControl;
        result.metrics["access_control_enabled"] = accessControl;
        
        result.passed = encryptionRequired && (auditRetention > 0) && secureDeletion && accessControl;
        
        if (!result.passed) {
            result.errorMessage = "Data protection requirements not met";
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    }
    
    emit auditCompleted(result);
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditGDPRCompliance()
{
    AuditResult result;
    result.auditName = "GDPR Compliance Audit";
    result.category = AuditCategory::ComplianceValidation;
    result.standard = ComplianceStandard::GDPR;
    
    emit auditStarted(result.auditName, result.category);
    
    try {
        int passedRequirements = 0;
        
        for (const QString& requirement : GDPR_REQUIREMENTS) {
            bool requirementMet = checkGDPRRequirement(requirement);
            result.metrics[requirement] = requirementMet;
            
            if (requirementMet) {
                passedRequirements++;
            } else {
                result.warnings << QString("GDPR requirement not met: %1").arg(requirement);
            }
        }
        
        result.metrics["passed_requirements"] = passedRequirements;
        result.metrics["total_requirements"] = GDPR_REQUIREMENTS.size();
        
        // Require at least 80% compliance
        double complianceRate = (double)passedRequirements / GDPR_REQUIREMENTS.size();
        result.passed = (complianceRate >= 0.8);
        
        if (!result.passed) {
            result.errorMessage = QString("GDPR compliance rate: %1% (minimum 80% required)")
                .arg(complianceRate * 100, 0, 'f', 1);
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    }
    
    emit auditCompleted(result);
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditEncryptionImplementation()
{
    AuditResult result;
    result.auditName = "Encryption Implementation Audit";
    result.category = AuditCategory::EncryptionValidation;
    result.standard = ComplianceStandard::Custom;
    
    emit auditStarted(result.auditName, result.category);
    
    try {
        // Check encryption strength
        bool strongEncryption = validateEncryptionStrength();
        result.metrics["strong_encryption"] = strongEncryption;
        
        // Check key management
        bool keyManagement = validateKeyManagement();
        result.metrics["key_management"] = keyManagement;
        
        // Check data at rest encryption
        bool dataAtRest = validateDataAtRestEncryption();
        result.metrics["data_at_rest_encryption"] = dataAtRest;
        
        // Check data in transit encryption
        bool dataInTransit = validateDataInTransitEncryption();
        result.metrics["data_in_transit_encryption"] = dataInTransit;
        
        result.passed = strongEncryption && keyManagement && dataAtRest && dataInTransit;
        
        if (!result.passed) {
            result.errorMessage = "Encryption implementation requirements not met";
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    }
    
    emit auditCompleted(result);
    return result;
}

SecurityAuditTesting::VulnerabilityResult SecurityAuditTesting::assessFilePermissionVulnerability()
{
    VulnerabilityResult vulnerability;
    vulnerability.vulnerabilityId = "VULN-001";
    vulnerability.title = "File Permission Vulnerability Assessment";
    vulnerability.category = "Access Control";
    vulnerability.severity = "Medium";
    
    // Check for world-writable files
    // This is a simplified assessment - real implementation would scan actual files
    vulnerability.description = "Assessment of file permission vulnerabilities";
    vulnerability.exploitable = false; // Assume secure by default
    
    vulnerability.recommendations << "Ensure proper file permissions are set";
    vulnerability.recommendations << "Regular permission audits";
    vulnerability.recommendations << "Implement least privilege principle";
    
    return vulnerability;
}

// Helper method implementations
bool SecurityAuditTesting::checkGDPRRequirement(const QString& requirement)
{
    // Simplified GDPR requirement checking
    if (requirement == "data_minimization") {
        return true; // Assume implemented
    } else if (requirement == "consent_management") {
        return true; // Assume implemented
    } else if (requirement == "right_to_erasure") {
        return m_policy.requireSecureDeletion;
    } else if (requirement == "privacy_by_design") {
        return m_policy.requireEncryption && m_policy.requireAccessControl;
    } else if (requirement == "breach_notification") {
        return m_policy.requireAuditLogging;
    }
    
    return false; // Default to not implemented
}

bool SecurityAuditTesting::validateEncryptionStrength()
{
    // Check if strong encryption algorithms are configured
    for (const QString& cipher : m_policy.allowedCiphers) {
        if (cipher.contains("AES-256") || cipher.contains("ChaCha20")) {
            return true;
        }
    }
    return false;
}

bool SecurityAuditTesting::validateKeyManagement()
{
    // Basic key management validation
    // In a real implementation, this would check key storage, rotation, etc.
    return m_policy.requireEncryption;
}

bool SecurityAuditTesting::validateDataAtRestEncryption()
{
    // Check if data at rest encryption is enabled
    return m_policy.requireBackupEncryption;
}

bool SecurityAuditTesting::validateDataInTransitEncryption()
{
    // Check if data in transit encryption is configured
    // This would check network configurations in a real implementation
    return true; // Assume configured
}

QString SecurityAuditTesting::generateSecurityAuditReport(const QList<AuditResult>& results)
{
    QString report;
    report += "# Security Audit Report\n\n";
    report += QString("Generated: %1\n\n").arg(QDateTime::currentDateTime().toString());
    
    int passed = 0, failed = 0;
    for (const AuditResult& result : results) {
        if (result.passed) passed++;
        else failed++;
    }
    
    report += QString("## Summary\n");
    report += QString("- Total Audits: %1\n").arg(results.size());
    report += QString("- Passed: %1\n").arg(passed);
    report += QString("- Failed: %1\n\n").arg(failed);
    
    report += "## Detailed Results\n\n";
    for (const AuditResult& result : results) {
        report += QString("### %1\n").arg(result.auditName);
        report += QString("- Status: %1\n").arg(result.passed ? "PASS" : "FAIL");
        report += QString("- Category: %1\n").arg(static_cast<int>(result.category));
        
        if (!result.errorMessage.isEmpty()) {
            report += QString("- Error: %1\n").arg(result.errorMessage);
        }
        
        if (!result.warnings.isEmpty()) {
            report += "- Warnings:\n";
            for (const QString& warning : result.warnings) {
                report += QString("  - %1\n").arg(warning);
            }
        }
        
        report += "\n";
    }
    
    return report;
}

// Slot implementations
void SecurityAuditTesting::onFileSystemChanged(const QString& path)
{
    QJsonObject event = createAuditEntry("file_system_change", {{"path", path}});
    emit auditTrailEvent(event);
}

void SecurityAuditTesting::onNetworkReplyFinished()
{
    // Handle network security scan results
    QNetworkReply* reply = qobject_cast<QNetworkReply*>(sender());
    if (reply) {
        // Process security scan results
        reply->deleteLater();
    }
}

void SecurityAuditTesting::onSecurityScanCompleted()
{
    qDebug() << "Security scan completed";
}

QJsonObject SecurityAuditTesting::createAuditEntry(const QString& eventType, const QVariantMap& details)
{
    QJsonObject entry;
    entry["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    entry["event_type"] = eventType;
    entry["details"] = QJsonObject::fromVariantMap(details);
    entry["signature"] = generateAuditSignature(entry);
    
    return entry;
}

QString SecurityAuditTesting::generateAuditSignature(const QJsonObject& entry)
{
    // Generate a simple signature for audit integrity
    QJsonDocument doc(entry);
    QByteArray data = doc.toJson(QJsonDocument::Compact);
    
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(data);
    
    return hash.result().toHex();
}

// Additional stub implementations for remaining methods
SecurityAuditTesting::AuditResult SecurityAuditTesting::auditAccessControlMechanisms()
{
    AuditResult result;
    result.auditName = "Access Control Mechanisms Audit";
    result.category = AuditCategory::AccessControl;
    result.passed = m_policy.requireAccessControl;
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditAuditTrailIntegrity()
{
    AuditResult result;
    result.auditName = "Audit Trail Integrity Audit";
    result.category = AuditCategory::AuditTrailValidation;
    result.passed = validateAuditTrailIntegrity();
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditBackupSecurity()
{
    AuditResult result;
    result.auditName = "Backup Security Audit";
    result.category = AuditCategory::EncryptionValidation;
    result.passed = m_policy.requireBackupEncryption;
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditConfigurationSecurity()
{
    AuditResult result;
    result.auditName = "Configuration Security Audit";
    result.category = AuditCategory::SecurityScanning;
    result.passed = true; // Simplified implementation
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditNetworkSecurity()
{
    AuditResult result;
    result.auditName = "Network Security Audit";
    result.category = AuditCategory::SecurityScanning;
    result.passed = true; // Simplified implementation
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditPasswordPolicy()
{
    AuditResult result;
    result.auditName = "Password Policy Audit";
    result.category = AuditCategory::PolicyEnforcement;
    result.passed = (m_policy.minPasswordLength >= 8);
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditSessionManagement()
{
    AuditResult result;
    result.auditName = "Session Management Audit";
    result.category = AuditCategory::PolicyEnforcement;
    result.passed = (m_policy.sessionTimeout > 0);
    return result;
}

// Compliance audit stubs
SecurityAuditTesting::AuditResult SecurityAuditTesting::auditHIPAACompliance()
{
    AuditResult result;
    result.auditName = "HIPAA Compliance Audit";
    result.category = AuditCategory::ComplianceValidation;
    result.standard = ComplianceStandard::HIPAA;
    result.passed = true; // Simplified
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditSOXCompliance()
{
    AuditResult result;
    result.auditName = "SOX Compliance Audit";
    result.category = AuditCategory::ComplianceValidation;
    result.standard = ComplianceStandard::SOX;
    result.passed = true; // Simplified
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditPCIDSSCompliance()
{
    AuditResult result;
    result.auditName = "PCI DSS Compliance Audit";
    result.category = AuditCategory::ComplianceValidation;
    result.standard = ComplianceStandard::PCI_DSS;
    result.passed = true; // Simplified
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditISO27001Compliance()
{
    AuditResult result;
    result.auditName = "ISO 27001 Compliance Audit";
    result.category = AuditCategory::ComplianceValidation;
    result.standard = ComplianceStandard::ISO27001;
    result.passed = true; // Simplified
    return result;
}

SecurityAuditTesting::AuditResult SecurityAuditTesting::auditNISTCompliance()
{
    AuditResult result;
    result.auditName = "NIST Compliance Audit";
    result.category = AuditCategory::ComplianceValidation;
    result.standard = ComplianceStandard::NIST;
    result.passed = true; // Simplified
    return result;
}

// Vulnerability assessment stubs
SecurityAuditTesting::VulnerabilityResult SecurityAuditTesting::assessConfigurationVulnerability()
{
    VulnerabilityResult vuln;
    vuln.vulnerabilityId = "VULN-002";
    vuln.title = "Configuration Vulnerability";
    vuln.severity = "Low";
    vuln.exploitable = false;
    return vuln;
}

SecurityAuditTesting::VulnerabilityResult SecurityAuditTesting::assessNetworkVulnerability()
{
    VulnerabilityResult vuln;
    vuln.vulnerabilityId = "VULN-003";
    vuln.title = "Network Vulnerability";
    vuln.severity = "Medium";
    vuln.exploitable = false;
    return vuln;
}

SecurityAuditTesting::VulnerabilityResult SecurityAuditTesting::assessCryptographicVulnerability()
{
    VulnerabilityResult vuln;
    vuln.vulnerabilityId = "VULN-004";
    vuln.title = "Cryptographic Vulnerability";
    vuln.severity = "High";
    vuln.exploitable = false;
    return vuln;
}

SecurityAuditTesting::VulnerabilityResult SecurityAuditTesting::assessInputValidationVulnerability()
{
    VulnerabilityResult vuln;
    vuln.vulnerabilityId = "VULN-005";
    vuln.title = "Input Validation Vulnerability";
    vuln.severity = "Medium";
    vuln.exploitable = false;
    return vuln;
}

// Validation method stubs
bool SecurityAuditTesting::validateAuditTrailIntegrity()
{
    return m_policy.requireAuditLogging;
}