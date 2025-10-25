#include <QtTest/QtTest>
#include <QApplication>
#include <QDir>
#include <QStandardPaths>
#include "final_theme_validator.h"
#include "theme_manager.h"
#include "core/logger.h"

class FinalThemeValidationTest : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    // Final validation tests
    void testPerformFinalValidation();
    void testValidateNoHardcodedStyling();
    void testValidateCompleteThemeCompliance();
    void testValidateAllRequirementsMet();
    
    // Source code scanning tests
    void testScanSourceCodeForHardcodedStyles();
    void testScanFileForHardcodedStyles();
    void testIsFileExemptFromScanning();
    
    // Runtime validation tests
    void testValidateRuntimeCompliance();
    void testScanAllWidgetsForViolations();
    void testValidateThemeSystemIntegrity();
    
    // Documentation generation tests
    void testGenerateComprehensiveDocumentation();
    void testGenerateValidationReport();
    void testGenerateComplianceMatrix();
    void testGeneratePerformanceReport();
    
    // Compliance certification tests
    void testGenerateComplianceCertification();
    void testSaveComplianceCertification();
    
    // Configuration tests
    void testConfiguration();

private:
    FinalThemeValidator* m_validator;
    QString m_testOutputDir;
    ThemeManager* m_themeManager;
};

void FinalThemeValidationTest::initTestCase()
{
    // Initialize logging
    Logger::initialize();
    
    // Create validator
    m_validator = new FinalThemeValidator(this);
    QVERIFY(m_validator != nullptr);
    
    // Get theme manager
    m_themeManager = ThemeManager::instance();
    QVERIFY(m_themeManager != nullptr);
    
    // Create test output directory
    m_testOutputDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/theme_validation_test";
    QDir().mkpath(m_testOutputDir);
    
    LOG_INFO(LogCategories::UI, "=== Starting Final Theme Validation Tests ===");
}

void FinalThemeValidationTest::cleanupTestCase()
{
    // Clean up test output directory
    QDir testDir(m_testOutputDir);
    if (testDir.exists()) {
        testDir.removeRecursively();
    }
    
    delete m_validator;
    
    LOG_INFO(LogCategories::UI, "=== Final Theme Validation Tests Completed ===");
}

void FinalThemeValidationTest::testPerformFinalValidation()
{
    LOG_INFO(LogCategories::UI, "Testing performFinalValidation");
    
    // Connect to progress signals for testing
    bool progressReceived = false;
    bool completedReceived = false;
    
    connect(m_validator, &FinalThemeValidator::validationProgress, 
            [&progressReceived](int percentage, const QString& task) {
                progressReceived = true;
                QVERIFY(percentage >= 0 && percentage <= 100);
                QVERIFY(!task.isEmpty());
            });
    
    connect(m_validator, &FinalThemeValidator::validationCompleted,
            [&completedReceived](bool success, const QString& summary) {
                completedReceived = true;
                QVERIFY(!summary.isEmpty());
            });
    
    // Perform final validation
    bool result = m_validator->performFinalValidation();
    
    // Verify signals were emitted
    QVERIFY(progressReceived);
    QVERIFY(completedReceived);
    
    // Result should be true (assuming no violations in test environment)
    QVERIFY(result);
    
    LOG_INFO(LogCategories::UI, "performFinalValidation test completed");
}

void FinalThemeValidationTest::testValidateNoHardcodedStyling()
{
    LOG_INFO(LogCategories::UI, "Testing validateNoHardcodedStyling");
    
    bool result = m_validator->validateNoHardcodedStyling();
    
    // In a properly implemented system, this should return true
    QVERIFY(result);
    
    LOG_INFO(LogCategories::UI, "validateNoHardcodedStyling test completed");
}

void FinalThemeValidationTest::testValidateCompleteThemeCompliance()
{
    LOG_INFO(LogCategories::UI, "Testing validateCompleteThemeCompliance");
    
    bool result = m_validator->validateCompleteThemeCompliance();
    
    // Should return true if theme system is properly implemented
    QVERIFY(result);
    
    LOG_INFO(LogCategories::UI, "validateCompleteThemeCompliance test completed");
}

void FinalThemeValidationTest::testValidateAllRequirementsMet()
{
    LOG_INFO(LogCategories::UI, "Testing validateAllRequirementsMet");
    
    bool result = m_validator->validateAllRequirementsMet();
    
    // Should return true if all requirements are implemented
    QVERIFY(result);
    
    LOG_INFO(LogCategories::UI, "validateAllRequirementsMet test completed");
}

void FinalThemeValidationTest::testScanSourceCodeForHardcodedStyles()
{
    LOG_INFO(LogCategories::UI, "Testing scanSourceCodeForHardcodedStyles");
    
    // Create a temporary test file with hardcoded styles
    QString testFile = m_testOutputDir + "/test_hardcoded.cpp";
    QFile file(testFile);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
    
    QTextStream out(&file);
    out << "#include <QWidget>\n";
    out << "void test() {\n";
    out << "    widget->setStyleSheet(\"background-color: #FF0000;\");\n"; // Hardcoded color
    out << "    widget->setStyleSheet(\"color: rgb(255, 0, 0);\");\n";      // Hardcoded RGB
    out << "}\n";
    file.close();
    
    // Scan the test directory
    QStringList violations = m_validator->scanSourceCodeForHardcodedStyles(m_testOutputDir);
    
    // Should find violations in our test file
    QVERIFY(!violations.isEmpty());
    
    // Check that violations contain expected patterns
    bool foundHexColor = false;
    bool foundRgbColor = false;
    
    for (const QString& violation : violations) {
        if (violation.contains("#FF0000")) {
            foundHexColor = true;
        }
        if (violation.contains("rgb(255, 0, 0)")) {
            foundRgbColor = true;
        }
    }
    
    QVERIFY(foundHexColor);
    QVERIFY(foundRgbColor);
    
    LOG_INFO(LogCategories::UI, QString("Found %1 violations as expected").arg(violations.size()));
}

void FinalThemeValidationTest::testScanFileForHardcodedStyles()
{
    LOG_INFO(LogCategories::UI, "Testing scanFileForHardcodedStyles");
    
    // Create a test file with hardcoded styles
    QString testFile = m_testOutputDir + "/test_single_file.cpp";
    QFile file(testFile);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
    
    QTextStream out(&file);
    out << "widget->setStyleSheet(\"background: #123456;\");\n";
    file.close();
    
    // Scan the specific file
    QStringList violations = m_validator->scanFileForHardcodedStyles(testFile);
    
    // Should find the hardcoded color
    QVERIFY(!violations.isEmpty());
    QVERIFY(violations.first().contains("#123456"));
    
    LOG_INFO(LogCategories::UI, "scanFileForHardcodedStyles test completed");
}

void FinalThemeValidationTest::testIsFileExemptFromScanning()
{
    LOG_INFO(LogCategories::UI, "Testing isFileExemptFromScanning");
    
    // Test exempt patterns
    QVERIFY(m_validator->isFileExemptFromScanning("test_file.cpp"));      // test_* pattern
    QVERIFY(m_validator->isFileExemptFromScanning("file_test.cpp"));      // *_test.cpp pattern
    QVERIFY(m_validator->isFileExemptFromScanning("build/file.cpp"));     // build/* pattern
    QVERIFY(!m_validator->isFileExemptFromScanning("src/main.cpp"));      // Should not be exempt
    
    // Add custom exempt file
    m_validator->addExemptFile("custom_exempt.cpp");
    QVERIFY(m_validator->isFileExemptFromScanning("custom_exempt.cpp"));
    
    LOG_INFO(LogCategories::UI, "isFileExemptFromScanning test completed");
}

void FinalThemeValidationTest::testValidateRuntimeCompliance()
{
    LOG_INFO(LogCategories::UI, "Testing validateRuntimeCompliance");
    
    bool result = m_validator->validateRuntimeCompliance();
    
    // Should return true if no runtime violations
    QVERIFY(result);
    
    LOG_INFO(LogCategories::UI, "validateRuntimeCompliance test completed");
}

void FinalThemeValidationTest::testScanAllWidgetsForViolations()
{
    LOG_INFO(LogCategories::UI, "Testing scanAllWidgetsForViolations");
    
    // Create a widget with hardcoded styling for testing
    QWidget* testWidget = new QWidget();
    testWidget->setStyleSheet("background-color: #FF0000;"); // Hardcoded color
    testWidget->show();
    
    QStringList violations = m_validator->scanAllWidgetsForViolations();
    
    // Should find the violation in our test widget
    bool foundViolation = false;
    for (const QString& violation : violations) {
        if (violation.contains("#FF0000")) {
            foundViolation = true;
            break;
        }
    }
    
    QVERIFY(foundViolation);
    
    // Clean up
    delete testWidget;
    
    LOG_INFO(LogCategories::UI, "scanAllWidgetsForViolations test completed");
}

void FinalThemeValidationTest::testValidateThemeSystemIntegrity()
{
    LOG_INFO(LogCategories::UI, "Testing validateThemeSystemIntegrity");
    
    bool result = m_validator->validateThemeSystemIntegrity();
    
    // Should return true if theme system is working properly
    QVERIFY(result);
    
    LOG_INFO(LogCategories::UI, "validateThemeSystemIntegrity test completed");
}

void FinalThemeValidationTest::testGenerateComprehensiveDocumentation()
{
    LOG_INFO(LogCategories::UI, "Testing generateComprehensiveDocumentation");
    
    QString outputDir = m_testOutputDir + "/documentation";
    bool result = m_validator->generateComprehensiveDocumentation(outputDir);
    
    QVERIFY(result);
    
    // Verify documentation files were created
    QDir docDir(outputDir);
    QVERIFY(docDir.exists());
    QVERIFY(QFile::exists(outputDir + "/validation_report.json"));
    QVERIFY(QFile::exists(outputDir + "/compliance_matrix.html"));
    QVERIFY(QFile::exists(outputDir + "/performance_report.html"));
    QVERIFY(QFile::exists(outputDir + "/test_results.html"));
    QVERIFY(QFile::exists(outputDir + "/compliance_certification.json"));
    
    LOG_INFO(LogCategories::UI, "generateComprehensiveDocumentation test completed");
}

void FinalThemeValidationTest::testGenerateValidationReport()
{
    LOG_INFO(LogCategories::UI, "Testing generateValidationReport");
    
    QString reportPath = m_testOutputDir + "/validation_report.json";
    bool result = m_validator->generateValidationReport(reportPath);
    
    QVERIFY(result);
    QVERIFY(QFile::exists(reportPath));
    
    // Verify report content
    QFile file(reportPath);
    QVERIFY(file.open(QIODevice::ReadOnly));
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();
    
    QJsonObject report = doc.object();
    QVERIFY(report.contains("timestamp"));
    QVERIFY(report.contains("validator_version"));
    QVERIFY(report.contains("scan_results"));
    QVERIFY(report.contains("requirements"));
    
    LOG_INFO(LogCategories::UI, "generateValidationReport test completed");
}

void FinalThemeValidationTest::testGenerateComplianceMatrix()
{
    LOG_INFO(LogCategories::UI, "Testing generateComplianceMatrix");
    
    QString matrixPath = m_testOutputDir + "/compliance_matrix.html";
    bool result = m_validator->generateComplianceMatrix(matrixPath);
    
    QVERIFY(result);
    QVERIFY(QFile::exists(matrixPath));
    
    // Verify HTML content
    QFile file(matrixPath);
    QVERIFY(file.open(QIODevice::ReadOnly | QIODevice::Text));
    QString content = file.readAll();
    file.close();
    
    QVERIFY(content.contains("<html>"));
    QVERIFY(content.contains("Theme Compliance Matrix"));
    QVERIFY(content.contains("Requirements Compliance"));
    
    LOG_INFO(LogCategories::UI, "generateComplianceMatrix test completed");
}

void FinalThemeValidationTest::testGeneratePerformanceReport()
{
    LOG_INFO(LogCategories::UI, "Testing generatePerformanceReport");
    
    QString reportPath = m_testOutputDir + "/performance_report.html";
    bool result = m_validator->generatePerformanceReport(reportPath);
    
    QVERIFY(result);
    QVERIFY(QFile::exists(reportPath));
    
    // Verify HTML content
    QFile file(reportPath);
    QVERIFY(file.open(QIODevice::ReadOnly | QIODevice::Text));
    QString content = file.readAll();
    file.close();
    
    QVERIFY(content.contains("<html>"));
    QVERIFY(content.contains("Theme Performance Report"));
    QVERIFY(content.contains("Performance Metrics"));
    
    LOG_INFO(LogCategories::UI, "generatePerformanceReport test completed");
}

void FinalThemeValidationTest::testGenerateComplianceCertification()
{
    LOG_INFO(LogCategories::UI, "Testing generateComplianceCertification");
    
    FinalThemeValidator::ComplianceCertification cert = m_validator->generateComplianceCertification();
    
    // Verify certification structure
    QVERIFY(!cert.certificationDate.isNull());
    QVERIFY(!cert.certificationVersion.isEmpty());
    QVERIFY(cert.complianceScore >= 0.0 && cert.complianceScore <= 100.0);
    QVERIFY(!cert.certificationSummary.isEmpty());
    
    LOG_INFO(LogCategories::UI, QString("Generated certification with score: %1%").arg(cert.complianceScore));
}

void FinalThemeValidationTest::testSaveComplianceCertification()
{
    LOG_INFO(LogCategories::UI, "Testing saveComplianceCertification");
    
    FinalThemeValidator::ComplianceCertification cert = m_validator->generateComplianceCertification();
    QString certPath = m_testOutputDir + "/certification.json";
    
    bool result = m_validator->saveComplianceCertification(cert, certPath);
    
    QVERIFY(result);
    QVERIFY(QFile::exists(certPath));
    
    // Verify JSON content
    QFile file(certPath);
    QVERIFY(file.open(QIODevice::ReadOnly));
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();
    
    QJsonObject certJson = doc.object();
    QVERIFY(certJson.contains("is_fully_compliant"));
    QVERIFY(certJson.contains("certification_date"));
    QVERIFY(certJson.contains("compliance_score"));
    QVERIFY(certJson.contains("certification_summary"));
    
    LOG_INFO(LogCategories::UI, "saveComplianceCertification test completed");
}

void FinalThemeValidationTest::testConfiguration()
{
    LOG_INFO(LogCategories::UI, "Testing configuration methods");
    
    // Test setSourceDirectory
    m_validator->setSourceDirectory("/test/source");
    
    // Test addExemptFile
    m_validator->addExemptFile("exempt_file.cpp");
    QVERIFY(m_validator->isFileExemptFromScanning("exempt_file.cpp"));
    
    // Test addExemptPattern
    m_validator->addExemptPattern("custom_*");
    QVERIFY(m_validator->isFileExemptFromScanning("custom_test.cpp"));
    
    // Test setStrictMode
    m_validator->setStrictMode(false);
    m_validator->setStrictMode(true);
    
    LOG_INFO(LogCategories::UI, "Configuration test completed");
}

QTEST_MAIN(FinalThemeValidationTest)
#include "final_theme_validation_test.moc"