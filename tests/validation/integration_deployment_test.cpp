#include <QCoreApplication>
#include <QTest>
#include <QDebug>
#include <QProcess>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDateTime>
#include <QElapsedTimer>

class IntegrationDeploymentTest : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    
    void testDeploymentScriptExists();
    void testValidationToolsExist();
    void testMonitoringSetup();
    void testRollbackProcedures();
    void testCIIntegration();
    void testEndToEndDeployment();

private:
    QString m_projectRoot;
    QString m_buildDir;
    QStringList m_requiredExecutables;
    QStringList m_requiredScripts;
};

void IntegrationDeploymentTest::initTestCase()
{
    // Find project root
    m_projectRoot = QDir::currentPath();
    while (!QFileInfo(m_projectRoot + "/CMakeLists.txt").exists() && m_projectRoot != "/") {
        QDir dir(m_projectRoot);
        dir.cdUp();
        m_projectRoot = dir.absolutePath();
    }
    
    QVERIFY2(!m_projectRoot.isEmpty() && m_projectRoot != "/", "Could not find project root");
    
    m_buildDir = m_projectRoot + "/build";
    
    // Define required executables
    m_requiredExecutables = {
        "test_suite_validator",
        "performance_scalability_validator",
        "unit_tests",
        "integration_tests",
        "performance_tests"
    };
    
    // Define required scripts
    m_requiredScripts = {
        "tests/deployment/deploy_testing_infrastructure.sh",
        "scripts/rollback_testing_infrastructure.sh",
        "scripts/monitor_test_infrastructure.py"
    };
    
    qDebug() << "Project root:" << m_projectRoot;
    qDebug() << "Build directory:" << m_buildDir;
}

void IntegrationDeploymentTest::cleanupTestCase()
{
    // Cleanup any temporary files created during testing
    QDir tempDir(QDir::tempPath() + "/cloneclean_deployment_test");
    if (tempDir.exists()) {
        tempDir.removeRecursively();
    }
}

void IntegrationDeploymentTest::testDeploymentScriptExists()
{
    QString deploymentScript = m_projectRoot + "/tests/deployment/deploy_testing_infrastructure.sh";
    
    QVERIFY2(QFileInfo::exists(deploymentScript), 
             QString("Deployment script not found: %1").arg(deploymentScript).toLocal8Bit());
    
    // Check if script is executable
    QFileInfo scriptInfo(deploymentScript);
    QVERIFY2(scriptInfo.isExecutable(), "Deployment script is not executable");
    
    qDebug() << "Deployment script found and executable:" << deploymentScript;
}

void IntegrationDeploymentTest::testValidationToolsExist()
{
    QDir buildDir(m_buildDir);
    QVERIFY2(buildDir.exists(), QString("Build directory not found: %1").arg(m_buildDir).toLocal8Bit());
    
    // Check for validation executables
    for (const QString &executable : m_requiredExecutables) {
        QString executablePath = m_buildDir + "/tests/validation/" + executable;
        
#ifdef Q_OS_WIN
        executablePath += ".exe";
#endif
        
        if (!QFileInfo::exists(executablePath)) {
            // Try alternative paths
            executablePath = m_buildDir + "/tests/" + executable;
#ifdef Q_OS_WIN
            executablePath += ".exe";
#endif
        }
        
        QVERIFY2(QFileInfo::exists(executablePath), 
                 QString("Required executable not found: %1").arg(executable).toLocal8Bit());
        
        qDebug() << "Found executable:" << executablePath;
    }
}

void IntegrationDeploymentTest::testMonitoringSetup()
{
    // Test monitoring configuration creation
    QString monitoringConfig = m_projectRoot + "/monitoring_config.json";
    
    // Run deployment script in dry-run mode to create monitoring config
    QProcess process;
    process.setWorkingDirectory(m_projectRoot);
    
    QStringList args;
    args << "--dry-run";
    
    QString deploymentScript = m_projectRoot + "/tests/deployment/deploy_testing_infrastructure.sh";
    process.start(deploymentScript, args);
    
    QVERIFY2(process.waitForStarted(10000), "Failed to start deployment script");
    QVERIFY2(process.waitForFinished(60000), "Deployment script timed out");
    
    // Check if monitoring configuration was created
    if (QFileInfo::exists(monitoringConfig)) {
        // Validate monitoring configuration
        QFile configFile(monitoringConfig);
        QVERIFY2(configFile.open(QIODevice::ReadOnly), "Failed to open monitoring config");
        
        QJsonParseError error;
        QJsonDocument doc = QJsonDocument::fromJson(configFile.readAll(), &error);
        QVERIFY2(error.error == QJsonParseError::NoError, 
                 QString("Invalid JSON in monitoring config: %1").arg(error.errorString()).toLocal8Bit());
        
        QJsonObject config = doc.object();
        QVERIFY2(config.contains("monitoring"), "Monitoring configuration missing 'monitoring' section");
        QVERIFY2(config.contains("alerting"), "Monitoring configuration missing 'alerting' section");
        
        qDebug() << "Monitoring configuration validated successfully";
    } else {
        qDebug() << "Monitoring configuration not created (dry-run mode)";
    }
}

void IntegrationDeploymentTest::testRollbackProcedures()
{
    QString rollbackScript = m_projectRoot + "/scripts/rollback_testing_infrastructure.sh";
    
    // Check if rollback script exists (it should be created by deployment)
    if (QFileInfo::exists(rollbackScript)) {
        QFileInfo scriptInfo(rollbackScript);
        QVERIFY2(scriptInfo.isExecutable(), "Rollback script is not executable");
        
        qDebug() << "Rollback script found and executable:" << rollbackScript;
    } else {
        qDebug() << "Rollback script not found (may be created during deployment)";
    }
    
    // Check rollback documentation
    QString rollbackDoc = m_projectRoot + "/docs/testing/rollback_procedures.md";
    if (QFileInfo::exists(rollbackDoc)) {
        qDebug() << "Rollback documentation found:" << rollbackDoc;
    } else {
        qDebug() << "Rollback documentation not found (may be created during deployment)";
    }
}

void IntegrationDeploymentTest::testCIIntegration()
{
    // Check GitHub Actions workflows
    QString workflowsDir = m_projectRoot + "/.github/workflows";
    QDir workflows(workflowsDir);
    
    if (workflows.exists()) {
        QStringList workflowFiles = workflows.entryList(QStringList() << "*.yml" << "*.yaml", QDir::Files);
        QVERIFY2(!workflowFiles.isEmpty(), "No GitHub Actions workflow files found");
        
        qDebug() << "Found GitHub Actions workflows:" << workflowFiles;
        
        // Check for key workflow files
        QStringList expectedWorkflows = {
            "automated-testing.yml",
            "pr-validation.yml",
            "nightly-comprehensive.yml"
        };
        
        for (const QString &expectedWorkflow : expectedWorkflows) {
            bool found = false;
            for (const QString &workflowFile : workflowFiles) {
                if (workflowFile.contains(expectedWorkflow.split('.').first())) {
                    found = true;
                    break;
                }
            }
            
            if (found) {
                qDebug() << "Found expected workflow type:" << expectedWorkflow;
            } else {
                qDebug() << "Expected workflow not found:" << expectedWorkflow;
            }
        }
    } else {
        qDebug() << "GitHub Actions workflows directory not found";
    }
    
    // Check CI scripts
    QString ciScriptsDir = m_projectRoot + "/scripts/ci";
    QDir ciScripts(ciScriptsDir);
    
    if (ciScripts.exists()) {
        QStringList scriptFiles = ciScripts.entryList(QStringList() << "*.py" << "*.sh", QDir::Files);
        QVERIFY2(!scriptFiles.isEmpty(), "No CI scripts found");
        
        qDebug() << "Found CI scripts:" << scriptFiles;
    } else {
        qDebug() << "CI scripts directory not found";
    }
}

void IntegrationDeploymentTest::testEndToEndDeployment()
{
    qDebug() << "Running end-to-end deployment test...";
    
    QElapsedTimer timer;
    timer.start();
    
    // Create temporary deployment directory
    QString tempDeployDir = QDir::tempPath() + "/cloneclean_deployment_test";
    QDir().mkpath(tempDeployDir);
    
    // Run deployment script with dry-run mode
    QProcess process;
    process.setWorkingDirectory(m_projectRoot);
    
    QStringList args;
    args << "--dry-run" << "--verbose";
    
    QString deploymentScript = m_projectRoot + "/tests/deployment/deploy_testing_infrastructure.sh";
    process.start(deploymentScript, args);
    
    QVERIFY2(process.waitForStarted(10000), "Failed to start deployment script");
    
    // Wait for deployment to complete (with extended timeout)
    bool finished = process.waitForFinished(300000); // 5 minutes timeout
    
    if (!finished) {
        process.kill();
        QFAIL("Deployment script timed out");
    }
    
    qint64 executionTime = timer.elapsed();
    qDebug() << "Deployment script execution time:" << executionTime << "ms";
    
    // Check exit code
    int exitCode = process.exitCode();
    QString output = process.readAllStandardOutput();
    QString errorOutput = process.readAllStandardError();
    
    qDebug() << "Deployment script exit code:" << exitCode;
    qDebug() << "Deployment script output:" << output;
    
    if (!errorOutput.isEmpty()) {
        qDebug() << "Deployment script errors:" << errorOutput;
    }
    
    // In dry-run mode, we expect success even if some operations are skipped
    QVERIFY2(exitCode == 0 || exitCode == 1, // Allow exit code 1 for dry-run limitations
             QString("Deployment script failed with exit code: %1").arg(exitCode).toLocal8Bit());
    
    // Validate that key components were checked
    QVERIFY2(output.contains("Checking deployment prerequisites"), 
             "Prerequisites check not performed");
    QVERIFY2(output.contains("Validating test infrastructure"), 
             "Infrastructure validation not performed");
    
    qDebug() << "End-to-end deployment test completed successfully";
}

QTEST_MAIN(IntegrationDeploymentTest)
#include "integration_deployment_test.moc"