#include "test_base.h"
#include "ui_automation.h"
#include "widget_selector.h"
#include <QTest>
#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QTextEdit>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QDialog>
#include <QDialogButtonBox>
#include <QTimer>
#include <QDebug>

/**
 * @brief Test application window for UI automation demonstration
 */
class TestApplicationWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit TestApplicationWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setupUI();
        setObjectName("TestApplicationWindow");
        setWindowTitle("UI Automation Test Application");
        resize(800, 600);
    }

private slots:
    void onButtonClicked() {
        m_statusLabel->setText("Button clicked!");
        m_clickCount++;
        m_counterLabel->setText(QString("Clicks: %1").arg(m_clickCount));
    }

    void onTextChanged() {
        QString text = m_lineEdit->text();
        m_echoLabel->setText("Echo: " + text);
    }

    void onComboChanged() {
        QString selection = m_comboBox->currentText();
        m_selectionLabel->setText("Selected: " + selection);
    }

    void onCheckToggled(bool checked) {
        m_checkLabel->setText(checked ? "Checked" : "Unchecked");
    }

    void showDialog() {
        QDialog dialog(this);
        dialog.setObjectName("TestDialog");
        dialog.setWindowTitle("Test Dialog");
        dialog.resize(300, 200);

        QVBoxLayout* layout = new QVBoxLayout(&dialog);
        
        QLabel* label = new QLabel("This is a test dialog", &dialog);
        label->setObjectName("DialogLabel");
        layout->addWidget(label);

        QLineEdit* dialogInput = new QLineEdit(&dialog);
        dialogInput->setObjectName("DialogInput");
        dialogInput->setPlaceholderText("Enter some text...");
        layout->addWidget(dialogInput);

        QDialogButtonBox* buttonBox = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
        buttonBox->setObjectName("DialogButtons");
        layout->addWidget(buttonBox);

        connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
        connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

        if (dialog.exec() == QDialog::Accepted) {
            m_statusLabel->setText("Dialog accepted: " + dialogInput->text());
        } else {
            m_statusLabel->setText("Dialog cancelled");
        }
    }

    void delayedAction() {
        m_statusLabel->setText("Starting delayed action...");
        
        QTimer::singleShot(2000, [this]() {
            m_statusLabel->setText("Delayed action completed!");
            m_delayedButton->setEnabled(true);
        });
        
        m_delayedButton->setEnabled(false);
    }

private:
    void setupUI() {
        QWidget* centralWidget = new QWidget(this);
        setCentralWidget(centralWidget);

        QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);

        // Status area
        m_statusLabel = new QLabel("Ready", this);
        m_statusLabel->setObjectName("StatusLabel");
        m_statusLabel->setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }");
        mainLayout->addWidget(m_statusLabel);

        // Button section
        QHBoxLayout* buttonLayout = new QHBoxLayout();
        
        m_clickButton = new QPushButton("Click Me", this);
        m_clickButton->setObjectName("ClickButton");
        connect(m_clickButton, &QPushButton::clicked, this, &TestApplicationWindow::onButtonClicked);
        buttonLayout->addWidget(m_clickButton);

        m_counterLabel = new QLabel("Clicks: 0", this);
        m_counterLabel->setObjectName("CounterLabel");
        buttonLayout->addWidget(m_counterLabel);

        mainLayout->addLayout(buttonLayout);

        // Input section
        QHBoxLayout* inputLayout = new QHBoxLayout();
        
        QLabel* inputLabel = new QLabel("Input:", this);
        inputLayout->addWidget(inputLabel);

        m_lineEdit = new QLineEdit(this);
        m_lineEdit->setObjectName("TextInput");
        m_lineEdit->setPlaceholderText("Type something...");
        connect(m_lineEdit, &QLineEdit::textChanged, this, &TestApplicationWindow::onTextChanged);
        inputLayout->addWidget(m_lineEdit);

        m_echoLabel = new QLabel("Echo:", this);
        m_echoLabel->setObjectName("EchoLabel");
        inputLayout->addWidget(m_echoLabel);

        mainLayout->addLayout(inputLayout);

        // Combo box section
        QHBoxLayout* comboLayout = new QHBoxLayout();
        
        QLabel* comboLabel = new QLabel("Select:", this);
        comboLayout->addWidget(comboLabel);

        m_comboBox = new QComboBox(this);
        m_comboBox->setObjectName("SelectionCombo");
        m_comboBox->addItems({"Option 1", "Option 2", "Option 3", "Option 4"});
        connect(m_comboBox, QOverload<const QString&>::of(&QComboBox::currentTextChanged),
                this, &TestApplicationWindow::onComboChanged);
        comboLayout->addWidget(m_comboBox);

        m_selectionLabel = new QLabel("Selected: Option 1", this);
        m_selectionLabel->setObjectName("SelectionLabel");
        comboLayout->addWidget(m_selectionLabel);

        mainLayout->addLayout(comboLayout);

        // Checkbox section
        QHBoxLayout* checkLayout = new QHBoxLayout();
        
        m_checkBox = new QCheckBox("Enable feature", this);
        m_checkBox->setObjectName("FeatureCheckbox");
        connect(m_checkBox, &QCheckBox::toggled, this, &TestApplicationWindow::onCheckToggled);
        checkLayout->addWidget(m_checkBox);

        m_checkLabel = new QLabel("Unchecked", this);
        m_checkLabel->setObjectName("CheckLabel");
        checkLayout->addWidget(m_checkLabel);

        mainLayout->addLayout(checkLayout);

        // Text area
        m_textEdit = new QTextEdit(this);
        m_textEdit->setObjectName("TextArea");
        m_textEdit->setPlaceholderText("Enter multi-line text here...");
        m_textEdit->setMaximumHeight(100);
        mainLayout->addWidget(m_textEdit);

        // Dialog and delayed action buttons
        QHBoxLayout* actionLayout = new QHBoxLayout();
        
        QPushButton* dialogButton = new QPushButton("Show Dialog", this);
        dialogButton->setObjectName("DialogButton");
        connect(dialogButton, &QPushButton::clicked, this, &TestApplicationWindow::showDialog);
        actionLayout->addWidget(dialogButton);

        m_delayedButton = new QPushButton("Delayed Action", this);
        m_delayedButton->setObjectName("DelayedButton");
        connect(m_delayedButton, &QPushButton::clicked, this, &TestApplicationWindow::delayedAction);
        actionLayout->addWidget(m_delayedButton);

        mainLayout->addLayout(actionLayout);

        // Menu bar
        setupMenuBar();

        m_clickCount = 0;
    }

    void setupMenuBar() {
        QMenuBar* menuBar = this->menuBar();
        
        QMenu* fileMenu = menuBar->addMenu("File");
        fileMenu->setObjectName("FileMenu");
        
        QAction* newAction = fileMenu->addAction("New");
        newAction->setObjectName("NewAction");
        newAction->setShortcut(QKeySequence::New);
        
        QAction* openAction = fileMenu->addAction("Open");
        openAction->setObjectName("OpenAction");
        openAction->setShortcut(QKeySequence::Open);
        
        fileMenu->addSeparator();
        
        QAction* exitAction = fileMenu->addAction("Exit");
        exitAction->setObjectName("ExitAction");
        exitAction->setShortcut(QKeySequence::Quit);
        connect(exitAction, &QAction::triggered, this, &QWidget::close);

        QMenu* editMenu = menuBar->addMenu("Edit");
        editMenu->setObjectName("EditMenu");
        
        QAction* copyAction = editMenu->addAction("Copy");
        copyAction->setObjectName("CopyAction");
        copyAction->setShortcut(QKeySequence::Copy);
        
        QAction* pasteAction = editMenu->addAction("Paste");
        pasteAction->setObjectName("PasteAction");
        pasteAction->setShortcut(QKeySequence::Paste);
    }

private:
    QLabel* m_statusLabel;
    QPushButton* m_clickButton;
    QLabel* m_counterLabel;
    QLineEdit* m_lineEdit;
    QLabel* m_echoLabel;
    QComboBox* m_comboBox;
    QLabel* m_selectionLabel;
    QCheckBox* m_checkBox;
    QLabel* m_checkLabel;
    QTextEdit* m_textEdit;
    QPushButton* m_delayedButton;
    int m_clickCount;
};

/**
 * @brief Example test class demonstrating the UI automation framework
 */
DECLARE_TEST_CLASS(UIAutomationExample, UI, High, "ui-automation", "framework", "example")

private:
    TestApplicationWindow* m_testWindow;
    UIAutomation* m_uiAutomation;

private slots:
    void initTestCase() {
        TestBase::initTestCase();
        logTestInfo("Setting up UI automation example");
        
        // Create test application window
        m_testWindow = new TestApplicationWindow();
        m_testWindow->show();
        
        // Wait for window to be fully displayed
        QTest::qWaitForWindowActive(m_testWindow);
        
        // Create UI automation instance
        m_uiAutomation = new UIAutomation(this);
        m_uiAutomation->enableDetailedLogging(true);
        m_uiAutomation->enableAutomaticScreenshots(false); // Disable for CI
        m_uiAutomation->setDefaultTimeout(5000);
    }

    void cleanupTestCase() {
        logTestInfo("Cleaning up UI automation example");
        
        if (m_testWindow) {
            m_testWindow->close();
            delete m_testWindow;
            m_testWindow = nullptr;
        }
        
        TestBase::cleanupTestCase();
    }

    // Test basic widget finding and interaction
    TEST_METHOD(test_uiAutomation_basicInteraction_worksCorrectly) {
        logTestStep("Testing basic UI automation interactions");
        
        // Test button clicking
        auto buttonSelector = UIAutomation::byObjectName("ClickButton");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetExists(buttonSelector), 
                           "Click button should exist");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetVisible(buttonSelector), 
                           "Click button should be visible");
        
        // Click the button and verify result
        TEST_VERIFY_WITH_MSG(m_uiAutomation->clickWidget(buttonSelector), 
                           "Should be able to click the button");
        
        // Verify status label updated
        auto statusSelector = UIAutomation::byObjectName("StatusLabel");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetText(statusSelector, "Button clicked!"), 
                           "Status label should show button clicked message");
        
        // Verify counter updated
        auto counterSelector = UIAutomation::byObjectName("CounterLabel");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetText(counterSelector, "Clicks: 1"), 
                           "Counter should show 1 click");
        
        logTestStep("Basic interaction test completed successfully");
    }

    TEST_METHOD(test_uiAutomation_textInput_handlesTypingCorrectly) {
        logTestStep("Testing text input automation");
        
        // Test text input
        auto inputSelector = UIAutomation::byObjectName("TextInput");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetExists(inputSelector), 
                           "Text input should exist");
        
        QString testText = "Hello, UI Automation!";
        TEST_VERIFY_WITH_MSG(m_uiAutomation->typeText(inputSelector, testText), 
                           "Should be able to type text");
        
        // Verify input value
        QVariant inputValue = m_uiAutomation->getWidgetValue(inputSelector);
        TEST_COMPARE_WITH_MSG(inputValue.toString(), testText, 
                            "Input should contain typed text");
        
        // Verify echo label updated
        auto echoSelector = UIAutomation::byObjectName("EchoLabel");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetText(echoSelector, "Echo: " + testText), 
                           "Echo label should show typed text");
        
        // Test text clearing
        TEST_VERIFY_WITH_MSG(m_uiAutomation->clearText(inputSelector), 
                           "Should be able to clear text");
        
        QVariant clearedValue = m_uiAutomation->getWidgetValue(inputSelector);
        TEST_VERIFY_WITH_MSG(clearedValue.toString().isEmpty(), 
                           "Input should be empty after clearing");
        
        logTestStep("Text input test completed successfully");
    }

    TEST_METHOD(test_uiAutomation_formFilling_worksCorrectly) {
        logTestStep("Testing form filling automation");
        
        // Define form fields
        QList<UIAutomation::FormField> formFields = {
            {UIAutomation::byObjectName("TextInput"), "Automated form input", "", true},
            {UIAutomation::byObjectName("SelectionCombo"), "Option 3", "", false},
            {UIAutomation::byObjectName("FeatureCheckbox"), true, "", false}
        };
        
        // Fill the form
        TEST_VERIFY_WITH_MSG(m_uiAutomation->fillForm(formFields), 
                           "Should be able to fill form");
        
        // Verify form values
        auto inputSelector = UIAutomation::byObjectName("TextInput");
        QVariant inputValue = m_uiAutomation->getWidgetValue(inputSelector);
        TEST_COMPARE_WITH_MSG(inputValue.toString(), QString("Automated form input"), 
                            "Text input should have correct value");
        
        auto comboSelector = UIAutomation::byObjectName("SelectionCombo");
        QVariant comboValue = m_uiAutomation->getWidgetValue(comboSelector);
        TEST_COMPARE_WITH_MSG(comboValue.toString(), QString("Option 3"), 
                            "Combo box should have correct selection");
        
        auto checkSelector = UIAutomation::byObjectName("FeatureCheckbox");
        QVariant checkValue = m_uiAutomation->getWidgetValue(checkSelector);
        TEST_VERIFY_WITH_MSG(checkValue.toBool(), 
                           "Checkbox should be checked");
        
        // Verify UI feedback
        auto selectionLabelSelector = UIAutomation::byObjectName("SelectionLabel");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetText(selectionLabelSelector, "Selected: Option 3"), 
                           "Selection label should show correct option");
        
        auto checkLabelSelector = UIAutomation::byObjectName("CheckLabel");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetText(checkLabelSelector, "Checked"), 
                           "Check label should show checked state");
        
        logTestStep("Form filling test completed successfully");
    }

    TEST_METHOD(test_uiAutomation_dialogHandling_worksCorrectly) {
        logTestStep("Testing dialog handling automation");
        
        // Click button to show dialog
        auto dialogButtonSelector = UIAutomation::byObjectName("DialogButton");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->clickWidget(dialogButtonSelector), 
                           "Should be able to click dialog button");
        
        // Wait for dialog to appear
        auto dialogSelector = UIAutomation::byObjectName("TestDialog");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->waitForWidget(dialogSelector, 2000), 
                           "Dialog should appear");
        
        // Interact with dialog
        auto dialogInputSelector = UIAutomation::byObjectName("DialogInput");
        QString dialogText = "Dialog automation test";
        TEST_VERIFY_WITH_MSG(m_uiAutomation->typeText(dialogInputSelector, dialogText), 
                           "Should be able to type in dialog");
        
        // Accept dialog by clicking OK button
        auto dialogButtonsSelector = UIAutomation::byObjectName("DialogButtons");
        QWidget* buttonBox = m_uiAutomation->findWidget(dialogButtonsSelector);
        TEST_VERIFY_WITH_MSG(buttonBox != nullptr, "Dialog button box should exist");
        
        // Find OK button within the button box
        auto okButtonSelector = UIAutomation::byText("OK");
        QWidget* okButton = m_uiAutomation->findWidget(okButtonSelector, buttonBox);
        if (okButton) {
            TEST_VERIFY_WITH_MSG(m_uiAutomation->clickWidget(UIAutomation::byObjectName(okButton->objectName())), 
                               "Should be able to click OK button");
        } else {
            // Alternative: use keyboard shortcut
            TEST_VERIFY_WITH_MSG(m_uiAutomation->pressKey(dialogInputSelector, Qt::Key_Return), 
                               "Should be able to press Enter to accept dialog");
        }
        
        // Wait for dialog to close
        TEST_VERIFY_WITH_MSG(m_uiAutomation->waitForCondition({
            UIAutomation::WaitCondition::WidgetHidden,
            dialogSelector,
            "",
            "",
            nullptr,
            nullptr,
            nullptr,
            3000,
            100
        }), "Dialog should close");
        
        // Verify status message
        auto statusSelector = UIAutomation::byObjectName("StatusLabel");
        QString expectedStatus = "Dialog accepted: " + dialogText;
        TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetText(statusSelector, expectedStatus), 
                           "Status should show dialog was accepted with correct text");
        
        logTestStep("Dialog handling test completed successfully");
    }

    TEST_METHOD(test_uiAutomation_waitConditions_workCorrectly) {
        logTestStep("Testing wait conditions and synchronization");
        
        // Test delayed action that disables/enables button
        auto delayedButtonSelector = UIAutomation::byObjectName("DelayedButton");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->verifyWidgetEnabled(delayedButtonSelector), 
                           "Delayed button should initially be enabled");
        
        // Click delayed action button
        TEST_VERIFY_WITH_MSG(m_uiAutomation->clickWidget(delayedButtonSelector), 
                           "Should be able to click delayed button");
        
        // Verify button becomes disabled
        TEST_VERIFY_WITH_MSG(m_uiAutomation->waitForCondition({
            UIAutomation::WaitCondition::WidgetDisabled,
            delayedButtonSelector,
            "",
            "",
            nullptr,
            nullptr,
            nullptr,
            1000,
            50
        }), "Button should become disabled");
        
        // Wait for status to change to "Starting delayed action..."
        auto statusSelector = UIAutomation::byObjectName("StatusLabel");
        TEST_VERIFY_WITH_MSG(m_uiAutomation->waitForCondition({
            UIAutomation::WaitCondition::TextChanged,
            statusSelector,
            "Starting delayed action...",
            "",
            nullptr,
            nullptr,
            nullptr,
            1000,
            50
        }), "Status should show delayed action started");
        
        // Wait for delayed action to complete (button re-enabled)
        TEST_VERIFY_WITH_MSG(m_uiAutomation->waitForCondition({
            UIAutomation::WaitCondition::WidgetEnabled,
            delayedButtonSelector,
            "",
            "",
            nullptr,
            nullptr,
            nullptr,
            3000,
            100
        }), "Button should become enabled again after delay");
        
        // Verify final status
        TEST_VERIFY_WITH_MSG(m_uiAutomation->waitForCondition({
            UIAutomation::WaitCondition::TextChanged,
            statusSelector,
            "Delayed action completed!",
            "",
            nullptr,
            nullptr,
            nullptr,
            1000,
            50
        }), "Status should show delayed action completed");
        
        logTestStep("Wait conditions test completed successfully");
    }

    TEST_METHOD(test_uiAutomation_screenshotCapture_worksCorrectly) {
        logTestStep("Testing screenshot capture functionality");
        
        // Capture screenshot of main window
        auto windowSelector = UIAutomation::byObjectName("TestApplicationWindow");
        QPixmap screenshot = m_uiAutomation->captureWidget(windowSelector);
        TEST_VERIFY_WITH_MSG(!screenshot.isNull(), "Should capture valid screenshot");
        TEST_VERIFY_WITH_MSG(screenshot.width() > 0 && screenshot.height() > 0, 
                           "Screenshot should have valid dimensions");
        
        // Save screenshot to file
        QString screenshotPath = createTestDirectory("screenshots") + "/test_window.png";
        TEST_VERIFY_WITH_MSG(m_uiAutomation->saveScreenshot(windowSelector, screenshotPath), 
                           "Should save screenshot to file");
        TEST_VERIFY_WITH_MSG(QFile::exists(screenshotPath), 
                           "Screenshot file should exist");
        
        // Capture screenshot of specific widget
        auto buttonSelector = UIAutomation::byObjectName("ClickButton");
        QPixmap buttonScreenshot = m_uiAutomation->captureWidget(buttonSelector);
        TEST_VERIFY_WITH_MSG(!buttonScreenshot.isNull(), "Should capture button screenshot");
        TEST_VERIFY_WITH_MSG(buttonScreenshot.width() < screenshot.width(), 
                           "Button screenshot should be smaller than window screenshot");
        
        logTestStep("Screenshot capture test completed successfully");
    }

END_TEST_CLASS()

/**
 * @brief Main function for running the UI automation example
 */
int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    
    qDebug() << "========================================";
    qDebug() << "UI Automation Framework Example";
    qDebug() << "========================================";
    
    // Load test configuration
    TestConfig::instance().loadConfiguration();
    
    // Create and run the test
    UIAutomationExample test;
    
    if (test.shouldRunTest()) {
        int result = QTest::qExec(&test, argc, argv);
        
        if (result == 0) {
            qDebug() << "✅ UI automation example PASSED";
        } else {
            qDebug() << "❌ UI automation example FAILED";
        }
        
        return result;
    } else {
        qDebug() << "⏭️  UI automation example SKIPPED (disabled by configuration)";
        return 0;
    }
}

#include "example_ui_automation.moc"