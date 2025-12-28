# End-to-End Testing Framework

## Overview

The End-to-End Testing Framework provides comprehensive testing capabilities for the CloneClean application through three integrated components:

1. **Workflow Testing** - Complete user journey validation
2. **User Scenario Testing** - Persona-based testing with UX metrics
3. **Cross-Platform Testing** - Platform compatibility validation

## Components

### 1. Workflow Testing (`workflow_testing.h/cpp`)

Provides workflow definition, execution, and validation capabilities.

#### Key Features:
- **Workflow Definition Language**: JSON-based workflow specification
- **Step Types**: UI actions, file operations, validations, waits, setup/cleanup
- **Error Simulation**: Inject errors to test recovery mechanisms
- **State Validation**: Verify application and file system state
- **Execution Engine**: Robust workflow execution with retry logic

#### Example Usage:
```cpp
#include "workflow_testing.h"

// Create workflow testing instance
auto workflowTesting = std::make_shared<WorkflowTesting>();
workflowTesting->setTestEnvironment(testEnvironment);
workflowTesting->setUIAutomation(uiAutomation);

// Create and execute a workflow
UserWorkflow workflow = workflowTesting->createScanToDeleteWorkflow();
WorkflowResult result = workflowTesting->executeWorkflow(workflow);

// Validate results
if (result.success) {
    qDebug() << "Workflow completed successfully";
    qDebug() << "Execution time:" << result.executionTimeMs << "ms";
    qDebug() << "Steps completed:" << result.completedSteps;
}
```

#### Predefined Workflows:
- **Scan-to-Delete Workflow**: Complete duplicate scanning and deletion
- **First-Time User Workflow**: Initial user onboarding experience
- **Power User Workflow**: Advanced features and batch operations
- **Safety-Focused Workflow**: Backup and restore operations

### 2. User Scenario Testing (`user_scenario_testing.h/cpp`)

Extends workflow testing with user persona validation and UX metrics.

#### Key Features:
- **User Personas**: First-time, casual, power, safety-focused, accessibility users
- **UX Metrics**: Interaction time, wait time, action count, satisfaction scores
- **Goal Tracking**: Monitor completion of user goals
- **Usability Issue Detection**: Identify interface and workflow problems
- **Accessibility Testing**: Validate keyboard navigation and screen reader support

#### Example Usage:
```cpp
#include "user_scenario_testing.h"

// Create scenario testing instance
auto scenarioTesting = std::make_shared<UserScenarioTesting>();
scenarioTesting->setWorkflowTesting(workflowTesting);

// Create and execute a user scenario
UserScenario scenario = scenarioTesting->createFirstTimeUserScenario();
ScenarioResult result = scenarioTesting->executeScenario(scenario);

// Analyze user experience
if (result.success) {
    qDebug() << "Scenario completed for persona:" << 
                UserScenarioTesting::personaToString(result.persona);
    qDebug() << "Satisfaction score:" << result.satisfactionScore << "/10";
    qDebug() << "Goals completed:" << result.completedGoals.size();
    qDebug() << "Usability issues:" << result.usabilityIssues.size();
}
```

#### Supported Personas:
- **FirstTimeUser**: New users learning the application
- **CasualUser**: Occasional users with basic needs
- **PowerUser**: Advanced users utilizing complex features
- **SafetyFocusedUser**: Users prioritizing data safety
- **BatchUser**: Users processing large amounts of data
- **AccessibilityUser**: Users requiring accessibility features
- **MobileUser**: Users on mobile/touch devices

### 3. Cross-Platform Testing (`cross_platform_testing.h/cpp`)

Validates application behavior across different operating systems and configurations.

#### Key Features:
- **Platform Detection**: Automatic OS and version detection
- **File System Testing**: Compatibility across NTFS, HFS+, ext4, etc.
- **Display Scaling**: High-DPI and multi-monitor support
- **OS Integration**: File manager, trash, system dialogs
- **Path Handling**: Cross-platform path normalization
- **Platform Adaptation**: Automatic workflow adaptation per platform

#### Example Usage:
```cpp
#include "cross_platform_testing.h"

// Create cross-platform testing instance
auto crossPlatformTesting = std::make_shared<CrossPlatformTesting>();
crossPlatformTesting->setWorkflowTesting(workflowTesting);

// Create and execute cross-platform test
CrossPlatformTest test = crossPlatformTesting->createFileOperationTest();
CrossPlatformResult result = crossPlatformTesting->executeCrossPlatformTest(test);

// Analyze platform compatibility
if (result.success) {
    Platform platform = crossPlatformTesting->getCurrentPlatform();
    qDebug() << "Test passed on" << CrossPlatformTesting::platformToString(platform);
    
    if (!result.platformDifferences[platform].isEmpty()) {
        qDebug() << "Platform differences:" << result.platformDifferences[platform];
    }
}
```

#### Supported Platforms:
- **Windows**: Windows 10/11 with NTFS file system
- **macOS**: macOS 10.15+ with APFS/HFS+ file systems
- **Linux**: Various distributions with ext4/btrfs file systems

## Integration Example

Here's how to integrate all three components:

```cpp
#include "workflow_testing.h"
#include "user_scenario_testing.h"
#include "cross_platform_testing.h"
#include "framework/test_environment.h"
#include "ui_automation.h"

void setupEndToEndTesting() {
    // Initialize core components
    auto testEnvironment = std::make_shared<TestEnvironment>();
    auto uiAutomation = std::make_shared<UIAutomation>();
    auto workflowTesting = std::make_shared<WorkflowTesting>();
    auto scenarioTesting = std::make_shared<UserScenarioTesting>();
    auto crossPlatformTesting = std::make_shared<CrossPlatformTesting>();
    
    // Setup component relationships
    workflowTesting->setTestEnvironment(testEnvironment);
    workflowTesting->setUIAutomation(uiAutomation);
    scenarioTesting->setWorkflowTesting(workflowTesting);
    crossPlatformTesting->setWorkflowTesting(workflowTesting);
    
    // Configure testing
    workflowTesting->enableDetailedLogging(true);
    workflowTesting->enableAutomaticScreenshots(true);
    scenarioTesting->enableUserExperienceMetrics(true);
    scenarioTesting->enableAccessibilityTesting(true);
    
    // Setup test environment
    testEnvironment->setupTestEnvironment();
}

void runComprehensiveTest() {
    // Execute workflow test
    UserWorkflow workflow = workflowTesting->createScanToDeleteWorkflow();
    WorkflowResult workflowResult = workflowTesting->executeWorkflow(workflow);
    
    // Execute scenario test
    UserScenario scenario = scenarioTesting->createFirstTimeUserScenario();
    ScenarioResult scenarioResult = scenarioTesting->executeScenario(scenario);
    
    // Execute cross-platform test
    CrossPlatformTest cpTest = crossPlatformTesting->createFileOperationTest();
    CrossPlatformResult cpResult = crossPlatformTesting->executeCrossPlatformTest(cpTest);
    
    // Analyze combined results
    bool allTestsPassed = workflowResult.success && 
                         scenarioResult.success && 
                         cpResult.success;
    
    qDebug() << "Comprehensive test result:" << (allTestsPassed ? "PASS" : "FAIL");
}
```

## Workflow Definition Format

Workflows are defined using a structured format:

```cpp
UserWorkflow workflow;
workflow.id = "my_workflow";
workflow.name = "My Test Workflow";
workflow.description = "Description of what this workflow tests";

// Add workflow steps
WorkflowStep step;
step.id = "step_1";
step.name = "Step Name";
step.type = WorkflowStepType::UIAction;
step.parameters["action"] = "click";
step.parameters["selector"] = "button_id";
step.preconditions = {"app_launched"};
step.postconditions = {"button_clicked"};
step.timeoutMs = 30000;

workflow.steps.append(step);
```

### Step Types:
- **UIAction**: User interface interactions (click, type, navigate)
- **FileOperation**: File system operations (create, delete, copy)
- **Validation**: State validation and verification
- **Wait**: Wait for conditions or timeouts
- **Setup**: Environment setup and preparation
- **Cleanup**: Environment cleanup and teardown
- **Custom**: Custom action functions

## Error Handling and Recovery

The framework provides comprehensive error handling:

```cpp
// Configure error recovery
WorkflowStep step;
step.retryOnFailure = true;
step.maxRetries = 3;
step.optional = false; // Step must succeed

// Error simulation
ErrorScenario errorScenario;
errorScenario.errorType = ErrorType::FileSystemError;
errorScenario.triggerStepId = "file_operation_step";
errorScenario.shouldRecover = true;

WorkflowResult result = workflowTesting->testErrorRecovery(errorScenario);
```

## Configuration Options

### Workflow Testing Configuration:
```cpp
workflowTesting->setDefaultTimeout(60000);
workflowTesting->setScreenshotDirectory("/path/to/screenshots");
workflowTesting->setLogDirectory("/path/to/logs");
workflowTesting->enableDetailedLogging(true);
workflowTesting->enableAutomaticScreenshots(true);
workflowTesting->setRetryAttempts(3);
```

### Scenario Testing Configuration:
```cpp
scenarioTesting->setScenarioTimeout(300000);
scenarioTesting->enableUserExperienceMetrics(true);
scenarioTesting->enableAccessibilityTesting(true);
scenarioTesting->setErrorRecoveryTimeout(60000);
```

### Cross-Platform Testing Configuration:
```cpp
crossPlatformTesting->enablePlatformEmulation(true);
crossPlatformTesting->setEmulatedPlatform(Platform::Windows);
crossPlatformTesting->enableFileSystemEmulation(true);
crossPlatformTesting->enableDisplayEmulation(true);
```

## Reporting and Analysis

The framework generates comprehensive reports:

### Workflow Reports:
- Execution time per step
- Success/failure rates
- Error messages and stack traces
- Screenshots at failure points
- Performance metrics

### Scenario Reports:
- User experience metrics
- Satisfaction scores
- Goal completion rates
- Usability issue identification
- Accessibility compliance

### Cross-Platform Reports:
- Platform compatibility matrix
- Performance comparison across platforms
- Platform-specific behavior differences
- Missing feature identification

## Best Practices

### 1. Workflow Design:
- Keep workflows focused on specific user journeys
- Use descriptive step names and IDs
- Include appropriate preconditions and postconditions
- Set realistic timeouts for each step
- Design for error recovery

### 2. Scenario Testing:
- Define clear user goals and success criteria
- Consider different user personas and their needs
- Include accessibility testing for all scenarios
- Monitor user experience metrics
- Test edge cases and error conditions

### 3. Cross-Platform Testing:
- Test on all target platforms regularly
- Account for platform-specific behaviors
- Validate file system compatibility
- Test display scaling and multi-monitor setups
- Verify OS integration features

### 4. Maintenance:
- Keep workflows updated with application changes
- Review and update baselines regularly
- Monitor test execution times and optimize
- Address flaky tests promptly
- Document platform-specific behaviors

## Troubleshooting

### Common Issues:

1. **Workflow Timeouts**:
   - Increase step timeouts for slow operations
   - Check for UI synchronization issues
   - Verify test environment performance

2. **UI Automation Failures**:
   - Verify widget selectors are correct
   - Check for timing issues with UI updates
   - Ensure application is in expected state

3. **Cross-Platform Differences**:
   - Review expected platform differences
   - Update platform adaptations as needed
   - Check file system compatibility

4. **Scenario Failures**:
   - Review user goals and success criteria
   - Check persona-specific configurations
   - Validate accessibility requirements

## Dependencies

The End-to-End Testing Framework depends on:
- Qt Test Framework
- Test Environment (`framework/test_environment.h`)
- UI Automation (`ui_automation.h`)
- Test Harness (`framework/test_harness.h`)

## Files

### Core Framework:
- `workflow_testing.h/cpp` - Workflow testing implementation
- `user_scenario_testing.h/cpp` - User scenario testing implementation
- `cross_platform_testing.h/cpp` - Cross-platform testing implementation

### Examples and Documentation:
- `example_end_to_end_testing.cpp` - Complete usage examples
- `END_TO_END_TESTING_README.md` - This documentation file

### Integration:
- Integrates with existing test framework in `tests/framework/`
- Uses UI automation from `ui_automation.h/cpp`
- Leverages test environment from `framework/test_environment.h/cpp`