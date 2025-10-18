# DupFinder Testing Documentation

Welcome to the comprehensive testing documentation for DupFinder. This documentation provides everything you need to understand, use, and contribute to our automated testing suite.

## Quick Start

- **New to testing?** Start with the [Testing Framework Overview](framework-overview.md)
- **Writing your first test?** Check out [Test Writing Guidelines](test-writing-guidelines.md)
- **Need examples?** Browse our [Test Examples](examples/)
- **Having issues?** See the [Troubleshooting Guide](troubleshooting.md)

## Documentation Structure

### Core Documentation
- [Testing Framework Overview](framework-overview.md) - High-level architecture and concepts
- [Test Writing Guidelines](test-writing-guidelines.md) - Standards and best practices
- [API Reference](api/) - Complete API documentation for all testing frameworks
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions

### Testing Categories
- [Unit Testing Guide](unit-testing.md) - Component-level testing
- [Integration Testing Guide](integration-testing.md) - Component interaction testing
- [UI Testing Guide](ui-testing.md) - User interface automation
- [End-to-End Testing Guide](e2e-testing.md) - Complete workflow testing
- [Performance Testing Guide](performance-testing.md) - Performance and load testing
- [Accessibility Testing Guide](accessibility-testing.md) - A11y compliance testing
- [Cross-Platform Testing Guide](cross-platform-testing.md) - Multi-platform validation

### Advanced Topics
- [Test Configuration](configuration.md) - Test suite configuration options
- [CI/CD Integration](ci-cd-integration.md) - Continuous integration setup
- [Test Maintenance](test-maintenance.md) - Keeping tests healthy
- [Performance Optimization](performance-optimization.md) - Optimizing test execution

### Examples and Templates
- [Test Examples](examples/) - Working examples for each test category
- [Test Templates](templates/) - Boilerplate code for common scenarios
- [Code Generation Tools](tools/) - Automated test creation utilities

## Getting Started

### Prerequisites
- Qt 6.x development environment
- CMake 3.16 or higher
- C++17 compatible compiler
- Basic understanding of Qt Test Framework

### Running Tests
```bash
# Build and run all tests
mkdir build && cd build
cmake ..
make
ctest

# Run specific test categories
ctest -L unit
ctest -L integration
ctest -L ui
ctest -L performance
```

### Writing Your First Test
1. Choose the appropriate test category (unit, integration, UI, etc.)
2. Follow the [Test Writing Guidelines](test-writing-guidelines.md)
3. Use the appropriate [template](templates/) for your test type
4. Run and validate your test
5. Add documentation for complex test scenarios

## Contributing

We welcome contributions to our testing suite! Please read:
- [Test Writing Guidelines](test-writing-guidelines.md) for coding standards
- [Contributing Guide](../CONTRIBUTING.md) for general contribution guidelines
- [Code Review Checklist](code-review-checklist.md) for review criteria

## Support

- **Documentation Issues**: Create an issue in our repository
- **Test Framework Bugs**: Report via our issue tracker
- **Questions**: Check the [FAQ](faq.md) or ask in discussions

## Version Information

This documentation is for DupFinder Testing Suite v2.0.
Last updated: October 2025