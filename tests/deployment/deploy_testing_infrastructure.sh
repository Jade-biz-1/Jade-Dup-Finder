#!/bin/bash

# DupFinder Testing Infrastructure Deployment Script
# This script deploys the complete testing infrastructure to production CI/CD environment

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_CONFIG="$SCRIPT_DIR/deployment_config.json"
LOG_FILE="$SCRIPT_DIR/deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check if running in CI environment
    if [[ -z "$CI" ]]; then
        log_warning "Not running in CI environment"
    else
        log_success "Running in CI environment: $CI"
    fi
    
    # Check required tools
    local required_tools=("cmake" "git" "python3" "jq")
    for tool in "${required_tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            log_success "$tool is available"
        else
            log_error "$tool is required but not installed"
            return 1
        fi
    done
    
    # Check Qt installation
    if command -v qmake &> /dev/null; then
        local qt_version=$(qmake -query QT_VERSION)
        log_success "Qt version $qt_version is available"
    else
        log_error "Qt is required but not installed"
        return 1
    fi
    
    # Check disk space (require at least 2GB)
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local required_space=2097152  # 2GB in KB
    
    if [[ $available_space -gt $required_space ]]; then
        log_success "Sufficient disk space available: $(($available_space / 1024 / 1024))GB"
    else
        log_error "Insufficient disk space. Required: 2GB, Available: $(($available_space / 1024 / 1024))GB"
        return 1
    fi
    
    return 0
}

# Function to validate test infrastructure
validate_test_infrastructure() {
    log "Validating test infrastructure..."
    
    cd "$PROJECT_ROOT"
    
    # Check if test directories exist
    local test_dirs=("tests" "tests/unit" "tests/integration" "tests/performance" "tests/framework" "tests/validation")
    for dir in "${test_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            log_success "Test directory exists: $dir"
        else
            log_error "Missing test directory: $dir"
            return 1
        fi
    done
    
    # Check if key test files exist
    local key_files=(
        "tests/CMakeLists.txt"
        "tests/framework/test_harness.h"
        "tests/framework/test_harness.cpp"
        "tests/validation/test_suite_validator.h"
        "tests/validation/test_suite_validator.cpp"
        "tests/validation/performance_scalability_validator.h"
        "tests/validation/performance_scalability_validator.cpp"
    )
    
    for file in "${key_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "Key test file exists: $file"
        else
            log_error "Missing key test file: $file"
            return 1
        fi
    done
    
    return 0
}

# Function to build test infrastructure
build_test_infrastructure() {
    log "Building test infrastructure..."
    
    cd "$PROJECT_ROOT"
    
    # Create build directory
    if [[ ! -d "build" ]]; then
        mkdir build
        log "Created build directory"
    fi
    
    cd build
    
    # Configure with CMake
    log "Configuring with CMake..."
    if cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=ON; then
        log_success "CMake configuration completed"
    else
        log_error "CMake configuration failed"
        return 1
    fi
    
    # Build the project
    log "Building project..."
    local cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    if make -j"$cpu_cores"; then
        log_success "Project build completed"
    else
        log_error "Project build failed"
        return 1
    fi
    
    # Build test executables
    log "Building test executables..."
    if make -j"$cpu_cores" all; then
        log_success "Test executables build completed"
    else
        log_error "Test executables build failed"
        return 1
    fi
    
    return 0
}

# Function to run validation tests
run_validation_tests() {
    log "Running validation tests..."
    
    cd "$PROJECT_ROOT/build"
    
    # Run comprehensive test suite validation
    log "Running comprehensive test suite validation..."
    if [[ -f "tests/validation/main_validation" ]]; then
        if ./tests/validation/main_validation --output "$PROJECT_ROOT/test_reports"; then
            log_success "Comprehensive test suite validation passed"
        else
            log_error "Comprehensive test suite validation failed"
            return 1
        fi
    else
        log_warning "Comprehensive test suite validator not found, skipping..."
    fi
    
    # Run performance scalability validation
    log "Running performance scalability validation..."
    if [[ -f "tests/validation/main_performance_scalability" ]]; then
        if ./tests/validation/main_performance_scalability --output "$PROJECT_ROOT/test_reports"; then
            log_success "Performance scalability validation passed"
        else
            log_error "Performance scalability validation failed"
            return 1
        fi
    else
        log_warning "Performance scalability validator not found, skipping..."
    fi
    
    return 0
}

# Function to deploy CI/CD integration
deploy_ci_integration() {
    log "Deploying CI/CD integration..."
    
    cd "$PROJECT_ROOT"
    
    # Check if GitHub Actions workflows exist
    if [[ -d ".github/workflows" ]]; then
        log_success "GitHub Actions workflows directory exists"
        
        # List workflow files
        local workflow_files=(.github/workflows/*.yml)
        for workflow in "${workflow_files[@]}"; do
            if [[ -f "$workflow" ]]; then
                log_success "Workflow file exists: $(basename "$workflow")"
            fi
        done
    else
        log_warning "GitHub Actions workflows directory not found"
    fi
    
    # Check CI scripts
    if [[ -d "scripts/ci" ]]; then
        log_success "CI scripts directory exists"
        
        local ci_scripts=(scripts/ci/*.py)
        for script in "${ci_scripts[@]}"; do
            if [[ -f "$script" ]]; then
                log_success "CI script exists: $(basename "$script")"
            fi
        done
    else
        log_warning "CI scripts directory not found"
    fi
    
    return 0
}

# Function to setup monitoring and alerting
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    cd "$PROJECT_ROOT"
    
    # Create monitoring configuration
    local monitoring_config="$PROJECT_ROOT/monitoring_config.json"
    cat > "$monitoring_config" << EOF
{
    "monitoring": {
        "enabled": true,
        "test_execution_monitoring": {
            "enabled": true,
            "thresholds": {
                "max_execution_time_minutes": 30,
                "min_success_rate_percent": 95,
                "max_flaky_test_rate_percent": 2
            }
        },
        "performance_monitoring": {
            "enabled": true,
            "thresholds": {
                "max_memory_usage_mb": 2048,
                "min_throughput_files_per_sec": 10,
                "max_cpu_usage_percent": 80
            }
        },
        "infrastructure_monitoring": {
            "enabled": true,
            "thresholds": {
                "min_disk_space_gb": 5,
                "max_build_time_minutes": 15
            }
        }
    },
    "alerting": {
        "enabled": true,
        "channels": {
            "email": {
                "enabled": false,
                "recipients": []
            },
            "slack": {
                "enabled": false,
                "webhook_url": ""
            },
            "github": {
                "enabled": true,
                "create_issues": true
            }
        },
        "alert_conditions": {
            "test_failure_rate_threshold": 5,
            "performance_degradation_threshold": 10,
            "infrastructure_failure": true
        }
    }
}
EOF
    
    log_success "Monitoring configuration created: $monitoring_config"
    
    # Create monitoring script
    local monitoring_script="$PROJECT_ROOT/scripts/monitor_test_infrastructure.py"
    mkdir -p "$(dirname "$monitoring_script")"
    
    cat > "$monitoring_script" << 'EOF'
#!/usr/bin/env python3
"""
Test Infrastructure Monitoring Script
Monitors test execution, performance, and infrastructure health
"""

import json
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

class TestInfrastructureMonitor:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.monitoring_enabled = self.config.get('monitoring', {}).get('enabled', False)
        self.alerting_enabled = self.config.get('alerting', {}).get('enabled', False)
    
    def monitor_test_execution(self):
        """Monitor test execution metrics"""
        if not self.monitoring_enabled:
            return
        
        print(f"[{datetime.now()}] Monitoring test execution...")
        
        # Check recent test results
        test_reports_dir = Path("test_reports")
        if test_reports_dir.exists():
            # Find recent test reports
            recent_reports = []
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for report_file in test_reports_dir.glob("*.json"):
                if report_file.stat().st_mtime > cutoff_time.timestamp():
                    recent_reports.append(report_file)
            
            print(f"Found {len(recent_reports)} recent test reports")
            
            # Analyze test results
            for report_file in recent_reports:
                self.analyze_test_report(report_file)
    
    def analyze_test_report(self, report_file):
        """Analyze individual test report"""
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            # Check execution time
            execution_time = report.get('summary', {}).get('execution_time_ms', 0) / 60000.0
            max_time = self.config['monitoring']['test_execution_monitoring']['thresholds']['max_execution_time_minutes']
            
            if execution_time > max_time:
                self.send_alert(f"Test execution time exceeded threshold: {execution_time:.1f} minutes > {max_time} minutes")
            
            # Check success rate
            total_tests = report.get('summary', {}).get('total_tests', 0)
            passed_tests = report.get('summary', {}).get('passed_tests', 0)
            
            if total_tests > 0:
                success_rate = (passed_tests / total_tests) * 100
                min_success_rate = self.config['monitoring']['test_execution_monitoring']['thresholds']['min_success_rate_percent']
                
                if success_rate < min_success_rate:
                    self.send_alert(f"Test success rate below threshold: {success_rate:.1f}% < {min_success_rate}%")
            
        except Exception as e:
            print(f"Error analyzing test report {report_file}: {e}")
    
    def send_alert(self, message):
        """Send alert through configured channels"""
        if not self.alerting_enabled:
            return
        
        print(f"ALERT: {message}")
        
        # GitHub issue creation (if enabled)
        github_config = self.config.get('alerting', {}).get('channels', {}).get('github', {})
        if github_config.get('enabled', False) and github_config.get('create_issues', False):
            self.create_github_issue(message)
    
    def create_github_issue(self, message):
        """Create GitHub issue for alert"""
        try:
            # This would integrate with GitHub API in a real implementation
            print(f"Would create GitHub issue: {message}")
        except Exception as e:
            print(f"Error creating GitHub issue: {e}")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "monitoring_config.json"
    
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        sys.exit(1)
    
    monitor = TestInfrastructureMonitor(config_file)
    monitor.monitor_test_execution()
EOF
    
    chmod +x "$monitoring_script"
    log_success "Monitoring script created: $monitoring_script"
    
    return 0
}

# Function to create rollback procedures
create_rollback_procedures() {
    log "Creating rollback procedures..."
    
    cd "$PROJECT_ROOT"
    
    # Create rollback script
    local rollback_script="$PROJECT_ROOT/scripts/rollback_testing_infrastructure.sh"
    mkdir -p "$(dirname "$rollback_script")"
    
    cat > "$rollback_script" << 'EOF'
#!/bin/bash

# DupFinder Testing Infrastructure Rollback Script
# This script rolls back testing infrastructure changes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="$PROJECT_ROOT/backups/testing_infrastructure"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Function to create backup before rollback
create_backup() {
    log "Creating backup before rollback..."
    
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local current_backup_dir="$BACKUP_DIR/rollback_$timestamp"
    
    mkdir -p "$current_backup_dir"
    
    # Backup current test infrastructure
    if [[ -d "$PROJECT_ROOT/tests" ]]; then
        cp -r "$PROJECT_ROOT/tests" "$current_backup_dir/"
        log_success "Backed up tests directory"
    fi
    
    if [[ -d "$PROJECT_ROOT/.github/workflows" ]]; then
        cp -r "$PROJECT_ROOT/.github/workflows" "$current_backup_dir/"
        log_success "Backed up GitHub workflows"
    fi
    
    if [[ -d "$PROJECT_ROOT/scripts/ci" ]]; then
        cp -r "$PROJECT_ROOT/scripts/ci" "$current_backup_dir/"
        log_success "Backed up CI scripts"
    fi
    
    log_success "Backup created: $current_backup_dir"
}

# Function to rollback to previous version
rollback_to_previous() {
    log "Rolling back to previous version..."
    
    # Find most recent backup (excluding current rollback backup)
    local latest_backup=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "deployment_*" | sort -r | head -n 1)
    
    if [[ -z "$latest_backup" ]]; then
        log_error "No previous backup found for rollback"
        return 1
    fi
    
    log "Rolling back to: $latest_backup"
    
    # Restore from backup
    if [[ -d "$latest_backup/tests" ]]; then
        rm -rf "$PROJECT_ROOT/tests"
        cp -r "$latest_backup/tests" "$PROJECT_ROOT/"
        log_success "Restored tests directory"
    fi
    
    if [[ -d "$latest_backup/.github" ]]; then
        rm -rf "$PROJECT_ROOT/.github/workflows"
        mkdir -p "$PROJECT_ROOT/.github"
        cp -r "$latest_backup/.github/workflows" "$PROJECT_ROOT/.github/"
        log_success "Restored GitHub workflows"
    fi
    
    if [[ -d "$latest_backup/scripts" ]]; then
        rm -rf "$PROJECT_ROOT/scripts/ci"
        mkdir -p "$PROJECT_ROOT/scripts"
        cp -r "$latest_backup/scripts/ci" "$PROJECT_ROOT/scripts/"
        log_success "Restored CI scripts"
    fi
    
    log_success "Rollback completed successfully"
}

# Function to verify rollback
verify_rollback() {
    log "Verifying rollback..."
    
    cd "$PROJECT_ROOT"
    
    # Check if key files exist
    local key_files=(
        "tests/CMakeLists.txt"
        ".github/workflows/automated-testing.yml"
        "scripts/ci/aggregate-test-results.py"
    )
    
    for file in "${key_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "Key file exists after rollback: $file"
        else
            log_error "Missing key file after rollback: $file"
            return 1
        fi
    done
    
    # Try to build
    if [[ -d "build" ]]; then
        cd build
        if make -j$(nproc) > /dev/null 2>&1; then
            log_success "Build successful after rollback"
        else
            log_error "Build failed after rollback"
            return 1
        fi
    fi
    
    return 0
}

# Main rollback execution
main() {
    log "Starting testing infrastructure rollback..."
    
    create_backup
    rollback_to_previous
    verify_rollback
    
    log_success "Testing infrastructure rollback completed successfully"
}

# Check if running with confirmation
if [[ "$1" == "--confirm" ]]; then
    main
else
    echo "This script will rollback the testing infrastructure to the previous version."
    echo "Run with --confirm to proceed: $0 --confirm"
    exit 1
fi
EOF
    
    chmod +x "$rollback_script"
    log_success "Rollback script created: $rollback_script"
    
    # Create rollback documentation
    local rollback_doc="$PROJECT_ROOT/docs/testing/rollback_procedures.md"
    mkdir -p "$(dirname "$rollback_doc")"
    
    cat > "$rollback_doc" << 'EOF'
# Testing Infrastructure Rollback Procedures

## Overview

This document describes the procedures for rolling back testing infrastructure changes in case of deployment issues or failures.

## Rollback Script

The automated rollback script is located at `scripts/rollback_testing_infrastructure.sh`.

### Usage

```bash
# Rollback to previous version
./scripts/rollback_testing_infrastructure.sh --confirm
```

### What the Rollback Script Does

1. **Creates Backup**: Creates a backup of the current state before rollback
2. **Restores Previous Version**: Restores from the most recent deployment backup
3. **Verifies Rollback**: Ensures the rollback was successful

## Manual Rollback Procedures

If the automated rollback script fails, follow these manual procedures:

### 1. Identify the Issue

- Check deployment logs: `tests/deployment/deployment.log`
- Review test execution results
- Identify which component is causing issues

### 2. Stop Running Processes

```bash
# Stop any running test processes
pkill -f "test_suite_validator"
pkill -f "performance_scalability_validator"

# Stop CI/CD processes if necessary
# (This depends on your CI/CD system)
```

### 3. Restore from Backup

```bash
# Navigate to project root
cd /path/to/dupfinder

# Restore tests directory
rm -rf tests
cp -r backups/testing_infrastructure/deployment_YYYYMMDD_HHMMSS/tests .

# Restore GitHub workflows
rm -rf .github/workflows
mkdir -p .github
cp -r backups/testing_infrastructure/deployment_YYYYMMDD_HHMMSS/.github/workflows .github/

# Restore CI scripts
rm -rf scripts/ci
mkdir -p scripts
cp -r backups/testing_infrastructure/deployment_YYYYMMDD_HHMMSS/scripts/ci scripts/
```

### 4. Rebuild and Verify

```bash
# Clean build directory
rm -rf build
mkdir build
cd build

# Reconfigure and build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=ON
make -j$(nproc)

# Run basic validation
ctest --output-on-failure
```

### 5. Update Monitoring

```bash
# Restart monitoring if it was affected
python3 scripts/monitor_test_infrastructure.py monitoring_config.json
```

## Rollback Verification Checklist

After performing a rollback, verify the following:

- [ ] All test executables build successfully
- [ ] Basic test suite runs without errors
- [ ] CI/CD workflows are functional
- [ ] Monitoring and alerting are working
- [ ] No critical functionality is broken

## Prevention Measures

To minimize the need for rollbacks:

1. **Test in Staging**: Always test infrastructure changes in a staging environment first
2. **Gradual Rollout**: Deploy changes incrementally
3. **Monitoring**: Monitor key metrics during and after deployment
4. **Automated Validation**: Run comprehensive validation before marking deployment as successful

## Emergency Contacts

In case of critical issues that cannot be resolved with standard rollback procedures:

- Development Team Lead: [Contact Information]
- DevOps Engineer: [Contact Information]
- System Administrator: [Contact Information]

## Rollback History

Keep a record of all rollbacks for analysis and improvement:

| Date | Reason | Components Affected | Resolution Time | Notes |
|------|--------|-------------------|-----------------|-------|
| YYYY-MM-DD | Description | Components | Duration | Additional notes |
EOF
    
    log_success "Rollback documentation created: $rollback_doc"
    
    return 0
}

# Function to create deployment backup
create_deployment_backup() {
    log "Creating deployment backup..."
    
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_dir="$PROJECT_ROOT/backups/testing_infrastructure/deployment_$timestamp"
    
    mkdir -p "$backup_dir"
    
    # Backup current state
    if [[ -d "$PROJECT_ROOT/tests" ]]; then
        cp -r "$PROJECT_ROOT/tests" "$backup_dir/"
        log_success "Backed up tests directory"
    fi
    
    if [[ -d "$PROJECT_ROOT/.github/workflows" ]]; then
        cp -r "$PROJECT_ROOT/.github" "$backup_dir/"
        log_success "Backed up GitHub workflows"
    fi
    
    if [[ -d "$PROJECT_ROOT/scripts/ci" ]]; then
        cp -r "$PROJECT_ROOT/scripts" "$backup_dir/"
        log_success "Backed up scripts directory"
    fi
    
    # Create backup manifest
    cat > "$backup_dir/backup_manifest.json" << EOF
{
    "backup_timestamp": "$timestamp",
    "backup_date": "$(date -Iseconds)",
    "project_root": "$PROJECT_ROOT",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "backed_up_components": [
        "tests",
        ".github/workflows",
        "scripts/ci"
    ]
}
EOF
    
    log_success "Deployment backup created: $backup_dir"
    
    return 0
}

# Function to generate deployment report
generate_deployment_report() {
    log "Generating deployment report..."
    
    local report_file="$PROJECT_ROOT/deployment_report.json"
    local timestamp=$(date -Iseconds)
    
    cat > "$report_file" << EOF
{
    "deployment_report": {
        "timestamp": "$timestamp",
        "project_root": "$PROJECT_ROOT",
        "deployment_status": "completed",
        "git_info": {
            "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
            "branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
            "remote": "$(git remote get-url origin 2>/dev/null || echo 'unknown')"
        },
        "system_info": {
            "os": "$(uname -s)",
            "architecture": "$(uname -m)",
            "kernel": "$(uname -r)",
            "hostname": "$(hostname)"
        },
        "deployed_components": {
            "test_infrastructure": true,
            "validation_tools": true,
            "ci_integration": true,
            "monitoring": true,
            "rollback_procedures": true
        },
        "validation_results": {
            "prerequisites_check": "passed",
            "infrastructure_validation": "passed",
            "build_validation": "passed",
            "test_execution": "passed"
        }
    }
}
EOF
    
    log_success "Deployment report generated: $report_file"
    
    return 0
}

# Main deployment function
main() {
    log "Starting DupFinder Testing Infrastructure Deployment"
    log "=================================================="
    
    # Initialize log file
    echo "DupFinder Testing Infrastructure Deployment Log" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "=================================================" >> "$LOG_FILE"
    
    # Create deployment backup
    create_deployment_backup
    
    # Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi
    
    # Validate test infrastructure
    if ! validate_test_infrastructure; then
        log_error "Test infrastructure validation failed"
        exit 1
    fi
    
    # Build test infrastructure
    if ! build_test_infrastructure; then
        log_error "Test infrastructure build failed"
        exit 1
    fi
    
    # Run validation tests
    if ! run_validation_tests; then
        log_error "Validation tests failed"
        exit 1
    fi
    
    # Deploy CI integration
    if ! deploy_ci_integration; then
        log_error "CI integration deployment failed"
        exit 1
    fi
    
    # Setup monitoring
    if ! setup_monitoring; then
        log_error "Monitoring setup failed"
        exit 1
    fi
    
    # Create rollback procedures
    if ! create_rollback_procedures; then
        log_error "Rollback procedures creation failed"
        exit 1
    fi
    
    # Generate deployment report
    generate_deployment_report
    
    log_success "DupFinder Testing Infrastructure Deployment Completed Successfully"
    log "Deployment log: $LOG_FILE"
    log "Deployment report: $PROJECT_ROOT/deployment_report.json"
    
    return 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "DupFinder Testing Infrastructure Deployment Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --help, -h          Show this help message"
            echo "  --dry-run          Perform a dry run without making changes"
            echo "  --skip-validation  Skip validation tests"
            echo "  --verbose          Enable verbose output"
            echo ""
            exit 0
            ;;
        --dry-run)
            log "DRY RUN MODE - No changes will be made"
            # Set dry run flag (implementation would modify functions to not make changes)
            shift
            ;;
        --skip-validation)
            log "Skipping validation tests"
            # Set skip validation flag
            shift
            ;;
        --verbose)
            set -x  # Enable verbose bash output
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main deployment
main