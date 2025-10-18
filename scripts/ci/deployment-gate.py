#!/usr/bin/env python3
"""
Deployment Gate Controller

This script implements deployment gates and controls based on quality metrics,
test results, and other criteria to ensure safe deployments.
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse
import requests


class DeploymentGateController:
    """Controls deployment gates based on quality metrics."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.db_path = self.config.get('database_path', './deployment-gates.db')
        self._init_database()
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load deployment gate configuration."""
        default_config = {
            'quality_gates': {
                'min_coverage_percentage': 85.0,
                'max_failed_tests': 0,
                'max_critical_issues': 0,
                'max_performance_regression': 10.0,  # percentage
                'require_security_scan': True,
                'require_manual_approval': False
            },
            'deployment_windows': {
                'allowed_days': ['monday', 'tuesday', 'wednesday', 'thursday'],
                'allowed_hours': {'start': 9, 'end': 17},  # 9 AM to 5 PM
                'timezone': 'UTC',
                'emergency_override': True
            },
            'rollback_criteria': {
                'error_rate_threshold': 5.0,  # percentage
                'response_time_threshold': 2000,  # milliseconds
                'monitoring_duration': 300  # seconds
            },
            'notifications': {
                'slack_webhook': '',
                'email_recipients': [],
                'github_create_issue': True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Deep merge with defaults
                    for key, value in user_config.items():
                        if key in default_config and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _init_database(self) -> None:
        """Initialize deployment tracking database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    build_number TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    status TEXT NOT NULL,
                    quality_score REAL DEFAULT 0,
                    coverage_percentage REAL DEFAULT 0,
                    failed_tests INTEGER DEFAULT 0,
                    critical_issues INTEGER DEFAULT 0,
                    performance_score REAL DEFAULT 0,
                    security_passed BOOLEAN DEFAULT FALSE,
                    manual_approval BOOLEAN DEFAULT FALSE,
                    approved_by TEXT,
                    deployed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deployment_approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_id INTEGER NOT NULL,
                    approver TEXT NOT NULL,
                    approval_type TEXT NOT NULL,
                    approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    comments TEXT,
                    FOREIGN KEY (deployment_id) REFERENCES deployments (id)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_deployments_build ON deployments(build_number)
            ''')
    
    def evaluate_deployment_readiness(self, build_info: Dict[str, Any], 
                                    test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if a build is ready for deployment."""
        evaluation = {
            'ready_for_deployment': False,
            'quality_score': 0.0,
            'gate_results': {},
            'blocking_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Extract metrics from test results
        summary = test_results.get('results', {}).get('summary', {})
        coverage_percentage = build_info.get('coverage_percentage', 0.0)
        failed_tests = summary.get('failed_tests', 0)
        
        # Evaluate each quality gate
        gates = self.config['quality_gates']
        
        # Coverage gate
        coverage_passed = coverage_percentage >= gates['min_coverage_percentage']
        evaluation['gate_results']['coverage'] = {
            'passed': coverage_passed,
            'current': coverage_percentage,
            'required': gates['min_coverage_percentage']
        }
        
        if not coverage_passed:
            evaluation['blocking_issues'].append(
                f"Coverage {coverage_percentage:.1f}% below minimum {gates['min_coverage_percentage']:.1f}%"
            )
        
        # Test failure gate
        test_passed = failed_tests <= gates['max_failed_tests']
        evaluation['gate_results']['tests'] = {
            'passed': test_passed,
            'failed_count': failed_tests,
            'max_allowed': gates['max_failed_tests']
        }
        
        if not test_passed:
            evaluation['blocking_issues'].append(
                f"{failed_tests} test failures (maximum allowed: {gates['max_failed_tests']})"
            )
        
        # Performance regression gate
        performance_regression = build_info.get('performance_regression', 0.0)
        performance_passed = performance_regression <= gates['max_performance_regression']
        evaluation['gate_results']['performance'] = {
            'passed': performance_passed,
            'regression_percentage': performance_regression,
            'max_allowed': gates['max_performance_regression']
        }
        
        if not performance_passed:
            evaluation['blocking_issues'].append(
                f"Performance regression {performance_regression:.1f}% exceeds limit {gates['max_performance_regression']:.1f}%"
            )
        
        # Security gate
        security_passed = build_info.get('security_scan_passed', False)
        if gates['require_security_scan']:
            evaluation['gate_results']['security'] = {
                'passed': security_passed,
                'required': True
            }
            
            if not security_passed:
                evaluation['blocking_issues'].append("Security scan required but not passed")
        
        # Manual approval gate
        manual_approval_required = gates['require_manual_approval']
        manual_approval_received = build_info.get('manual_approval', False)
        
        if manual_approval_required:
            evaluation['gate_results']['manual_approval'] = {
                'passed': manual_approval_received,
                'required': True
            }
            
            if not manual_approval_received:
                evaluation['blocking_issues'].append("Manual approval required")
        
        # Calculate overall quality score
        total_gates = len(evaluation['gate_results'])
        passed_gates = sum(1 for gate in evaluation['gate_results'].values() if gate['passed'])
        evaluation['quality_score'] = (passed_gates / total_gates * 100) if total_gates > 0 else 0
        
        # Check deployment window
        window_check = self._check_deployment_window()
        if not window_check['allowed']:
            evaluation['warnings'].append(f"Outside deployment window: {window_check['reason']}")
        
        # Determine overall readiness
        evaluation['ready_for_deployment'] = (
            len(evaluation['blocking_issues']) == 0 and
            (window_check['allowed'] or self.config['deployment_windows']['emergency_override'])
        )
        
        # Add recommendations
        if not evaluation['ready_for_deployment']:
            evaluation['recommendations'].extend([
                "Address all blocking issues before deployment",
                "Ensure all quality gates pass",
                "Consider deploying during allowed deployment window"
            ])
        
        return evaluation
    
    def _check_deployment_window(self) -> Dict[str, Any]:
        """Check if current time is within allowed deployment window."""
        now = datetime.utcnow()
        
        # Check day of week
        allowed_days = [day.lower() for day in self.config['deployment_windows']['allowed_days']]
        current_day = now.strftime('%A').lower()
        
        if current_day not in allowed_days:
            return {
                'allowed': False,
                'reason': f"Deployment not allowed on {current_day.title()}"
            }
        
        # Check time of day
        allowed_hours = self.config['deployment_windows']['allowed_hours']
        current_hour = now.hour
        
        if not (allowed_hours['start'] <= current_hour < allowed_hours['end']):
            return {
                'allowed': False,
                'reason': f"Deployment not allowed at {current_hour}:00 (allowed: {allowed_hours['start']}:00-{allowed_hours['end']}:00)"
            }
        
        return {'allowed': True, 'reason': 'Within deployment window'}
    
    def record_deployment_attempt(self, build_info: Dict[str, Any], 
                                evaluation: Dict[str, Any]) -> int:
        """Record a deployment attempt in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO deployments (
                    build_number, commit_sha, branch, environment, status,
                    quality_score, coverage_percentage, failed_tests,
                    critical_issues, performance_score, security_passed,
                    manual_approval, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                build_info.get('build_number', ''),
                build_info.get('commit_sha', ''),
                build_info.get('branch', ''),
                build_info.get('environment', 'production'),
                'approved' if evaluation['ready_for_deployment'] else 'blocked',
                evaluation['quality_score'],
                build_info.get('coverage_percentage', 0.0),
                evaluation['gate_results'].get('tests', {}).get('failed_count', 0),
                build_info.get('critical_issues', 0),
                build_info.get('performance_score', 0.0),
                evaluation['gate_results'].get('security', {}).get('passed', False),
                evaluation['gate_results'].get('manual_approval', {}).get('passed', False),
                json.dumps({**build_info, 'evaluation': evaluation})
            ))
            
            return cursor.lastrowid
    
    def approve_deployment(self, deployment_id: int, approver: str, 
                         approval_type: str = 'manual', comments: str = '') -> bool:
        """Record deployment approval."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Record approval
                conn.execute('''
                    INSERT INTO deployment_approvals (
                        deployment_id, approver, approval_type, comments
                    ) VALUES (?, ?, ?, ?)
                ''', (deployment_id, approver, approval_type, comments))
                
                # Update deployment record
                conn.execute('''
                    UPDATE deployments 
                    SET manual_approval = TRUE, approved_by = ?
                    WHERE id = ?
                ''', (approver, deployment_id))
                
                return True
        except Exception as e:
            print(f"Error recording approval: {e}")
            return False
    
    def get_deployment_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get deployment history for analysis."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM deployments
                WHERE created_at >= datetime('now', '-{} days')
                ORDER BY created_at DESC
            '''.format(days))
            
            columns = [description[0] for description in cursor.description]
            deployments = []
            
            for row in cursor.fetchall():
                deployment = dict(zip(columns, row))
                # Parse metadata
                try:
                    deployment['metadata'] = json.loads(deployment['metadata'] or '{}')
                except json.JSONDecodeError:
                    deployment['metadata'] = {}
                
                deployments.append(deployment)
            
            return deployments
    
    def generate_deployment_report(self, output_file: str) -> None:
        """Generate deployment analytics report."""
        history = self.get_deployment_history(90)  # 90 days
        
        # Calculate metrics
        total_deployments = len(history)
        successful_deployments = len([d for d in history if d['status'] == 'approved'])
        blocked_deployments = len([d for d in history if d['status'] == 'blocked'])
        
        success_rate = (successful_deployments / total_deployments * 100) if total_deployments > 0 else 0
        
        # Quality trends
        avg_quality_score = sum(d['quality_score'] for d in history) / len(history) if history else 0
        avg_coverage = sum(d['coverage_percentage'] for d in history) / len(history) if history else 0
        
        # Common blocking reasons
        blocking_reasons = {}
        for deployment in history:
            if deployment['status'] == 'blocked':
                metadata = deployment.get('metadata', {})
                evaluation = metadata.get('evaluation', {})
                for issue in evaluation.get('blocking_issues', []):
                    blocking_reasons[issue] = blocking_reasons.get(issue, 0) + 1
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'period_days': 90,
            'summary': {
                'total_deployments': total_deployments,
                'successful_deployments': successful_deployments,
                'blocked_deployments': blocked_deployments,
                'success_rate': success_rate
            },
            'quality_metrics': {
                'average_quality_score': avg_quality_score,
                'average_coverage': avg_coverage
            },
            'blocking_analysis': {
                'common_reasons': dict(sorted(blocking_reasons.items(), 
                                            key=lambda x: x[1], reverse=True)[:10])
            },
            'recommendations': self._generate_recommendations(history)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Deployment report generated: {output_file}")
    
    def _generate_recommendations(self, history: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on deployment history."""
        recommendations = []
        
        if not history:
            return ["No deployment history available for analysis"]
        
        # Analyze success rate
        success_rate = len([d for d in history if d['status'] == 'approved']) / len(history) * 100
        
        if success_rate < 80:
            recommendations.append("Low deployment success rate. Review quality gates and processes.")
        
        # Analyze coverage trends
        recent_coverage = [d['coverage_percentage'] for d in history[:10]]
        if recent_coverage and sum(recent_coverage) / len(recent_coverage) < 85:
            recommendations.append("Coverage trending below target. Focus on test coverage improvements.")
        
        # Analyze blocking patterns
        blocked_count = len([d for d in history if d['status'] == 'blocked'])
        if blocked_count > len(history) * 0.3:  # More than 30% blocked
            recommendations.append("High blocking rate. Consider adjusting quality gate thresholds.")
        
        return recommendations
    
    def send_deployment_notification(self, evaluation: Dict[str, Any], 
                                   build_info: Dict[str, Any]) -> None:
        """Send deployment status notification."""
        if not evaluation['ready_for_deployment']:
            self._send_blocked_notification(evaluation, build_info)
        else:
            self._send_approved_notification(evaluation, build_info)
    
    def _send_blocked_notification(self, evaluation: Dict[str, Any], 
                                 build_info: Dict[str, Any]) -> None:
        """Send notification for blocked deployment."""
        message = f"""
🚫 **Deployment Blocked**

**Build:** {build_info.get('build_number', 'Unknown')}
**Commit:** {build_info.get('commit_sha', 'Unknown')[:8]}
**Branch:** {build_info.get('branch', 'Unknown')}
**Quality Score:** {evaluation['quality_score']:.1f}%

**Blocking Issues:**
{chr(10).join(f"• {issue}" for issue in evaluation['blocking_issues'])}

**Gate Results:**
{chr(10).join(f"• {gate}: {'✅' if result['passed'] else '❌'}" 
              for gate, result in evaluation['gate_results'].items())}

Please address the blocking issues before attempting deployment.
"""
        
        self._send_notification(message, is_blocked=True)
    
    def _send_approved_notification(self, evaluation: Dict[str, Any], 
                                  build_info: Dict[str, Any]) -> None:
        """Send notification for approved deployment."""
        message = f"""
✅ **Deployment Approved**

**Build:** {build_info.get('build_number', 'Unknown')}
**Commit:** {build_info.get('commit_sha', 'Unknown')[:8]}
**Branch:** {build_info.get('branch', 'Unknown')}
**Quality Score:** {evaluation['quality_score']:.1f}%

All quality gates passed. Deployment can proceed.
"""
        
        self._send_notification(message, is_blocked=False)
    
    def _send_notification(self, message: str, is_blocked: bool) -> None:
        """Send notification via configured channels."""
        # Slack notification
        slack_webhook = self.config['notifications']['slack_webhook']
        if slack_webhook:
            try:
                payload = {
                    "text": message,
                    "color": "danger" if is_blocked else "good"
                }
                requests.post(slack_webhook, json=payload)
            except Exception as e:
                print(f"Failed to send Slack notification: {e}")
        
        # GitHub issue creation
        if self.config['notifications']['github_create_issue'] and is_blocked:
            # This would create a GitHub issue - implementation depends on environment
            print("GitHub issue creation would be triggered here")


def main():
    parser = argparse.ArgumentParser(description='Control deployment gates')
    parser.add_argument('action', choices=['evaluate', 'approve', 'history', 'report'])
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--build-info', help='Build information JSON file')
    parser.add_argument('--test-results', help='Test results JSON file')
    parser.add_argument('--deployment-id', type=int, help='Deployment ID for approval')
    parser.add_argument('--approver', help='Approver name')
    parser.add_argument('--comments', help='Approval comments')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--days', type=int, default=30, help='Number of days for history')
    
    args = parser.parse_args()
    
    controller = DeploymentGateController(args.config)
    
    if args.action == 'evaluate':
        if not args.build_info or not args.test_results:
            print("Error: --build-info and --test-results required for evaluate action")
            sys.exit(1)
        
        with open(args.build_info, 'r') as f:
            build_info = json.load(f)
        
        with open(args.test_results, 'r') as f:
            test_results = json.load(f)
        
        evaluation = controller.evaluate_deployment_readiness(build_info, test_results)
        deployment_id = controller.record_deployment_attempt(build_info, evaluation)
        
        evaluation['deployment_id'] = deployment_id
        
        print(json.dumps(evaluation, indent=2))
        
        # Send notifications
        controller.send_deployment_notification(evaluation, build_info)
        
        # Exit with error code if deployment is blocked
        if not evaluation['ready_for_deployment']:
            sys.exit(1)
    
    elif args.action == 'approve':
        if not args.deployment_id or not args.approver:
            print("Error: --deployment-id and --approver required for approve action")
            sys.exit(1)
        
        success = controller.approve_deployment(
            args.deployment_id, 
            args.approver, 
            comments=args.comments or ''
        )
        
        if success:
            print(f"Deployment {args.deployment_id} approved by {args.approver}")
        else:
            print(f"Failed to approve deployment {args.deployment_id}")
            sys.exit(1)
    
    elif args.action == 'history':
        history = controller.get_deployment_history(args.days)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(history, f, indent=2, default=str)
        else:
            print(json.dumps(history, indent=2, default=str))
    
    elif args.action == 'report':
        output_file = args.output or 'deployment-report.json'
        controller.generate_deployment_report(output_file)


if __name__ == '__main__':
    main()