#!/usr/bin/env python3
"""
Test Result Notification System

This script sends notifications about test results to various channels
including GitHub issues, Slack, email, etc.
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse


class TestNotificationSystem:
    """Handles notifications for test results."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load notification configuration."""
        default_config = {
            'github': {
                'enabled': True,
                'create_issues_on_failure': True,
                'labels': ['bug', 'ci-failure'],
                'assignees': []
            },
            'slack': {
                'enabled': False,
                'channel': '#ci-notifications',
                'notify_on_success': False,
                'notify_on_failure': True
            },
            'email': {
                'enabled': False,
                'recipients': [],
                'smtp_server': '',
                'smtp_port': 587
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    for key, value in user_config.items():
                        if key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        
        return default_config
    
    def notify_test_results(self, results_file: str, build_info: Dict[str, Any]) -> None:
        """Send notifications based on test results."""
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            
            test_results = results_data.get('results', {})
            summary = test_results.get('summary', {})
            
            has_failures = summary.get('failed_tests', 0) > 0
            
            if has_failures:
                self._notify_failure(test_results, build_info)
            else:
                self._notify_success(test_results, build_info)
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading results file {results_file}: {e}")
            sys.exit(1)
    
    def _notify_failure(self, results: Dict[str, Any], build_info: Dict[str, Any]) -> None:
        """Send failure notifications."""
        summary = results.get('summary', {})
        failures = results.get('failures', [])
        
        message = self._format_failure_message(summary, failures, build_info)
        
        if self.config['github']['enabled'] and self.config['github']['create_issues_on_failure']:
            self._create_github_issue(message, build_info)
        
        if self.config['slack']['enabled'] and self.config['slack']['notify_on_failure']:
            self._send_slack_notification(message, is_failure=True)
        
        if self.config['email']['enabled']:
            self._send_email_notification(message, "Test Failure", build_info)
    
    def _notify_success(self, results: Dict[str, Any], build_info: Dict[str, Any]) -> None:
        """Send success notifications."""
        summary = results.get('summary', {})
        
        if self.config['slack']['enabled'] and self.config['slack']['notify_on_success']:
            message = self._format_success_message(summary, build_info)
            self._send_slack_notification(message, is_failure=False)
    
    def _format_failure_message(self, summary: Dict[str, Any], failures: List[Dict], 
                               build_info: Dict[str, Any]) -> str:
        """Format failure notification message."""
        total_tests = summary.get('total_tests', 0)
        failed_tests = summary.get('failed_tests', 0)
        passed_tests = summary.get('passed_tests', 0)
        platforms = summary.get('platforms', [])
        
        message = f"""# 🚨 Test Failure Alert

## Build Information
- **Build Number:** {build_info.get('build_number', 'Unknown')}
- **Commit:** {build_info.get('commit_sha', 'Unknown')[:8]}
- **Branch:** {build_info.get('branch', 'Unknown')}
- **Trigger:** {build_info.get('trigger', 'Unknown')}
- **Platforms:** {', '.join(platforms)}

## Test Results Summary
- **Total Tests:** {total_tests}
- **Passed:** {passed_tests}
- **Failed:** {failed_tests}
- **Success Rate:** {(passed_tests / total_tests * 100):.1f}% if total_tests > 0 else 0

## Failed Tests
"""
        
        # Add top failures
        for i, failure in enumerate(failures[:5]):  # Limit to first 5 failures
            message += f"""
### {i+1}. {failure.get('test_name', 'Unknown Test')}
- **Platform:** {failure.get('platform', 'Unknown')}
- **Category:** {failure.get('category', 'Unknown')}
- **Error:** {failure.get('message', 'No message')[:200]}{'...' if len(failure.get('message', '')) > 200 else ''}
"""
        
        if len(failures) > 5:
            message += f"\n... and {len(failures) - 5} more failures"
        
        message += f"\n\n**View full results:** {build_info.get('workflow_url', 'N/A')}"
        
        return message
    
    def _format_success_message(self, summary: Dict[str, Any], build_info: Dict[str, Any]) -> str:
        """Format success notification message."""
        total_tests = summary.get('total_tests', 0)
        execution_time = summary.get('execution_time', 0)
        platforms = summary.get('platforms', [])
        
        return f"""# ✅ All Tests Passed!

**Build:** {build_info.get('build_number', 'Unknown')} | **Commit:** {build_info.get('commit_sha', 'Unknown')[:8]}
**Tests:** {total_tests} | **Time:** {execution_time:.1f}s | **Platforms:** {', '.join(platforms)}
"""
    
    def _create_github_issue(self, message: str, build_info: Dict[str, Any]) -> None:
        """Create GitHub issue for test failure."""
        if not self.github_token:
            print("Warning: GITHUB_TOKEN not set, skipping GitHub issue creation")
            return
        
        repo_owner = build_info.get('repo_owner')
        repo_name = build_info.get('repo_name')
        
        if not repo_owner or not repo_name:
            print("Warning: Repository information not available, skipping GitHub issue creation")
            return
        
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
        
        title = f"CI Test Failure - Build #{build_info.get('build_number', 'Unknown')}"
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        data = {
            'title': title,
            'body': message,
            'labels': self.config['github']['labels'],
            'assignees': self.config['github']['assignees']
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            issue_data = response.json()
            print(f"Created GitHub issue: {issue_data['html_url']}")
            
        except requests.RequestException as e:
            print(f"Error creating GitHub issue: {e}")
    
    def _send_slack_notification(self, message: str, is_failure: bool) -> None:
        """Send Slack notification."""
        if not self.slack_webhook:
            print("Warning: SLACK_WEBHOOK_URL not set, skipping Slack notification")
            return
        
        # Convert markdown to Slack format
        slack_message = self._markdown_to_slack(message)
        
        color = "danger" if is_failure else "good"
        icon = ":x:" if is_failure else ":white_check_mark:"
        
        payload = {
            "channel": self.config['slack']['channel'],
            "username": "CI Bot",
            "icon_emoji": icon,
            "attachments": [
                {
                    "color": color,
                    "text": slack_message,
                    "mrkdwn_in": ["text"]
                }
            ]
        }
        
        try:
            response = requests.post(self.slack_webhook, json=payload)
            response.raise_for_status()
            print("Sent Slack notification")
            
        except requests.RequestException as e:
            print(f"Error sending Slack notification: {e}")
    
    def _send_email_notification(self, message: str, subject: str, build_info: Dict[str, Any]) -> None:
        """Send email notification."""
        # Email implementation would go here
        # This is a placeholder for email functionality
        print("Email notifications not implemented yet")
    
    def _markdown_to_slack(self, markdown: str) -> str:
        """Convert basic markdown to Slack format."""
        # Simple conversion - could be enhanced
        slack_text = markdown
        slack_text = slack_text.replace('# ', '*')
        slack_text = slack_text.replace('## ', '*')
        slack_text = slack_text.replace('### ', '*')
        slack_text = slack_text.replace('**', '*')
        return slack_text


def main():
    parser = argparse.ArgumentParser(description='Send test result notifications')
    parser.add_argument('--results-file', required=True, help='Path to test results JSON file')
    parser.add_argument('--config-file', help='Path to notification config file')
    parser.add_argument('--build-number', help='Build number')
    parser.add_argument('--commit-sha', help='Commit SHA')
    parser.add_argument('--branch', help='Branch name')
    parser.add_argument('--trigger', help='Build trigger (push, pr, schedule)')
    parser.add_argument('--repo-owner', help='Repository owner')
    parser.add_argument('--repo-name', help='Repository name')
    parser.add_argument('--workflow-url', help='Workflow run URL')
    
    args = parser.parse_args()
    
    build_info = {
        'build_number': args.build_number or os.getenv('GITHUB_RUN_NUMBER'),
        'commit_sha': args.commit_sha or os.getenv('GITHUB_SHA'),
        'branch': args.branch or os.getenv('GITHUB_REF_NAME'),
        'trigger': args.trigger or os.getenv('GITHUB_EVENT_NAME'),
        'repo_owner': args.repo_owner or os.getenv('GITHUB_REPOSITORY_OWNER'),
        'repo_name': args.repo_name or os.getenv('GITHUB_REPOSITORY', '').split('/')[-1],
        'workflow_url': args.workflow_url or f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
    }
    
    notifier = TestNotificationSystem(args.config_file)
    notifier.notify_test_results(args.results_file, build_info)


if __name__ == '__main__':
    main()