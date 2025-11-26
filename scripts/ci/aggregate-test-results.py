#!/usr/bin/env python3
"""
Test Result Aggregation Script for CI/CD Pipeline

This script aggregates test results from multiple platforms and generates
comprehensive reports for the automated testing suite.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse


class TestResultAggregator:
    """Aggregates test results from multiple sources and platforms."""
    
    def __init__(self, artifacts_dir: str, output_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'summary': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'skipped_tests': 0,
                'execution_time': 0.0,
                'platforms': [],
                'categories': set()
            },
            'platforms': {},
            'failures': [],
            'performance_metrics': {}
        }
    
    def aggregate_junit_results(self) -> None:
        """Aggregate JUnit XML test results."""
        junit_files = list(self.artifacts_dir.rglob("*.xml"))
        
        for junit_file in junit_files:
            try:
                tree = ET.parse(junit_file)
                root = tree.getroot()
                
                # Extract platform from file path
                platform = self._extract_platform_from_path(junit_file)
                
                if platform not in self.results['platforms']:
                    self.results['platforms'][platform] = {
                        'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'time': 0.0
                    }
                
                # Parse testsuite or testsuites
                testsuites = root.findall('.//testsuite')
                if not testsuites and root.tag == 'testsuite':
                    testsuites = [root]
                
                for testsuite in testsuites:
                    tests = int(testsuite.get('tests', 0))
                    failures = int(testsuite.get('failures', 0))
                    errors = int(testsuite.get('errors', 0))
                    skipped = int(testsuite.get('skipped', 0))
                    time = float(testsuite.get('time', 0))
                    
                    # Update totals
                    self.results['summary']['total_tests'] += tests
                    self.results['summary']['failed_tests'] += failures + errors
                    self.results['summary']['skipped_tests'] += skipped
                    self.results['summary']['execution_time'] += time
                    
                    # Update platform totals
                    self.results['platforms'][platform]['total'] += tests
                    self.results['platforms'][platform]['failed'] += failures + errors
                    self.results['platforms'][platform]['skipped'] += skipped
                    self.results['platforms'][platform]['time'] += time
                    
                    # Extract category from testsuite name
                    category = self._extract_category_from_name(testsuite.get('name', ''))
                    if category:
                        self.results['summary']['categories'].add(category)
                    
                    # Collect failure details
                    for testcase in testsuite.findall('.//testcase'):
                        failure = testcase.find('failure')
                        error = testcase.find('error')
                        
                        if failure is not None or error is not None:
                            failure_info = {
                                'test_name': testcase.get('name'),
                                'class_name': testcase.get('classname'),
                                'platform': platform,
                                'category': category,
                                'message': (failure or error).get('message', ''),
                                'details': (failure or error).text or ''
                            }
                            self.results['failures'].append(failure_info)
                
            except ET.ParseError as e:
                print(f"Warning: Could not parse {junit_file}: {e}")
            except Exception as e:
                print(f"Error processing {junit_file}: {e}")
        
        # Calculate passed tests
        self.results['summary']['passed_tests'] = (
            self.results['summary']['total_tests'] - 
            self.results['summary']['failed_tests'] - 
            self.results['summary']['skipped_tests']
        )
        
        # Update platform passed counts
        for platform_data in self.results['platforms'].values():
            platform_data['passed'] = (
                platform_data['total'] - 
                platform_data['failed'] - 
                platform_data['skipped']
            )
        
        # Convert categories set to list
        self.results['summary']['categories'] = list(self.results['summary']['categories'])
        self.results['summary']['platforms'] = list(self.results['platforms'].keys())
    
    def aggregate_performance_metrics(self) -> None:
        """Aggregate performance test results."""
        perf_files = list(self.artifacts_dir.rglob("*performance*.json"))
        
        for perf_file in perf_files:
            try:
                with open(perf_file, 'r') as f:
                    perf_data = json.load(f)
                
                platform = self._extract_platform_from_path(perf_file)
                
                if platform not in self.results['performance_metrics']:
                    self.results['performance_metrics'][platform] = {}
                
                # Merge performance data
                self.results['performance_metrics'][platform].update(perf_data)
                
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not parse performance file {perf_file}: {e}")
    
    def generate_html_report(self) -> None:
        """Generate comprehensive HTML report."""
        html_content = self._generate_html_template()
        
        output_file = self.output_dir / "test-report.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {output_file}")
    
    def generate_json_report(self) -> None:
        """Generate JSON report for programmatic access."""
        # Add metadata
        report_data = {
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'generator': 'TestResultAggregator',
                'version': '1.0'
            },
            'results': self.results
        }
        
        output_file = self.output_dir / "test-results.json"
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"JSON report generated: {output_file}")
    
    def generate_markdown_summary(self) -> None:
        """Generate markdown summary for GitHub."""
        summary = self._generate_markdown_content()
        
        output_file = self.output_dir / "summary.md"
        with open(output_file, 'w') as f:
            f.write(summary)
        
        print(f"Markdown summary generated: {output_file}")
    
    def _extract_platform_from_path(self, file_path: Path) -> str:
        """Extract platform name from file path."""
        path_str = str(file_path)
        if 'ubuntu' in path_str or 'linux' in path_str:
            return 'linux'
        elif 'windows' in path_str:
            return 'windows'
        elif 'macos' in path_str:
            return 'macos'
        else:
            return 'unknown'
    
    def _extract_category_from_name(self, name: str) -> str:
        """Extract test category from test suite name."""
        name_lower = name.lower()
        if 'unit' in name_lower:
            return 'unit'
        elif 'integration' in name_lower:
            return 'integration'
        elif 'ui' in name_lower or 'visual' in name_lower:
            return 'ui'
        elif 'performance' in name_lower:
            return 'performance'
        elif 'e2e' in name_lower or 'end-to-end' in name_lower:
            return 'e2e'
        else:
            return 'other'
    
    def _generate_html_template(self) -> str:
        """Generate HTML report template."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CloneClean Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; }}
        .metric.passed {{ border-left: 4px solid #28a745; }}
        .metric.failed {{ border-left: 4px solid #dc3545; }}
        .metric.skipped {{ border-left: 4px solid #ffc107; }}
        .metric.total {{ border-left: 4px solid #007bff; }}
        .platforms {{ margin: 20px 0; }}
        .platform {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
        .failures {{ margin: 20px 0; }}
        .failure {{ margin: 10px 0; padding: 10px; background: #f8d7da; border-radius: 5px; }}
        .performance {{ margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CloneClean Automated Test Results</h1>
        <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        <p>Platforms: {', '.join(self.results['summary']['platforms'])}</p>
        <p>Categories: {', '.join(self.results['summary']['categories'])}</p>
    </div>
    
    <div class="summary">
        <div class="metric total">
            <h3>Total Tests</h3>
            <div style="font-size: 2em; font-weight: bold;">{self.results['summary']['total_tests']}</div>
        </div>
        <div class="metric passed">
            <h3>Passed</h3>
            <div style="font-size: 2em; font-weight: bold; color: #28a745;">{self.results['summary']['passed_tests']}</div>
        </div>
        <div class="metric failed">
            <h3>Failed</h3>
            <div style="font-size: 2em; font-weight: bold; color: #dc3545;">{self.results['summary']['failed_tests']}</div>
        </div>
        <div class="metric skipped">
            <h3>Skipped</h3>
            <div style="font-size: 2em; font-weight: bold; color: #ffc107;">{self.results['summary']['skipped_tests']}</div>
        </div>
    </div>
    
    <div class="platforms">
        <h2>Platform Results</h2>
        {self._generate_platform_html()}
    </div>
    
    {self._generate_failures_html()}
    
    {self._generate_performance_html()}
</body>
</html>"""
    
    def _generate_platform_html(self) -> str:
        """Generate HTML for platform results."""
        html = ""
        for platform, data in self.results['platforms'].items():
            success_rate = (data['passed'] / data['total'] * 100) if data['total'] > 0 else 0
            html += f"""
            <div class="platform">
                <h3>{platform.title()}</h3>
                <p>Total: {data['total']} | Passed: {data['passed']} | Failed: {data['failed']} | Skipped: {data['skipped']}</p>
                <p>Success Rate: {success_rate:.1f}% | Execution Time: {data['time']:.2f}s</p>
            </div>"""
        return html
    
    def _generate_failures_html(self) -> str:
        """Generate HTML for test failures."""
        if not self.results['failures']:
            return ""
        
        html = '<div class="failures"><h2>Test Failures</h2>'
        for failure in self.results['failures']:
            html += f"""
            <div class="failure">
                <h4>{failure['test_name']} ({failure['platform']})</h4>
                <p><strong>Category:</strong> {failure['category']}</p>
                <p><strong>Message:</strong> {failure['message']}</p>
                <pre>{failure['details']}</pre>
            </div>"""
        html += '</div>'
        return html
    
    def _generate_performance_html(self) -> str:
        """Generate HTML for performance metrics."""
        if not self.results['performance_metrics']:
            return ""
        
        html = '<div class="performance"><h2>Performance Metrics</h2>'
        for platform, metrics in self.results['performance_metrics'].items():
            html += f'<h3>{platform.title()}</h3><table>'
            html += '<tr><th>Metric</th><th>Value</th></tr>'
            for key, value in metrics.items():
                html += f'<tr><td>{key}</td><td>{value}</td></tr>'
            html += '</table>'
        html += '</div>'
        return html
    
    def _generate_markdown_content(self) -> str:
        """Generate markdown summary content."""
        success_rate = (self.results['summary']['passed_tests'] / 
                       self.results['summary']['total_tests'] * 100) if self.results['summary']['total_tests'] > 0 else 0
        
        status_emoji = "✅" if self.results['summary']['failed_tests'] == 0 else "❌"
        
        content = f"""# {status_emoji} Test Results Summary

## Overview
- **Total Tests:** {self.results['summary']['total_tests']}
- **Passed:** {self.results['summary']['passed_tests']} ({success_rate:.1f}%)
- **Failed:** {self.results['summary']['failed_tests']}
- **Skipped:** {self.results['summary']['skipped_tests']}
- **Execution Time:** {self.results['summary']['execution_time']:.2f}s

## Platform Results
"""
        
        for platform, data in self.results['platforms'].items():
            platform_success = (data['passed'] / data['total'] * 100) if data['total'] > 0 else 0
            platform_emoji = "✅" if data['failed'] == 0 else "❌"
            content += f"""
### {platform_emoji} {platform.title()}
- Tests: {data['total']} | Passed: {data['passed']} | Failed: {data['failed']} | Skipped: {data['skipped']}
- Success Rate: {platform_success:.1f}%
- Execution Time: {data['time']:.2f}s
"""
        
        if self.results['failures']:
            content += f"\n## ❌ Failed Tests ({len(self.results['failures'])})\n"
            for failure in self.results['failures'][:10]:  # Limit to first 10 failures
                content += f"- **{failure['test_name']}** ({failure['platform']}) - {failure['message']}\n"
            
            if len(self.results['failures']) > 10:
                content += f"- ... and {len(self.results['failures']) - 10} more failures\n"
        
        return content


def main():
    parser = argparse.ArgumentParser(description='Aggregate test results from CI/CD pipeline')
    parser.add_argument('--artifacts-dir', required=True, help='Directory containing test artifacts')
    parser.add_argument('--output-dir', required=True, help='Directory for output reports')
    parser.add_argument('--format', choices=['html', 'json', 'markdown', 'all'], 
                       default='all', help='Output format')
    
    args = parser.parse_args()
    
    aggregator = TestResultAggregator(args.artifacts_dir, args.output_dir)
    
    print("Aggregating JUnit results...")
    aggregator.aggregate_junit_results()
    
    print("Aggregating performance metrics...")
    aggregator.aggregate_performance_metrics()
    
    if args.format in ['html', 'all']:
        print("Generating HTML report...")
        aggregator.generate_html_report()
    
    if args.format in ['json', 'all']:
        print("Generating JSON report...")
        aggregator.generate_json_report()
    
    if args.format in ['markdown', 'all']:
        print("Generating Markdown summary...")
        aggregator.generate_markdown_summary()
    
    # Exit with error code if tests failed
    if aggregator.results['summary']['failed_tests'] > 0:
        print(f"\n❌ {aggregator.results['summary']['failed_tests']} tests failed")
        sys.exit(1)
    else:
        print(f"\n✅ All {aggregator.results['summary']['total_tests']} tests passed")


if __name__ == '__main__':
    main()