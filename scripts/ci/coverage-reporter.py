#!/usr/bin/env python3
"""
Coverage Report Generator

This script generates comprehensive coverage reports with historical tracking,
trend analysis, and integration with various coverage tools.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse
import sqlite3
import subprocess


class CoverageReporter:
    """Generates and manages coverage reports."""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize coverage database
        self.db_path = str(self.storage_dir / "coverage.db")
        self._init_database()
        
        # Coverage thresholds
        self.thresholds = {
            'line_coverage': 85.0,
            'branch_coverage': 80.0,
            'function_coverage': 90.0,
            'warning_threshold': 75.0,
            'critical_threshold': 60.0
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for coverage tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS coverage_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    build_number TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    line_coverage REAL NOT NULL,
                    branch_coverage REAL DEFAULT 0,
                    function_coverage REAL DEFAULT 0,
                    lines_covered INTEGER DEFAULT 0,
                    lines_total INTEGER DEFAULT 0,
                    branches_covered INTEGER DEFAULT 0,
                    branches_total INTEGER DEFAULT 0,
                    functions_covered INTEGER DEFAULT 0,
                    functions_total INTEGER DEFAULT 0,
                    report_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS file_coverage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    line_coverage REAL NOT NULL,
                    branch_coverage REAL DEFAULT 0,
                    lines_covered INTEGER DEFAULT 0,
                    lines_total INTEGER DEFAULT 0,
                    branches_covered INTEGER DEFAULT 0,
                    branches_total INTEGER DEFAULT 0,
                    FOREIGN KEY (report_id) REFERENCES coverage_reports (id)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_coverage_build ON coverage_reports(build_number)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_coverage_branch ON coverage_reports(branch)
            ''')
    
    def generate_coverage_report(self, build_root: str, source_root: str, build_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate coverage report using gcovr."""
        try:
            # Generate XML report for parsing
            xml_report_path = self.storage_dir / f"coverage-{build_info['build_number']}.xml"
            html_report_path = self.storage_dir / f"coverage-{build_info['build_number']}.html"
            
            # Run gcovr to generate reports
            gcovr_cmd = [
                'gcovr',
                '--root', source_root,
                '--build-root', build_root,
                '--exclude-unreachable-branches',
                '--exclude-throw-branches',
                '--exclude', 'tests/.*',
                '--exclude', 'third_party/.*',
                '--xml', str(xml_report_path),
                '--html-details', str(html_report_path),
                '--print-summary'
            ]
            
            result = subprocess.run(gcovr_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error running gcovr: {result.stderr}")
                return {}
            
            # Parse XML report
            coverage_data = self._parse_xml_coverage(xml_report_path)
            coverage_data['html_report_path'] = str(html_report_path)
            coverage_data['xml_report_path'] = str(xml_report_path)
            
            # Store in database
            self._store_coverage_data(coverage_data, build_info)
            
            return coverage_data
            
        except Exception as e:
            print(f"Error generating coverage report: {e}")
            return {}
    
    def _parse_xml_coverage(self, xml_path: Path) -> Dict[str, Any]:
        """Parse XML coverage report."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            coverage_data = {
                'line_coverage': 0.0,
                'branch_coverage': 0.0,
                'function_coverage': 0.0,
                'lines_covered': 0,
                'lines_total': 0,
                'branches_covered': 0,
                'branches_total': 0,
                'functions_covered': 0,
                'functions_total': 0,
                'files': []
            }
            
            # Parse overall coverage
            for coverage_elem in root.findall('.//coverage'):
                line_rate = float(coverage_elem.get('line-rate', 0))
                branch_rate = float(coverage_elem.get('branch-rate', 0))
                
                coverage_data['line_coverage'] = line_rate * 100
                coverage_data['branch_coverage'] = branch_rate * 100
            
            # Parse file-level coverage
            for package in root.findall('.//package'):
                for class_elem in package.findall('.//class'):
                    filename = class_elem.get('filename', '')
                    line_rate = float(class_elem.get('line-rate', 0))
                    branch_rate = float(class_elem.get('branch-rate', 0))
                    
                    # Count lines and branches
                    lines = class_elem.findall('.//line')
                    lines_total = len(lines)
                    lines_covered = sum(1 for line in lines if int(line.get('hits', 0)) > 0)
                    
                    branches_total = sum(1 for line in lines if line.get('branch') == 'true')
                    branches_covered = sum(1 for line in lines 
                                         if line.get('branch') == 'true' and 
                                         line.get('condition-coverage', '0%').rstrip('%') != '0')
                    
                    file_data = {
                        'file_path': filename,
                        'line_coverage': line_rate * 100,
                        'branch_coverage': branch_rate * 100,
                        'lines_covered': lines_covered,
                        'lines_total': lines_total,
                        'branches_covered': branches_covered,
                        'branches_total': branches_total
                    }
                    
                    coverage_data['files'].append(file_data)
                    
                    # Update totals
                    coverage_data['lines_covered'] += lines_covered
                    coverage_data['lines_total'] += lines_total
                    coverage_data['branches_covered'] += branches_covered
                    coverage_data['branches_total'] += branches_total
            
            return coverage_data
            
        except Exception as e:
            print(f"Error parsing XML coverage: {e}")
            return {}
    
    def _store_coverage_data(self, coverage_data: Dict[str, Any], build_info: Dict[str, Any]) -> int:
        """Store coverage data in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO coverage_reports (
                    build_number, commit_sha, branch, platform,
                    line_coverage, branch_coverage, function_coverage,
                    lines_covered, lines_total, branches_covered, branches_total,
                    functions_covered, functions_total, report_path, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                build_info.get('build_number', ''),
                build_info.get('commit_sha', ''),
                build_info.get('branch', ''),
                build_info.get('platform', ''),
                coverage_data.get('line_coverage', 0),
                coverage_data.get('branch_coverage', 0),
                coverage_data.get('function_coverage', 0),
                coverage_data.get('lines_covered', 0),
                coverage_data.get('lines_total', 0),
                coverage_data.get('branches_covered', 0),
                coverage_data.get('branches_total', 0),
                coverage_data.get('functions_covered', 0),
                coverage_data.get('functions_total', 0),
                coverage_data.get('html_report_path', ''),
                json.dumps(build_info)
            ))
            
            report_id = cursor.lastrowid
            
            # Store file-level coverage
            for file_data in coverage_data.get('files', []):
                conn.execute('''
                    INSERT INTO file_coverage (
                        report_id, file_path, line_coverage, branch_coverage,
                        lines_covered, lines_total, branches_covered, branches_total
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    report_id,
                    file_data['file_path'],
                    file_data['line_coverage'],
                    file_data['branch_coverage'],
                    file_data['lines_covered'],
                    file_data['lines_total'],
                    file_data['branches_covered'],
                    file_data['branches_total']
                ))
            
            return report_id
    
    def generate_coverage_trend(self, branch: str = 'main', days: int = 30) -> Dict[str, Any]:
        """Generate coverage trend analysis."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT build_number, commit_sha, line_coverage, branch_coverage,
                       lines_covered, lines_total, created_at
                FROM coverage_reports
                WHERE branch = ? AND created_at >= datetime('now', '-{} days')
                ORDER BY created_at DESC
            '''.format(days), (branch,))
            
            rows = cursor.fetchall()
            
            if not rows:
                return {'error': f'No coverage data found for branch {branch}'}
            
            trend_data = {
                'branch': branch,
                'period_days': days,
                'builds': [],
                'statistics': {
                    'current_coverage': rows[0][2],
                    'average_coverage': 0,
                    'min_coverage': min(row[2] for row in rows),
                    'max_coverage': max(row[2] for row in rows),
                    'trend': 'stable'
                },
                'threshold_analysis': self._analyze_thresholds(rows[0][2])
            }
            
            # Build historical data
            for row in rows:
                build_data = {
                    'build_number': row[0],
                    'commit_sha': row[1][:8],
                    'line_coverage': row[2],
                    'branch_coverage': row[3],
                    'lines_covered': row[4],
                    'lines_total': row[5],
                    'date': row[6]
                }
                trend_data['builds'].append(build_data)
            
            # Calculate statistics
            total_coverage = sum(row[2] for row in rows)
            trend_data['statistics']['average_coverage'] = total_coverage / len(rows)
            
            # Determine trend
            if len(rows) >= 5:
                recent_avg = sum(row[2] for row in rows[:5]) / 5
                older_avg = sum(row[2] for row in rows[-5:]) / 5
                
                if recent_avg > older_avg + 2:
                    trend_data['statistics']['trend'] = 'improving'
                elif recent_avg < older_avg - 2:
                    trend_data['statistics']['trend'] = 'declining'
            
            return trend_data
    
    def _analyze_thresholds(self, current_coverage: float) -> Dict[str, Any]:
        """Analyze coverage against thresholds."""
        analysis = {
            'status': 'excellent',
            'meets_target': current_coverage >= self.thresholds['line_coverage'],
            'target_coverage': self.thresholds['line_coverage'],
            'gap_to_target': max(0, self.thresholds['line_coverage'] - current_coverage),
            'recommendations': []
        }
        
        if current_coverage >= self.thresholds['line_coverage']:
            analysis['status'] = 'excellent'
            analysis['recommendations'].append('Coverage meets target. Consider increasing target.')
        elif current_coverage >= self.thresholds['warning_threshold']:
            analysis['status'] = 'good'
            analysis['recommendations'].append(f'Close to target. Need {analysis["gap_to_target"]:.1f}% more coverage.')
        elif current_coverage >= self.thresholds['critical_threshold']:
            analysis['status'] = 'warning'
            analysis['recommendations'].append('Coverage below target. Review untested code.')
        else:
            analysis['status'] = 'critical'
            analysis['recommendations'].append('Coverage critically low. Immediate action required.')
        
        return analysis
    
    def generate_coverage_comparison(self, build1: str, build2: str) -> Dict[str, Any]:
        """Compare coverage between two builds."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM coverage_reports WHERE build_number IN (?, ?)
            ''', (build1, build2))
            
            builds = {row[1]: dict(zip([col[0] for col in cursor.description], row)) 
                     for row in cursor.fetchall()}
            
            if len(builds) != 2:
                return {'error': 'One or both builds not found'}
            
            build1_data = builds[build1]
            build2_data = builds[build2]
            
            comparison = {
                'build1': build1_data,
                'build2': build2_data,
                'changes': {
                    'line_coverage_change': build2_data['line_coverage'] - build1_data['line_coverage'],
                    'branch_coverage_change': build2_data['branch_coverage'] - build1_data['branch_coverage'],
                    'lines_added': build2_data['lines_total'] - build1_data['lines_total'],
                    'coverage_delta_per_line': 0
                },
                'analysis': {
                    'improved': build2_data['line_coverage'] > build1_data['line_coverage'],
                    'significant_change': abs(build2_data['line_coverage'] - build1_data['line_coverage']) > 1.0
                }
            }
            
            # Calculate coverage delta per line
            if comparison['changes']['lines_added'] != 0:
                comparison['changes']['coverage_delta_per_line'] = (
                    comparison['changes']['line_coverage_change'] / comparison['changes']['lines_added']
                )
            
            return comparison
    
    def identify_uncovered_files(self, build_number: str, threshold: float = 80.0) -> List[Dict[str, Any]]:
        """Identify files with coverage below threshold."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT cr.id FROM coverage_reports cr WHERE cr.build_number = ?
            ''', (build_number,))
            
            report_row = cursor.fetchone()
            if not report_row:
                return []
            
            report_id = report_row[0]
            
            cursor = conn.execute('''
                SELECT file_path, line_coverage, branch_coverage, 
                       lines_covered, lines_total, branches_covered, branches_total
                FROM file_coverage
                WHERE report_id = ? AND line_coverage < ?
                ORDER BY line_coverage ASC
            ''', (report_id, threshold))
            
            uncovered_files = []
            for row in cursor.fetchall():
                file_data = {
                    'file_path': row[0],
                    'line_coverage': row[1],
                    'branch_coverage': row[2],
                    'lines_covered': row[3],
                    'lines_total': row[4],
                    'branches_covered': row[5],
                    'branches_total': row[6],
                    'priority': self._calculate_priority(row[1], row[4])
                }
                uncovered_files.append(file_data)
            
            return uncovered_files
    
    def _calculate_priority(self, coverage: float, total_lines: int) -> str:
        """Calculate priority for improving file coverage."""
        if coverage < 50 and total_lines > 100:
            return 'high'
        elif coverage < 70 and total_lines > 50:
            return 'medium'
        else:
            return 'low'
    
    def generate_html_dashboard(self, output_path: str, branch: str = 'main') -> None:
        """Generate HTML coverage dashboard."""
        trend_data = self.generate_coverage_trend(branch)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coverage Dashboard - {branch}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; text-align: center; }}
        .metric.excellent {{ border-left: 4px solid #28a745; }}
        .metric.good {{ border-left: 4px solid #17a2b8; }}
        .metric.warning {{ border-left: 4px solid #ffc107; }}
        .metric.critical {{ border-left: 4px solid #dc3545; }}
        .chart-container {{ width: 100%; height: 400px; margin: 20px 0; }}
        .recommendations {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Coverage Dashboard - {branch}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="metrics">
        <div class="metric {trend_data['threshold_analysis']['status']}">
            <h3>Current Coverage</h3>
            <div style="font-size: 2em; font-weight: bold;">{trend_data['statistics']['current_coverage']:.1f}%</div>
        </div>
        <div class="metric">
            <h3>Target</h3>
            <div style="font-size: 2em; font-weight: bold;">{self.thresholds['line_coverage']:.1f}%</div>
        </div>
        <div class="metric">
            <h3>Trend</h3>
            <div style="font-size: 1.5em; font-weight: bold;">{trend_data['statistics']['trend'].title()}</div>
        </div>
        <div class="metric">
            <h3>Gap to Target</h3>
            <div style="font-size: 1.5em; font-weight: bold;">{trend_data['threshold_analysis']['gap_to_target']:.1f}%</div>
        </div>
    </div>
    
    <div class="chart-container">
        <canvas id="coverageChart"></canvas>
    </div>
    
    <div class="recommendations">
        <h3>Recommendations</h3>
        <ul>
            {''.join(f'<li>{rec}</li>' for rec in trend_data['threshold_analysis']['recommendations'])}
        </ul>
    </div>
    
    <script>
        const ctx = document.getElementById('coverageChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps([build['date'][:10] for build in reversed(trend_data['builds'])])},
                datasets: [{{
                    label: 'Line Coverage',
                    data: {json.dumps([build['line_coverage'] for build in reversed(trend_data['builds'])])},
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }}, {{
                    label: 'Target',
                    data: Array({len(trend_data['builds'])}).fill({self.thresholds['line_coverage']}),
                    borderColor: 'rgb(255, 99, 132)',
                    borderDash: [5, 5]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Coverage dashboard generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate coverage reports and analysis')
    parser.add_argument('action', choices=['generate', 'trend', 'compare', 'dashboard', 'uncovered'])
    parser.add_argument('--storage-dir', default='./coverage-reports', help='Coverage storage directory')
    parser.add_argument('--build-root', help='Build root directory')
    parser.add_argument('--source-root', help='Source root directory')
    parser.add_argument('--build-number', help='Build number')
    parser.add_argument('--commit-sha', help='Commit SHA')
    parser.add_argument('--branch', default='main', help='Branch name')
    parser.add_argument('--platform', help='Platform name')
    parser.add_argument('--build1', help='First build for comparison')
    parser.add_argument('--build2', help='Second build for comparison')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--threshold', type=float, default=80.0, help='Coverage threshold')
    
    args = parser.parse_args()
    
    reporter = CoverageReporter(args.storage_dir)
    
    if args.action == 'generate':
        if not args.build_root or not args.source_root:
            print("Error: --build-root and --source-root required for generate action")
            sys.exit(1)
        
        build_info = {
            'build_number': args.build_number or os.getenv('GITHUB_RUN_NUMBER', 'unknown'),
            'commit_sha': args.commit_sha or os.getenv('GITHUB_SHA', 'unknown'),
            'branch': args.branch,
            'platform': args.platform or 'unknown'
        }
        
        coverage_data = reporter.generate_coverage_report(args.build_root, args.source_root, build_info)
        print(json.dumps(coverage_data, indent=2))
    
    elif args.action == 'trend':
        trend_data = reporter.generate_coverage_trend(args.branch)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(trend_data, f, indent=2, default=str)
        else:
            print(json.dumps(trend_data, indent=2, default=str))
    
    elif args.action == 'compare':
        if not args.build1 or not args.build2:
            print("Error: --build1 and --build2 required for compare action")
            sys.exit(1)
        
        comparison = reporter.generate_coverage_comparison(args.build1, args.build2)
        print(json.dumps(comparison, indent=2, default=str))
    
    elif args.action == 'dashboard':
        output_file = args.output or 'coverage-dashboard.html'
        reporter.generate_html_dashboard(output_file, args.branch)
    
    elif args.action == 'uncovered':
        if not args.build_number:
            print("Error: --build-number required for uncovered action")
            sys.exit(1)
        
        uncovered = reporter.identify_uncovered_files(args.build_number, args.threshold)
        print(json.dumps(uncovered, indent=2))


if __name__ == '__main__':
    main()