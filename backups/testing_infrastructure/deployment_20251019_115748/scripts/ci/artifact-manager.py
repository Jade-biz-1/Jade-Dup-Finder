#!/usr/bin/env python3
"""
Test Artifact Management System

This script manages test artifacts including screenshots, logs, coverage reports,
and performance metrics. It handles collection, storage, and historical tracking.
"""

import os
import sys
import json
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse
import hashlib
import zipfile


class TestArtifactManager:
    """Manages test artifacts and their storage."""
    
    def __init__(self, storage_dir: str, database_path: Optional[str] = None):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = database_path or str(self.storage_dir / "artifacts.db")
        self._init_database()
        
        # Artifact categories
        self.artifact_types = {
            'screenshots': {'extensions': ['.png', '.jpg', '.jpeg'], 'retention_days': 30},
            'logs': {'extensions': ['.log', '.txt'], 'retention_days': 14},
            'coverage': {'extensions': ['.html', '.xml', '.json'], 'retention_days': 90},
            'performance': {'extensions': ['.json', '.csv'], 'retention_days': 180},
            'reports': {'extensions': ['.html', '.pdf', '.json'], 'retention_days': 90},
            'videos': {'extensions': ['.mp4', '.avi', '.webm'], 'retention_days': 7}
        }
    
    def _init_database(self) -> None:
        """Initialize SQLite database for artifact tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    build_number TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS build_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    build_number TEXT NOT NULL UNIQUE,
                    commit_sha TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    total_tests INTEGER DEFAULT 0,
                    passed_tests INTEGER DEFAULT 0,
                    failed_tests INTEGER DEFAULT 0,
                    execution_time REAL DEFAULT 0,
                    coverage_percentage REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_artifacts_build ON artifacts(build_number)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created_at)
            ''')
    
    def collect_artifacts(self, source_dir: str, build_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Collect and organize artifacts from source directory."""
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Warning: Source directory {source_dir} does not exist")
            return {}
        
        collected = {}
        
        # Create build-specific directory
        build_dir = self._get_build_directory(build_info)
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect artifacts by type
        for artifact_type, config in self.artifact_types.items():
            collected[artifact_type] = []
            
            for ext in config['extensions']:
                for file_path in source_path.rglob(f"*{ext}"):
                    if file_path.is_file():
                        dest_path = self._store_artifact(file_path, artifact_type, build_info)
                        if dest_path:
                            collected[artifact_type].append(str(dest_path))
        
        # Store build summary
        self._store_build_summary(build_info, collected)
        
        return collected
    
    def _get_build_directory(self, build_info: Dict[str, Any]) -> Path:
        """Get build-specific directory path."""
        build_number = build_info.get('build_number', 'unknown')
        commit_sha = build_info.get('commit_sha', 'unknown')[:8]
        platform = build_info.get('platform', 'unknown')
        
        return self.storage_dir / f"build-{build_number}" / f"{commit_sha}-{platform}"
    
    def _store_artifact(self, source_path: Path, artifact_type: str, build_info: Dict[str, Any]) -> Optional[Path]:
        """Store individual artifact and record in database."""
        try:
            build_dir = self._get_build_directory(build_info)
            type_dir = build_dir / artifact_type
            type_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename to avoid conflicts
            timestamp = datetime.now().strftime("%H%M%S")
            dest_name = f"{timestamp}_{source_path.name}"
            dest_path = type_dir / dest_name
            
            # Copy file
            shutil.copy2(source_path, dest_path)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(dest_path)
            
            # Record in database
            self._record_artifact(dest_path, artifact_type, build_info, file_hash)
            
            return dest_path
            
        except Exception as e:
            print(f"Error storing artifact {source_path}: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _record_artifact(self, file_path: Path, artifact_type: str, build_info: Dict[str, Any], file_hash: str) -> None:
        """Record artifact in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO artifacts (
                    build_number, commit_sha, branch, platform, artifact_type,
                    file_path, file_size, file_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                build_info.get('build_number', ''),
                build_info.get('commit_sha', ''),
                build_info.get('branch', ''),
                build_info.get('platform', ''),
                artifact_type,
                str(file_path),
                file_path.stat().st_size,
                file_hash,
                json.dumps(build_info)
            ))
    
    def _store_build_summary(self, build_info: Dict[str, Any], artifacts: Dict[str, List[str]]) -> None:
        """Store build summary information."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO build_summary (
                    build_number, commit_sha, branch, total_tests, passed_tests,
                    failed_tests, execution_time, coverage_percentage, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                build_info.get('build_number', ''),
                build_info.get('commit_sha', ''),
                build_info.get('branch', ''),
                build_info.get('total_tests', 0),
                build_info.get('passed_tests', 0),
                build_info.get('failed_tests', 0),
                build_info.get('execution_time', 0.0),
                build_info.get('coverage_percentage', 0.0),
                json.dumps({**build_info, 'artifacts': artifacts})
            ))
    
    def generate_coverage_trend(self, days: int = 30) -> Dict[str, Any]:
        """Generate coverage trend data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT build_number, commit_sha, branch, coverage_percentage, created_at
                FROM build_summary
                WHERE created_at >= datetime('now', '-{} days')
                ORDER BY created_at DESC
            '''.format(days))
            
            rows = cursor.fetchall()
            
            trend_data = {
                'period_days': days,
                'builds': [],
                'average_coverage': 0.0,
                'coverage_trend': 'stable'  # stable, improving, declining
            }
            
            total_coverage = 0
            for row in rows:
                build_data = {
                    'build_number': row[0],
                    'commit_sha': row[1][:8],
                    'branch': row[2],
                    'coverage': row[3],
                    'date': row[4]
                }
                trend_data['builds'].append(build_data)
                total_coverage += row[3]
            
            if rows:
                trend_data['average_coverage'] = total_coverage / len(rows)
                
                # Determine trend
                if len(rows) >= 5:
                    recent_avg = sum(row[3] for row in rows[:5]) / 5
                    older_avg = sum(row[3] for row in rows[-5:]) / 5
                    
                    if recent_avg > older_avg + 1:
                        trend_data['coverage_trend'] = 'improving'
                    elif recent_avg < older_avg - 1:
                        trend_data['coverage_trend'] = 'declining'
            
            return trend_data
    
    def generate_performance_trend(self, days: int = 30) -> Dict[str, Any]:
        """Generate performance trend data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT build_number, commit_sha, branch, execution_time, total_tests, created_at
                FROM build_summary
                WHERE created_at >= datetime('now', '-{} days')
                ORDER BY created_at DESC
            '''.format(days))
            
            rows = cursor.fetchall()
            
            trend_data = {
                'period_days': days,
                'builds': [],
                'average_execution_time': 0.0,
                'performance_trend': 'stable'
            }
            
            total_time = 0
            for row in rows:
                build_data = {
                    'build_number': row[0],
                    'commit_sha': row[1][:8],
                    'branch': row[2],
                    'execution_time': row[3],
                    'total_tests': row[4],
                    'time_per_test': row[3] / row[4] if row[4] > 0 else 0,
                    'date': row[5]
                }
                trend_data['builds'].append(build_data)
                total_time += row[3]
            
            if rows:
                trend_data['average_execution_time'] = total_time / len(rows)
                
                # Determine performance trend
                if len(rows) >= 5:
                    recent_avg = sum(row[3] for row in rows[:5]) / 5
                    older_avg = sum(row[3] for row in rows[-5:]) / 5
                    
                    if recent_avg < older_avg * 0.9:  # 10% improvement
                        trend_data['performance_trend'] = 'improving'
                    elif recent_avg > older_avg * 1.1:  # 10% degradation
                        trend_data['performance_trend'] = 'declining'
            
            return trend_data
    
    def cleanup_old_artifacts(self, dry_run: bool = False) -> Dict[str, int]:
        """Clean up old artifacts based on retention policies."""
        cleanup_stats = {'deleted_files': 0, 'freed_space': 0}
        
        with sqlite3.connect(self.db_path) as conn:
            for artifact_type, config in self.artifact_types.items():
                retention_days = config['retention_days']
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                cursor = conn.execute('''
                    SELECT file_path, file_size FROM artifacts
                    WHERE artifact_type = ? AND created_at < ?
                ''', (artifact_type, cutoff_date.isoformat()))
                
                for row in cursor.fetchall():
                    file_path = Path(row[0])
                    file_size = row[1]
                    
                    if file_path.exists():
                        if not dry_run:
                            file_path.unlink()
                            print(f"Deleted: {file_path}")
                        
                        cleanup_stats['deleted_files'] += 1
                        cleanup_stats['freed_space'] += file_size
                
                if not dry_run:
                    # Remove database records for deleted files
                    conn.execute('''
                        DELETE FROM artifacts
                        WHERE artifact_type = ? AND created_at < ?
                    ''', (artifact_type, cutoff_date.isoformat()))
        
        return cleanup_stats
    
    def create_build_archive(self, build_number: str, output_path: str) -> bool:
        """Create compressed archive of build artifacts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT file_path FROM artifacts WHERE build_number = ?
                ''', (build_number,))
                
                artifact_files = [row[0] for row in cursor.fetchall()]
            
            if not artifact_files:
                print(f"No artifacts found for build {build_number}")
                return False
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in artifact_files:
                    path = Path(file_path)
                    if path.exists():
                        # Store with relative path in archive
                        arcname = path.relative_to(self.storage_dir)
                        zipf.write(file_path, arcname)
            
            print(f"Created archive: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating archive: {e}")
            return False
    
    def get_build_comparison(self, build1: str, build2: str) -> Dict[str, Any]:
        """Compare artifacts between two builds."""
        with sqlite3.connect(self.db_path) as conn:
            # Get build summaries
            cursor = conn.execute('''
                SELECT * FROM build_summary WHERE build_number IN (?, ?)
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
                'differences': {
                    'test_count_change': build2_data['total_tests'] - build1_data['total_tests'],
                    'coverage_change': build2_data['coverage_percentage'] - build1_data['coverage_percentage'],
                    'execution_time_change': build2_data['execution_time'] - build1_data['execution_time'],
                    'failure_rate_change': (
                        (build2_data['failed_tests'] / build2_data['total_tests'] if build2_data['total_tests'] > 0 else 0) -
                        (build1_data['failed_tests'] / build1_data['total_tests'] if build1_data['total_tests'] > 0 else 0)
                    )
                }
            }
            
            return comparison
    
    def generate_artifact_report(self, output_file: str) -> None:
        """Generate comprehensive artifact report."""
        with sqlite3.connect(self.db_path) as conn:
            # Get storage statistics
            cursor = conn.execute('''
                SELECT artifact_type, COUNT(*) as count, SUM(file_size) as total_size
                FROM artifacts
                GROUP BY artifact_type
            ''')
            
            storage_stats = {}
            total_files = 0
            total_size = 0
            
            for row in cursor.fetchall():
                storage_stats[row[0]] = {
                    'count': row[1],
                    'size': row[2],
                    'size_mb': row[2] / (1024 * 1024)
                }
                total_files += row[1]
                total_size += row[2]
            
            # Get recent builds
            cursor = conn.execute('''
                SELECT * FROM build_summary
                ORDER BY created_at DESC
                LIMIT 10
            ''')
            
            recent_builds = [dict(zip([col[0] for col in cursor.description], row)) 
                           for row in cursor.fetchall()]
            
            # Generate trends
            coverage_trend = self.generate_coverage_trend()
            performance_trend = self.generate_performance_trend()
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'storage_summary': {
                    'total_files': total_files,
                    'total_size_mb': total_size / (1024 * 1024),
                    'by_type': storage_stats
                },
                'recent_builds': recent_builds,
                'trends': {
                    'coverage': coverage_trend,
                    'performance': performance_trend
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"Artifact report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Manage test artifacts')
    parser.add_argument('action', choices=['collect', 'cleanup', 'archive', 'report', 'trends'])
    parser.add_argument('--storage-dir', default='./test-artifacts', help='Artifact storage directory')
    parser.add_argument('--source-dir', help='Source directory for artifact collection')
    parser.add_argument('--build-number', help='Build number')
    parser.add_argument('--commit-sha', help='Commit SHA')
    parser.add_argument('--branch', help='Branch name')
    parser.add_argument('--platform', help='Platform name')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    
    args = parser.parse_args()
    
    manager = TestArtifactManager(args.storage_dir)
    
    if args.action == 'collect':
        if not args.source_dir:
            print("Error: --source-dir required for collect action")
            sys.exit(1)
        
        build_info = {
            'build_number': args.build_number or os.getenv('GITHUB_RUN_NUMBER', 'unknown'),
            'commit_sha': args.commit_sha or os.getenv('GITHUB_SHA', 'unknown'),
            'branch': args.branch or os.getenv('GITHUB_REF_NAME', 'unknown'),
            'platform': args.platform or 'unknown'
        }
        
        collected = manager.collect_artifacts(args.source_dir, build_info)
        print(f"Collected artifacts: {json.dumps(collected, indent=2)}")
    
    elif args.action == 'cleanup':
        stats = manager.cleanup_old_artifacts(dry_run=args.dry_run)
        print(f"Cleanup stats: {stats}")
    
    elif args.action == 'archive':
        if not args.build_number or not args.output:
            print("Error: --build-number and --output required for archive action")
            sys.exit(1)
        
        success = manager.create_build_archive(args.build_number, args.output)
        sys.exit(0 if success else 1)
    
    elif args.action == 'report':
        output_file = args.output or 'artifact-report.json'
        manager.generate_artifact_report(output_file)
    
    elif args.action == 'trends':
        coverage_trend = manager.generate_coverage_trend()
        performance_trend = manager.generate_performance_trend()
        
        trends = {
            'coverage': coverage_trend,
            'performance': performance_trend
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(trends, f, indent=2, default=str)
        else:
            print(json.dumps(trends, indent=2, default=str))


if __name__ == '__main__':
    main()