#!/usr/bin/env python3
"""Generate checksums for existing packages in dist/ folder."""

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = REPO_ROOT / "dist"


def generate_checksums(file_path: Path) -> dict:
    """Generate SHA256 and MD5 checksums for a file."""
    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
            md5_hash.update(chunk)
    
    return {
        "sha256": sha256_hash.hexdigest(),
        "md5": md5_hash.hexdigest()
    }


def write_checksum_files(file_path: Path, checksums: dict) -> None:
    """Write checksum files in standard format."""
    # Write SHA256 file
    sha256_file = file_path.parent / f"{file_path.name}.sha256"
    with open(sha256_file, "w") as f:
        f.write(f"{checksums['sha256']} *{file_path.name}\n")
    print(f"  Created: {sha256_file.relative_to(REPO_ROOT)}")
    
    # Write MD5 file
    md5_file = file_path.parent / f"{file_path.name}.md5"
    with open(md5_file, "w") as f:
        f.write(f"{checksums['md5']} *{file_path.name}\n")
    print(f"  Created: {md5_file.relative_to(REPO_ROOT)}")


def main():
    """Generate checksums for all packages in dist/ folder."""
    if not DIST_ROOT.exists():
        print(f"Error: {DIST_ROOT} does not exist")
        sys.exit(1)
    
    # Package extensions to process
    extensions = [".exe", ".deb", ".rpm", ".tgz", ".dmg", ".pkg"]
    
    # Find all package files
    package_files = []
    for ext in extensions:
        package_files.extend(DIST_ROOT.rglob(f"*{ext}"))
    
    if not package_files:
        print("No package files found in dist/ folder")
        sys.exit(0)
    
    print(f"Found {len(package_files)} package(s) in dist/ folder\n")
    
    # Generate checksums for each package
    for package in sorted(package_files):
        print(f"Processing: {package.relative_to(REPO_ROOT)}")
        
        # Skip if checksums already exist
        sha256_file = package.parent / f"{package.name}.sha256"
        md5_file = package.parent / f"{package.name}.md5"
        
        if sha256_file.exists() and md5_file.exists():
            print(f"  Checksums already exist, skipping")
            continue
        
        # Generate checksums
        checksums = generate_checksums(package)
        write_checksum_files(package, checksums)
        print()
    
    print("Checksum generation complete!")


if __name__ == "__main__":
    main()
