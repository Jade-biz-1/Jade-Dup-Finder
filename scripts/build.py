#!/usr/bin/env python3
"""Unified build and packaging orchestration for DupFinder.

This script detects the current platform, selects an appropriate build target
(based on configuration stored in config/build_profiles.json), and drives the
CMake configure / build / package pipeline. Build artifacts are copied into the
required dist/ folder hierarchy for easy access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from shlex import join as shlex_join  # type: ignore
except ImportError:  # pragma: no cover - Python <3.8 fallback
    shlex_join = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "build_profiles.json"
CONFIG_TEMPLATE = CONFIG_DIR / "build_profiles.example.json"
BUILD_ROOT = REPO_ROOT / "build"
DIST_ROOT = REPO_ROOT / "dist"

DEFAULT_BUILD_TYPES = ("Debug", "Release")
WINDOWS = "Windows"
LINUX = "Linux"
DARWIN = "Darwin"
MACOS_LABEL = "MacOS"


class BuildScriptError(RuntimeError):
    """Raised when build orchestration encounters a recoverable error."""


@dataclass
class BuildTarget:
    id: str
    display_name: str
    os_name: str
    arch: List[str]
    requires_gpu: bool
    generator: Optional[str]
    architecture: Optional[str]
    multi_config: bool
    setup_scripts: List[str]
    environment: Dict[str, str]
    path_entries: List[str]
    cmake_args: List[str]
    default_build_type: str
    supported_build_types: List[str]
    artifact_extensions: List[str]
    dist_subdir: List[str]
    notes: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "BuildTarget":
        required = ["id", "os", "cmake_args"]
        for key in required:
            if key not in data:
                raise BuildScriptError(f"Missing required key '{key}' in build target definition: {data}")
        return cls(
            id=str(data["id"]),
            display_name=str(data.get("display_name", data["id"])),
            os_name=str(data.get("os", "")),
            arch=[a.lower() for a in data.get("arch", ["any"])],
            requires_gpu=bool(data.get("requires_gpu", False)),
            generator=str(data.get("generator")) if data.get("generator") else None,
            architecture=str(data.get("architecture")) if data.get("architecture") else None,
            multi_config=bool(data.get("multi_config", False)),
            setup_scripts=[str(item) for item in data.get("setup_scripts", [])],
            environment={str(k): str(v) for k, v in data.get("environment", {}).items()},
            path_entries=[str(item) for item in data.get("path", [])],
            cmake_args=[str(item) for item in data.get("cmake_args", [])],
            default_build_type=str(data.get("default_build_type", "Release")),
            supported_build_types=[str(item) for item in data.get("supported_build_types", DEFAULT_BUILD_TYPES)],
            artifact_extensions=[str(item) for item in data.get("artifact_extensions", [])],
            dist_subdir=[str(item) for item in data.get("dist", {}).get("subdir", [])],
            notes=str(data.get("notes")) if data.get("notes") else None,
        )

    def matches_os(self, system: str) -> bool:
        return self.os_name.lower() == system.lower()

    def matches_arch(self, machine: str) -> bool:
        if "any" in self.arch:
            return True
        normalized = machine.lower()
        return any(normalized == entry or normalized in entry for entry in self.arch)

    def prefers_gpu(self) -> bool:
        return self.requires_gpu

    def normalized_os_label(self) -> str:
        if self.os_name.lower() == DARWIN.lower():
            return MACOS_LABEL
        return self.os_name

    def normalized_arch_label(self) -> str:
        candidates = [entry for entry in self.arch if entry != "any"]
        if not candidates:
            return "generic"
        value = candidates[0]
        if value in ("amd64", "x86_64"):
            return "x64"
        if value in ("arm64", "aarch64"):
            return "arm"
        return value


@dataclass
class SelectionContext:
    system: str
    machine: str
    gpu_available: bool
    gpu_details: Optional[str]

    @property
    def display_system(self) -> str:
        if self.system == DARWIN:
            return MACOS_LABEL
        return self.system

    @property
    def display_machine(self) -> str:
        return self.machine


@dataclass
class ArtifactSnapshot:
    records: Dict[Path, float]

    @classmethod
    def capture(cls, search_root: Path, extensions: Sequence[str]) -> "ArtifactSnapshot":
        records: Dict[Path, float] = {}
        if not search_root.exists():
            return cls(records)
        subdirs = [child for child in search_root.iterdir() if child.is_dir()]
        for ext in extensions:
            pattern = f"*{ext}" if not ext.startswith("*") else ext
            # Allow multi-suffix like .tar.gz
            if ext == ".tar.gz":
                pattern = "*.tar.gz"
            for path in search_root.glob(pattern):
                if path.is_file() and path.name.lower().startswith("dupfinder-"):
                    records[path.resolve()] = path.stat().st_mtime
            # Look one level deeper for multi-config outputs (e.g., build/Release)
            for child in subdirs:
                for path in child.glob(pattern):
                    if path.is_file() and path.name.lower().startswith("dupfinder-"):
                        records[path.resolve()] = path.stat().st_mtime
        return cls(records)

    def diff(self, other: "ArtifactSnapshot") -> List[Path]:
        new_paths: List[Path] = []
        for path, mtime in other.records.items():
            previous = self.records.get(path)
            if previous is None or mtime > previous:
                new_paths.append(path)
        return sorted(new_paths)


def ensure_config_file(config_path: Path, template_path: Path) -> None:
    """Ensure configuration files exist.

    Checks for either:
    1. build_profiles.json (single-file mode), or
    2. build_profiles_*.json files (multi-file mode)

    If neither exists, provides helpful guidance.
    """
    config_dir = config_path.parent

    # Check if any profile files exist (single or multi-file mode)
    if config_path.exists():
        return  # Single-file mode active

    # Check for multi-file mode
    profile_files = list(config_dir.glob("build_profiles_*.json"))
    if profile_files:
        return  # Multi-file mode active

    # No configuration found - provide guidance
    if not template_path.exists():
        raise BuildScriptError(
            "Missing build configuration.\n"
            "Expected either:\n"
            "  - config/build_profiles.json (single-file mode), or\n"
            "  - config/build_profiles_*.json files (multi-file mode), or\n"
            "  - config/build_profiles.example.json (template)\n\n"
            "No configuration files or templates found."
        )

    # Suggest multi-file approach since it's the new standard
    raise BuildScriptError(
        f"No build configuration found.\n\n"
        f"You can either:\n"
        f"  1. Copy example profiles: Copy build_profiles.example.json to build_profiles.json\n"
        f"  2. Use multi-file mode: The example file shows targets that can be split into:\n"
        f"     - build_profiles_windows-msvc-cpu.json\n"
        f"     - build_profiles_windows-msvc-cuda.json\n"
        f"     - build_profiles_linux-cpu.json\n"
        f"     - etc.\n\n"
        f"See docs/BUILD_SYSTEM_OVERVIEW.md for more information."
    )


def load_targets(config_path: Path) -> Tuple[List[BuildTarget], Dict[str, str]]:
    """Load build targets from configuration file(s).

    Supports two modes:
    1. Single file: Load from build_profiles.json (legacy)
    2. Multi-file: Auto-discover and load all build_profiles_*.json files

    Returns:
        Tuple of (targets list, target_id -> source_file mapping)
    """
    targets: List[BuildTarget] = []
    target_sources: Dict[str, str] = {}

    # Check if we're using the single-file approach (legacy)
    if config_path.name == "build_profiles.json" and config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        target_dicts = raw.get("targets", [])
        for entry in target_dicts:
            target = BuildTarget.from_dict(entry)
            targets.append(target)
            target_sources[target.id] = "build_profiles.json"

    # Auto-discover all build_profiles_*.json files (new multi-file approach)
    config_dir = config_path.parent
    profile_files = sorted(config_dir.glob("build_profiles_*.json"))

    if profile_files:
        for profile_file in profile_files:
            try:
                with profile_file.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                target_dicts = raw.get("targets", [])
                for entry in target_dicts:
                    target = BuildTarget.from_dict(entry)
                    targets.append(target)
                    target_sources[target.id] = profile_file.name
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"Warning: Failed to load {profile_file.name}: {exc}")
                continue

    if not targets:
        raise BuildScriptError(
            "No build targets found. Either:\n"
            "  1. Create build_profiles.json (single-file mode), or\n"
            "  2. Create build_profiles_<target>.json files (multi-file mode)"
        )

    return targets, target_sources


def detect_gpu() -> Tuple[bool, Optional[str]]:
    commands: Sequence[Sequence[str]] = (
        ("nvidia-smi", "-L"),
        ("nvcc", "--version"),
    )
    for cmd in commands:
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            continue
        output = completed.stdout.strip()
        if output:
            first_line = output.splitlines()[0]
            return True, first_line
        return True, None
    # Fall back to environment variables commonly present when CUDA is installed
    for key in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        if os.environ.get(key):
            return True, f"{key}={os.environ[key]}"
    return False, None


def ensure_dist_structure() -> None:
    blueprint = [
        DIST_ROOT / "Win64" / "Debug",
        DIST_ROOT / "Win64" / "Release",
        DIST_ROOT / "Linux" / "Debug",
        DIST_ROOT / "Linux" / "Release",
        DIST_ROOT / "MacOS" / "X64" / "Debug",
        DIST_ROOT / "MacOS" / "X64" / "Release",
        DIST_ROOT / "MacOS" / "ARM" / "Debug",
        DIST_ROOT / "MacOS" / "ARM" / "Release",
    ]
    for directory in blueprint:
        directory.mkdir(parents=True, exist_ok=True)


def select_targets(targets: List[BuildTarget], context: SelectionContext) -> Tuple[List[BuildTarget], List[Tuple[BuildTarget, str]]]:
    supported: List[BuildTarget] = []
    blocked: List[Tuple[BuildTarget, str]] = []
    for target in targets:
        if not target.matches_os(context.system):
            continue
        if not target.matches_arch(context.machine):
            continue
        if target.requires_gpu and not context.gpu_available:
            blocked.append((target, "Requires NVIDIA/CUDA GPU"))
            continue
        supported.append(target)
    return supported, blocked


def choose_default_target(options: List[BuildTarget], gpu_available: bool) -> Optional[BuildTarget]:
    if not options:
        return None
    gpu_candidates = [target for target in options if target.prefers_gpu()]
    if gpu_available and gpu_candidates:
        return gpu_candidates[0]
    return options[0]


def prompt_choice(options: List[BuildTarget], default: BuildTarget) -> BuildTarget:
    print("\nAvailable build targets:")
    for idx, target in enumerate(options, start=1):
        suffix = " (default)" if target.id == default.id else ""
        gpu_flag = " [GPU]" if target.prefers_gpu() else ""
        print(f"  {idx}. {target.display_name}{gpu_flag} [{target.id}]{suffix}")
    while True:
        raw = input(f"Select build target [default {default.id}]: ").strip()
        if not raw:
            return default
        if not raw.isdigit():
            # allow typing the target id directly
            selected = next((opt for opt in options if opt.id == raw), None)
            if selected:
                return selected
            print("Invalid selection. Enter the list number or target id.")
            continue
        index = int(raw)
        if 1 <= index <= len(options):
            return options[index - 1]
        print("Selection out of range. Try again.")


def prompt_build_type(target: BuildTarget) -> str:
    supported = target.supported_build_types or list(DEFAULT_BUILD_TYPES)
    supported = list(dict.fromkeys(supported))  # remove duplicates preserving order
    default = target.default_build_type if target.default_build_type in supported else supported[0]
    options = ", ".join(supported)
    while True:
        raw = input(f"Select build type ({options}) [default {default}]: ").strip()
        if not raw:
            return default
        if raw in supported:
            return raw
        print("Invalid build type. Please choose one of:", options)


def compute_build_directory(target: BuildTarget, context: SelectionContext) -> Path:
    os_key = target.os_name.lower()
    if os_key == WINDOWS.lower():
        os_part = "windows"
        arch_part = "win64"
    elif os_key == DARWIN.lower():
        os_part = "macos"
        arch_part = "arm64" if any(item in ("arm64", "aarch64") for item in target.arch) else "x64"
    else:
        os_part = os_key
        arch_part = target.normalized_arch_label()
    return BUILD_ROOT / os_part / arch_part / target.id


def compute_dist_directory(target: BuildTarget, build_type: str) -> Path:
    if target.dist_subdir:
        parts = [
            item.format(
                build_type=build_type,
                target_id=target.id,
                target=target.id,
                os=target.normalized_os_label(),
                arch=target.normalized_arch_label(),
            )
            for item in target.dist_subdir
        ]
        return DIST_ROOT.joinpath(*parts)
    # Fallback mapping
    label = target.normalized_os_label()
    build_type_part = build_type
    if label == WINDOWS:
        parts = ["Win64", build_type_part]
    elif label == MACOS_LABEL:
        arch_label = target.normalized_arch_label().upper()
        parts = ["MacOS", arch_label, build_type_part]
    else:
        parts = ["Linux", build_type_part]
    return DIST_ROOT.joinpath(*parts)


def format_command(command: Sequence[str]) -> str:
    if shlex_join:
        try:
            return shlex_join(list(command))
        except Exception:  # pragma: no cover - defensive fallback
            pass
    return " ".join(command)


def run_command(
    command: Sequence[str],
    env: Dict[str, str],
    cwd: Path,
    setup_scripts: Sequence[str],
    dry_run: bool = False,
) -> None:
    human = format_command(command)
    if platform.system() == WINDOWS and setup_scripts:
        prefix = " && ".join(f'call "{script}"' for script in setup_scripts)
        human = f"{prefix} && {human}"
    print(f"\n>> {human}")
    if dry_run:
        return
    try:
        if platform.system() == WINDOWS and setup_scripts:
            wrapped = subprocess.list2cmdline(list(command))
            prefix = " && ".join(f'call "{script}"' for script in setup_scripts)
            full = f"{prefix} && {wrapped}"
            subprocess.run(full, cwd=str(cwd), env=env, check=True, shell=True)
        else:
            subprocess.run(list(command), cwd=str(cwd), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise BuildScriptError(f"Command failed with exit code {exc.returncode}") from exc


def prepare_environment(target: BuildTarget) -> Dict[str, str]:
    env = os.environ.copy()
    for key, value in target.environment.items():
        env[key] = value
    if target.path_entries:
        original = env.get("PATH", "")
        augmented = os.pathsep.join(target.path_entries + ([original] if original else []))
        env["PATH"] = augmented
    return env


def build_and_package(
    target: BuildTarget,
    build_type: str,
    build_dir: Path,
    dist_dir: Path,
    env: Dict[str, str],
    setup_scripts: Sequence[str],
    skip_package: bool,
    clean: bool,
    dry_run: bool,
) -> List[Path]:
    if clean and build_dir.exists() and not dry_run:
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    configure_cmd: List[str] = ["cmake", "-S", str(REPO_ROOT), "-B", str(build_dir)]
    if target.generator:
        configure_cmd.extend(["-G", target.generator])
    if target.multi_config:
        if target.architecture:
            configure_cmd.extend(["-A", target.architecture])
    else:
        configure_cmd.append(f"-DCMAKE_BUILD_TYPE={build_type}")
    for arg in target.cmake_args:
        configure_cmd.append(arg.format(build_type=build_type))

    run_command(configure_cmd, env, REPO_ROOT, setup_scripts, dry_run)

    build_cmd: List[str] = ["cmake", "--build", str(build_dir), "--target", "dupfinder", "--parallel"]
    if target.multi_config:
        build_cmd.extend(["--config", build_type])
    run_command(build_cmd, env, REPO_ROOT, setup_scripts, dry_run)

    artifact_extensions = target.artifact_extensions or [".exe", ".deb", ".rpm", ".tgz", ".tar.gz", ".dmg", ".pkg"]
    before = ArtifactSnapshot.capture(build_dir, artifact_extensions) if not dry_run else ArtifactSnapshot({})

    produced: List[Path] = []
    if not skip_package:
        package_cmd: List[str] = ["cmake", "--build", str(build_dir), "--target", "package"]
        if target.multi_config:
            package_cmd.extend(["--config", build_type])
        run_command(package_cmd, env, REPO_ROOT, setup_scripts, dry_run)

        # Also run copy_installer target to copy packages to dist folder
        copy_installer_cmd: List[str] = ["cmake", "--build", str(build_dir), "--target", "copy_installer"]
        if target.multi_config:
            copy_installer_cmd.extend(["--config", build_type])
        try:
            run_command(copy_installer_cmd, env, REPO_ROOT, setup_scripts, dry_run)
        except BuildScriptError:
            # copy_installer target might fail if package files don't exist yet
            # This is acceptable - we'll copy them manually if needed
            pass

        if not dry_run:
            after = ArtifactSnapshot.capture(build_dir, artifact_extensions)
            produced = before.diff(after)
    return produced


def generate_checksums(file_path: Path) -> Dict[str, str]:
    """Generate SHA256 and MD5 checksums for a file.
    
    Args:
        file_path: Path to the file to checksum
        
    Returns:
        Dictionary with 'sha256' and 'md5' keys
    """
    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()
    
    with open(file_path, "rb") as f:
        # Read in 64kb chunks for memory efficiency
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
            md5_hash.update(chunk)
    
    return {
        "sha256": sha256_hash.hexdigest(),
        "md5": md5_hash.hexdigest()
    }


def write_checksum_files(file_path: Path, checksums: Dict[str, str]) -> None:
    """Write checksum files in standard formats.
    
    Creates:
    - <filename>.sha256 - SHA256 checksum in standard format
    - <filename>.md5 - MD5 checksum in standard format
    
    Args:
        file_path: Path to the file that was checksummed
        checksums: Dictionary with 'sha256' and 'md5' keys
    """
    # Write SHA256 file (format: <hash> *<filename>)
    sha256_file = file_path.parent / f"{file_path.name}.sha256"
    with open(sha256_file, "w") as f:
        f.write(f"{checksums['sha256']} *{file_path.name}\n")
    print(f"  Created checksum: {sha256_file.name}")
    
    # Write MD5 file (format: <hash> *<filename>)
    md5_file = file_path.parent / f"{file_path.name}.md5"
    with open(md5_file, "w") as f:
        f.write(f"{checksums['md5']} *{file_path.name}\n")
    print(f"  Created checksum: {md5_file.name}")


def copy_artifacts(artifacts: Iterable[Path], dist_dir: Path) -> List[Path]:
    dist_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for artifact in artifacts:
        if artifact.suffixes[-2:] == [".tar", ".gz"]:
            target_name = artifact.name[:-7] + ".tgz"
        else:
            target_name = artifact.name
        destination = dist_dir / target_name
        shutil.copy2(artifact, destination)
        copied.append(destination)
        print(f"Copied {artifact.name} -> {destination.relative_to(REPO_ROOT)}")
        
        # Generate and write checksums for the copied artifact
        print(f"Generating checksums for {target_name}...")
        checksums = generate_checksums(destination)
        write_checksum_files(destination, checksums)
        
    if not copied:
        print("No new distribution artifacts were detected.")
    return copied


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-platform build and packaging helper for DupFinder.")
    parser.add_argument("--config", type=Path, help="Path to build profile configuration JSON.")
    parser.add_argument("--target", help="Explicit build target id to use.")
    parser.add_argument("--build-type", choices=DEFAULT_BUILD_TYPES, help="Override build type (Debug or Release).")
    parser.add_argument("--skip-package", action="store_true", help="Build binaries without running the CPack packaging step.")
    parser.add_argument("--clean", action="store_true", help="Delete the target build directory before configuring.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without executing commands.")
    parser.add_argument("--non-interactive", action="store_true", help="Do not prompt for selections; use provided defaults.")
    parser.add_argument("--force", action="store_true", help="Proceed without confirmation prompts.")
    parser.add_argument("--list-targets", action="store_true", help="List all configured build targets and exit.")
    return parser.parse_args()


def list_all_targets(targets: List[BuildTarget], target_sources: Dict[str, str]) -> None:
    print("Configured build targets:")
    for target in targets:
        gpu_flag = " [GPU]" if target.prefers_gpu() else ""
        source_file = target_sources.get(target.id, "unknown")
        print(f"- {target.id}: {target.display_name}{gpu_flag} ({target.os_name}) [from {source_file}]")


def main() -> None:
    args = parse_arguments()

    config_path = args.config if args.config else CONFIG_FILE
    try:
        ensure_config_file(config_path, CONFIG_TEMPLATE)
    except BuildScriptError as exc:
        print(exc)
        return

    try:
        targets, target_sources = load_targets(config_path)
    except BuildScriptError as exc:
        print(exc)
        return

    if args.list_targets:
        list_all_targets(targets, target_sources)
        return

    system = platform.system()
    machine = platform.machine().lower()
    gpu_available, gpu_details = detect_gpu()
    context = SelectionContext(system=system, machine=machine, gpu_available=gpu_available, gpu_details=gpu_details)

    supported, blocked = select_targets(targets, context)
    selected: Optional[BuildTarget] = None
    if args.target:
        selected = next((target for target in targets if target.id == args.target), None)
        if not selected:
            print(f"Unknown target id: {args.target}")
            return
        if not selected.matches_os(system):
            print(f"Target '{selected.id}' is not valid for {context.display_system}.")
            return
        if not selected.matches_arch(machine):
            print(f"Target '{selected.id}' does not support architecture {context.display_machine}.")
            return
        if selected.requires_gpu and not gpu_available and not args.force:
            print(f"Target '{selected.id}' requires a CUDA-capable GPU. Use --force to override.")
            return
    else:
        if not supported:
            print("No compatible build targets found for this machine.")
            if blocked:
                print("The following targets were skipped:")
                for target, reason in blocked:
                    print(f"  - {target.id}: {reason}")
            return
        default_target = choose_default_target(supported, gpu_available)
        if args.non_interactive:
            selected = default_target
        else:
            selected = prompt_choice(supported, default_target)

    if selected is None:
        print("No build target selected.")
        return

    build_type = args.build_type or selected.default_build_type
    if not args.non_interactive and not args.build_type:
        build_type = prompt_build_type(selected)

    print("\nEnvironment summary:")
    print(f"  Operating system : {context.display_system}")
    print(f"  Architecture     : {context.display_machine}")
    gpu_status = "Detected" if gpu_available else "Not detected"
    if selected.prefers_gpu():
        gpu_status += " (required for this build)"
    print(f"  GPU               : {gpu_status}")
    if gpu_details:
        print(f"  GPU details      : {gpu_details}")

    print("\nBuild configuration:")
    print(f"  Target           : {selected.display_name} [{selected.id}]")
    if selected.notes:
        print(f"  Notes            : {selected.notes}")
    print(f"  Build type       : {build_type}")
    print(f"  Generator        : {selected.generator or 'Default'}")
    print(f"  Multi-config     : {'Yes' if selected.multi_config else 'No'}")

    if not args.force and not args.non_interactive:
        confirmation = input("Proceed with this configuration? [Y/n]: ").strip().lower()
        if confirmation and confirmation not in ("y", "yes"):
            print("Build cancelled by user.")
            return

    ensure_dist_structure()

    build_dir = compute_build_directory(selected, context)
    dist_dir = compute_dist_directory(selected, build_type)
    env = prepare_environment(selected)

    try:
        artifacts = build_and_package(
            target=selected,
            build_type=build_type,
            build_dir=build_dir,
            dist_dir=dist_dir,
            env=env,
            setup_scripts=selected.setup_scripts,
            skip_package=args.skip_package,
            clean=args.clean,
            dry_run=args.dry_run,
        )
    except BuildScriptError as exc:
        print(exc)
        return

    if args.dry_run:
        print("Dry-run complete. No artifacts were generated.")
        return

    if args.skip_package:
        print("Packaging step skipped as requested.")
        return

    copied = copy_artifacts(artifacts, dist_dir)
    if copied:
        print("\nDistribution artifacts:")
        for item in copied:
            print(f"  - {item.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except BuildScriptError as err:
        print(err)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nBuild interrupted by user.")
        sys.exit(130)