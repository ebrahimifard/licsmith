import sys
import ast
from pathlib import Path
import re
from typing import List, Tuple, Optional, Set

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib 

from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion


def parse_requirement_line(line: str) -> Optional[Tuple[str, Optional[str]]]:
    """Parse a single requirement line and return (name, lower_bound_version)."""
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('-'):
        return None
    
    # Handle -e/--editable
    if line.startswith(('-e ', '--editable')):
        line = line.split(None, 1)[1] if ' ' in line else ''
    
    if Requirement:
        try:
            req = Requirement(line)
            # Extract lower bound version from specifiers
            lower_bound = None
            for spec in req.specifier:
                if spec.operator in ('>=', '==', '~='):
                    lower_bound = spec.version
                    break
                elif spec.operator == '>':
                    lower_bound = spec.version
                    break
            return req.name, lower_bound
        except:
            pass
    
    # Fallback: simple regex parsing
    match = re.match(r'^([a-zA-Z0-9\-_.]+)', line)
    if match:
        name = match.group(1)
        # Try to find version (look for >=, ==, ~=, >)
        version_match = re.search(r'(?:>=|==|~=|>)\s*([0-9.]+(?:\.[0-9]+)*)', line)
        version = version_match.group(1) if version_match else None
        return name, version
    
    return None


def parse_setup_py(file_path: Path) -> List[Tuple[str, Optional[str]]]:
    """Parse setup.py for install_requires dependencies."""
    if not file_path.exists():
        return []
    
    packages = []
    try:
        content = file_path.read_text(encoding='utf-8')
        tree = ast.parse(content)
        class SetupVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Look for setup() calls
                if (isinstance(node.func, ast.Name) and node.func.id == 'setup') or \
                   (isinstance(node.func, ast.Attribute) and node.func.attr == 'setup'):
                    
                    for keyword in node.keywords:
                        if keyword.arg == 'install_requires':
                            try:
                                # Evaluate the list/tuple of requirements
                                requirements = ast.literal_eval(keyword.value)
                                if isinstance(requirements, (list, tuple)):
                                    for req in requirements:
                                        if isinstance(req, str):
                                            result = parse_requirement_line(req)
                                            if result:
                                                packages.append(result)
                            except:
                                pass
                self.generic_visit(node)
        
        visitor = SetupVisitor()
        visitor.visit(tree)
    
    except Exception:
        pass
    
    return packages


def parse_requirements_file(file_path: Path, seen_files: Optional[Set[Path]] = None) -> List[Tuple[str, Optional[str]]]:
    """Parse a requirements.txt file with -r includes support."""
    if not file_path.exists():
        return []
    
    if seen_files is None:
        seen_files = set()
    
    resolved_path = file_path.resolve()
    if resolved_path in seen_files:
        return []
    seen_files.add(resolved_path)
    
    packages = []
    try:
        content = file_path.read_text(encoding='utf-8')
        for line in content.splitlines():
            line = line.strip()
            
            # Handle -r includes
            if line.startswith(('-r ', '--requirement')):
                include_file = line.split(None, 1)[1].strip()
                include_path = (file_path.parent / include_file).resolve()
                packages.extend(parse_requirements_file(include_path, seen_files))
                continue
            
            result = parse_requirement_line(line)
            if result:
                packages.append(result)
    except Exception:
        pass
    
    return packages


def parse_pyproject_toml(file_path: Path) -> List[Tuple[str, Optional[str]]]:
    """Parse pyproject.toml for dependencies."""
    if not file_path.exists() or not tomllib:
        return []
    
    packages = []
    try:
        with open(file_path, 'rb') as f:
            data = tomllib.load(f)
        
        # PEP 621 dependencies
        project = data.get('project', {})
        deps = project.get('dependencies', [])
        for dep in deps:
            result = parse_requirement_line(dep)
            if result:
                packages.append(result)
        
        # Optional dependencies
        optional_deps = project.get('optional-dependencies', {})
        for group_deps in optional_deps.values():
            for dep in group_deps:
                result = parse_requirement_line(dep)
                if result:
                    packages.append(result)
        
        # Poetry dependencies
        tool_poetry = data.get('tool', {}).get('poetry', {})
        poetry_deps = tool_poetry.get('dependencies', {})
        for name, spec in poetry_deps.items():
            if name.lower() == 'python':
                continue
            
            version = None
            if isinstance(spec, str):
                # Extract version from string like "^1.2.3" or ">=1.2.3"
                version_match = re.search(r'([0-9.]+(?:\.[0-9]+)*)', spec)
                version = version_match.group(1) if version_match else None
            elif isinstance(spec, dict):
                ver = spec.get('version')
                if isinstance(ver, str):
                    version_match = re.search(r'([0-9.]+(?:\.[0-9]+)*)', ver)
                    version = version_match.group(1) if version_match else None
            
            packages.append((name, version))
        
        # Poetry dev dependencies (groups)
        poetry_groups = tool_poetry.get('group', {})
        for group_data in poetry_groups.values():
            group_deps = group_data.get('dependencies', {})
            for name, spec in group_deps.items():
                version = None
                if isinstance(spec, str):
                    version_match = re.search(r'([0-9.]+(?:\.[0-9]+)*)', spec)
                    version = version_match.group(1) if version_match else None
                elif isinstance(spec, dict):
                    ver = spec.get('version')
                    if isinstance(ver, str):
                        version_match = re.search(r'([0-9.]+(?:\.[0-9]+)*)', ver)
                        version = version_match.group(1) if version_match else None
                
                packages.append((name, version))
        
        # PDM dependencies
        tool_pdm = data.get('tool', {}).get('pdm', {})
        
        # PDM regular dependencies
        pdm_deps = tool_pdm.get('dependencies')
        if isinstance(pdm_deps, list):
            # PEP 621 style list
            for dep in pdm_deps:
                result = parse_requirement_line(dep)
                if result:
                    packages.append(result)
        elif isinstance(pdm_deps, dict):
            # PDM dict style
            for name, spec in pdm_deps.items():
                if isinstance(spec, str):
                    result = parse_requirement_line(f"{name} {spec}")
                    if result:
                        packages.append(result)
                else:
                    packages.append((name, None))
        
        # PDM dev dependencies
        pdm_dev_deps = tool_pdm.get('dev-dependencies')
        if isinstance(pdm_dev_deps, list):
            for dep in pdm_dev_deps:
                result = parse_requirement_line(dep)
                if result:
                    packages.append(result)
        elif isinstance(pdm_dev_deps, dict):
            for name, spec in pdm_dev_deps.items():
                if isinstance(spec, str):
                    result = parse_requirement_line(f"{name} {spec}")
                    if result:
                        packages.append(result)
                else:
                    packages.append((name, None))
        
        # Flit legacy format
        tool_flit = data.get('tool', {}).get('flit', {})
        flit_metadata = tool_flit.get('metadata', {})
        flit_requires = flit_metadata.get('requires', [])
        if isinstance(flit_requires, list):
            for dep in flit_requires:
                result = parse_requirement_line(dep)
                if result:
                    packages.append(result)
    
    except Exception:
        pass
    
    return packages


def normalize_version(version: Optional[str]) -> Optional[str]:
    """Normalize version string using packaging.Version if available."""
    if version is None:
        return None
    
    if Version:
        try:
            return str(Version(version))
        except InvalidVersion:
            pass
    
    return version


def collect_project_packages(root_path: str | Path) -> List[Tuple[str, Optional[str]]]:
    """
    Collect all dependencies from a Python project.

    Returns list of (package_name, version) tuples.
    """
    root = Path(root_path)
    all_packages = []

    # Parse pyproject.toml
    all_packages.extend(parse_pyproject_toml(root / 'pyproject.toml'))

    # Parse setup.py
    all_packages.extend(parse_setup_py(root / 'setup.py'))

    # Parse requirements files (including -r includes)
    req_files = [
        'requirements.txt',
        'requirements-dev.txt',
        'requirements-test.txt',
        'dev-requirements.txt',
        'pip-req.txt',
    ]
    
    for req_file in req_files:
        all_packages.extend(parse_requirements_file(root / req_file))

    # Find any other requirements*.txt files in root
    for req_file in root.glob('requirements*.txt'):
        if req_file.name not in req_files:
            all_packages.extend(parse_requirements_file(req_file))
    
    # Check requirements/ subdirectory for all .txt files
    requirements_dir = root / 'requirements'
    if requirements_dir.is_dir():
        for req_file in requirements_dir.glob('*.txt'):
            all_packages.extend(parse_requirements_file(req_file))
    
    # Deduplicate with distinct versions logic
    seen_combinations: Set[Tuple[str, Optional[str]]] = set()
    name_casing: dict[str, str] = {}
    result: List[Tuple[str, Optional[str]]] = []
    
    for name, version in all_packages:
        name_lower = name.lower()
        
        # Track first-seen casing
        if name_lower not in name_casing:
            name_casing[name_lower] = name
        
        # Normalize version for comparison
        normalized_version = normalize_version(version)
        
        # Check if we've seen this exact combination
        combination = (name_lower, normalized_version)
        if combination not in seen_combinations:
            seen_combinations.add(combination)
            result.append((name_casing[name_lower], normalized_version))
    
    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python simple_collector.py <project_path>")
        sys.exit(1)
    
    packages = collect_project_packages(sys.argv[1])
    for name, version in packages:
        version_str = f" ({version})" if version else ""
        print(f"{name}{version_str}")