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
    """Parse setup.py for install_requires, tests_require, and extras_require (with branch-aware variable tracking)."""
    if not file_path.exists():
        return []
    
    packages: List[Tuple[str, Optional[str]]] = []

    def as_candidates(value):
        # Always represent env entries as a list of candidate values
        return value if isinstance(value, list) and value and not isinstance(value[0], (str, dict)) and all(isinstance(v, list) for v in value) else [value]

    def merge_candidate(env, name, value):
        # Keep distinct candidates (by repr) in insertion order
        candidates = env.get(name, [])
        key = repr(value)
        seen_keys = {repr(v) for v in candidates}
        if key not in seen_keys:
            candidates.append(value)
        env[name] = candidates

    def flatten_str_lists(value) -> List[str]:
        """
        Accept:
          - list[str]
          - tuple[str]
          - list[list[str]]  (candidates)
        Return flat list[str].
        """
        if isinstance(value, (list, tuple)) and (not value or isinstance(value[0], str)):
            return [v for v in value if isinstance(v, str)]
        if isinstance(value, list) and value and isinstance(value[0], list):
            out: List[str] = []
            for sub in value:
                out.extend([v for v in sub if isinstance(v, str)])
            return out
        return []

    def safe_eval_node(node, env):
        """
        Evaluate only simple literals relevant to setup metadata:
        - str, list/tuple of str
        - dict[str, list[str]]
        - names (may resolve to multiple candidates)
        - list concatenation via BinOp(Add)
        - list()/tuple() wrappers
        Returns either:
          - concrete value (list[str], dict, str), or
          - list of candidate lists (when resolving a Name with multiple assignments)
        """
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [safe_eval_node(elt, env) if not isinstance(elt, ast.Constant) else elt.value for elt in node.elts]
        if isinstance(node, ast.Dict):
            keys = [safe_eval_node(k, env) for k in node.keys]
            vals = [safe_eval_node(v, env) for v in node.values]
            return dict(zip(keys, vals))
        if isinstance(node, ast.Name):
            if node.id in env:
                # Could be multiple candidates; normalize to list[list[str]] or direct value
                return env[node.id] if isinstance(env[node.id], list) and env[node.id] and isinstance(env[node.id][0], list) else env[node.id]
            raise ValueError(f"Unresolved name: {node.id}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = safe_eval_node(node.left, env)
            right = safe_eval_node(node.right, env)
            # Handle concatenation in the presence of candidates
            left_lists = [left] if (isinstance(left, list) and (not left or isinstance(left[0], str))) else (left if isinstance(left, list) and left and isinstance(left[0], list) else [])
            right_lists = [right] if (isinstance(right, list) and (not right or isinstance(right[0], str))) else (right if isinstance(right, list) and right and isinstance(right[0], list) else [])
            if left_lists and right_lists:
                return [ (l or []) + (r or []) for l in left_lists for r in right_lists ]
            # Fallback: simple list + list
            if isinstance(left, list) and isinstance(right, list):
                return left + right
            raise ValueError("Unsupported BinOp operands")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            if node.func.id in ('list', 'tuple'):
                inner = safe_eval_node(node.args[0], env)
                if isinstance(inner, list) and inner and isinstance(inner[0], list):
                    # list(candidate_list) → keep candidates
                    return [list(c) for c in inner]
                return list(inner) if node.func.id == 'list' else list(inner)
        raise ValueError("Unsupported node for safe eval")

    try:
        content = file_path.read_text(encoding='utf-8')
        tree = ast.parse(content)

        class SetupVisitor(ast.NodeVisitor):
            def __init__(self):
                # env[name] holds a list of candidate values (each candidate is usually list[str] or dict)
                self.env: dict[str, list] = {}

            def visit_Assign(self, node: ast.Assign):
                try:
                    value = safe_eval_node(node.value, self.env)
                except Exception:
                    return self.generic_visit(node)
                # If the value itself is a list of candidate lists, keep all; otherwise store single candidate
                candidates = value if (isinstance(value, list) and value and isinstance(value[0], list) and not isinstance(value[0][0], (list, dict))) else [value]
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        for cand in candidates:
                            merge_candidate(self.env, target.id, cand)

            def _collect_kwarg_flat_strs(self, node: ast.Call, arg_name: str) -> List[str]:
                for kw in node.keywords:
                    if kw.arg == arg_name:
                        try:
                            value = safe_eval_node(kw.value, self.env)
                        except Exception:
                            return []
                        # Name with multiple assignments → list of candidate lists; flatten them
                        if isinstance(value, list) and value and isinstance(value[0], list):
                            return flatten_str_lists(value)
                        return flatten_str_lists(value)
                return []

            def _collect_extras_flat_strs(self, node: ast.Call) -> List[str]:
                for kw in node.keywords:
                    if kw.arg == 'extras_require':
                        try:
                            value = safe_eval_node(kw.value, self.env)
                        except Exception:
                            return []
                        reqs: List[str] = []
                        # If extras is provided via a Name with multiple dict candidates, union all lists
                        dict_candidates = value if (isinstance(value, list) and value and isinstance(value[0], dict)) else [value]
                        for d in dict_candidates:
                            if isinstance(d, dict):
                                for dep_list in d.values():
                                    reqs.extend(flatten_str_lists(dep_list))
                        return reqs
                return []

            def visit_Call(self, node: ast.Call):
                is_setup = (
                    (isinstance(node.func, ast.Name) and node.func.id == 'setup') or
                    (isinstance(node.func, ast.Attribute) and node.func.attr == 'setup')
                )
                if not is_setup:
                    return self.generic_visit(node)

                # install_requires — now union of all possible branch assignments
                for req in self._collect_kwarg_flat_strs(node, 'install_requires'):
                    res = parse_requirement_line(req)
                    if res:
                        packages.append(res)

                # tests_require (kept)
                for req in self._collect_kwarg_flat_strs(node, 'tests_require'):
                    res = parse_requirement_line(req)
                    if res:
                        packages.append(res)

                # extras_require (kept)
                for req in self._collect_extras_flat_strs(node):
                    res = parse_requirement_line(req)
                    if res:
                        packages.append(res)

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
            for dep in pdm_deps:
                result = parse_requirement_line(dep)
                if result:
                    packages.append(result)
        elif isinstance(pdm_deps, dict):
            for name, spec in pdm_deps.items():
                if isinstance(spec, str):
                    result = parse_requirement_line(f"{name} {spec}")
                    if result:
                        packages.append(result)
                else:
                    packages.append((name, None))
        
        # PDM dev dependencies (***fixed***)
        # Support both list *and* dict-of-groups => list-of-strings
        pdm_dev_deps = tool_pdm.get('dev-dependencies')
        if isinstance(pdm_dev_deps, list):
            for dep in pdm_dev_deps:
                result = parse_requirement_line(dep)
                if result:
                    packages.append(result)
        elif isinstance(pdm_dev_deps, dict):
            # Example:
            # [tool.pdm.dev-dependencies]
            # test = ["pytest>=8.0", "pytest-cov>=5.0"]
            for group_name, group_spec in pdm_dev_deps.items():
                if isinstance(group_spec, list):
                    for dep in group_spec:
                        result = parse_requirement_line(dep)
                        if result:
                            packages.append(result)
                elif isinstance(group_spec, str):
                    # Rare, but be defensive
                    result = parse_requirement_line(group_spec)
                    if result:
                        packages.append(result)
                elif isinstance(group_spec, dict):
                    # Extremely rare patterns; fall back to names
                    for name, spec in group_spec.items():
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
        # ***NEW*** Flit extras
        # [tool.flit.metadata.requires-extra]
        flit_requires_extra = flit_metadata.get('requires-extra', {})
        if isinstance(flit_requires_extra, dict):
            for _extra_name, extra_list in flit_requires_extra.items():
                if isinstance(extra_list, list):
                    for dep in extra_list:
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
        print("Usage: python collect_dependencies.py <project_path>")
        sys.exit(1)
    
    packages = collect_project_packages(sys.argv[1])
    for name, version in packages:
        version_str = f" ({version})" if version else ""
        print(f"{name}{version_str}")