import requests
import tarfile
import zipfile
import tempfile
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

from collect_dependencies import collect_project_packages as cpp

def download_and_extract_licenses(
    packages: List[Tuple[str, Optional[str]]],
    output_file: str = "COMBINED_LICENSES.txt",
    temp_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Download Python packages, extract their license files, and combine them.
   
    Args:
        packages: List of tuples (package_name, version). Version can be None for latest.
        output_file: Path to the output file containing all licenses
        temp_dir: Optional temporary directory path. If None, uses system temp.
   
    Returns:
        Dictionary mapping package names to their license status
   
    Example:
        packages = [
            ("requests", "2.31.0"),
            ("numpy", None),  # Latest version
            ("pandas", "2.0.3")
        ]
        results = download_and_extract_licenses(packages)
    """
   
    # Convert output_file to absolute path and show where it will be saved
    output_path = os.path.abspath(output_file)
    print(f"License file will be saved to: {output_path}")
   
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
   
    # Create temporary directory
    if temp_dir is None:
        temp_base = tempfile.mkdtemp(prefix="license_extraction_")
    else:
        temp_base = temp_dir
        os.makedirs(temp_base, exist_ok=True)
   
    results = {}
    license_contents = []
   
    try:
        print(f"Using temporary directory: {temp_base}")
        print(f"Processing {len(packages)} packages...\n")
       
        for i, (package_name, version) in enumerate(packages, 1):
            print(f"[{i}/{len(packages)}] Processing {package_name}" + (f" v{version}" if version else " (latest)"))
           
            try:
                # Download package
                package_path = download_package(package_name, version, temp_base)
                if not package_path:
                    results[f"{package_name}---{version}"] = "Download failed"
                    continue
               
                # Extract package
                extract_path = extract_package(package_path, temp_base)
                if not extract_path:
                    results[f"{package_name}---{version}"] = "Extraction failed"
                    continue
               
                # Find and read license
                license_text = find_license_in_extracted_package(extract_path, package_name)
                if license_text:
                    license_contents.append({
                        'package': package_name,
                        'version': version or "latest",
                        'license': license_text
                    })
                    results[f"{package_name}---{version}"] = "License found"
                    print(f"    ✓ License found for {package_name}---{version}")
                else:
                    results[package_name] = "License not found"
                    print(f"    ✗ License not found for {package_name}---{version}")
                   
            except Exception as e:
                results[f"{package_name}---{version}"] = f"Error: {str(e)}"
                print(f"    ✗ Error processing {package_name}---{version}: {str(e)}")
       
        # Write combined license file
        write_combined_licenses(license_contents, output_path)
       
        # Success message with file location
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"\n🎉 SUCCESS!")
            print(f"📄 Combined licenses written to: {output_path}")
            print(f"📊 File size: {file_size:,} bytes")
            print(f"📦 Licenses found for {len(license_contents)} out of {len(packages)} packages")
        else:
            print(f"\n❌ ERROR: Failed to create output file at {output_path}")
       
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_base):
            shutil.rmtree(temp_base)
            print(f"🧹 Cleaned up temporary directory: {temp_base}")
   
    return results


def get_package_download_url(package_name: str, version: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Get download URL for package from PyPI."""
    try:
        if version:
            url = f"https://pypi.org/pypi/{package_name}/{version}/json"
        else:
            url = f"https://pypi.org/pypi/{package_name}/json"
       
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
       
        # Look for source distribution first, then wheel
        urls = data.get('urls', [])
       
        # Prefer source distribution
        for url_info in urls:
            if url_info['packagetype'] == 'sdist':
                return url_info['url'], url_info['filename']
       
        # Fallback to wheel if no source distribution
        for url_info in urls:
            if url_info['packagetype'] == 'bdist_wheel':
                return url_info['url'], url_info['filename']
       
        return None, None
       
    except Exception as e:
        print(f"    Error getting download URL for {package_name}: {e}")
        return None, None


def download_package(package_name: str, version: Optional[str], temp_dir: str) -> Optional[str]:
    """Download package to temporary directory."""
    try:
        download_url, filename = get_package_download_url(package_name, version)
        if not download_url:
            print(f"    No download URL found for {package_name}")
            return None
       
        print(f"    Downloading {filename}...")
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()
       
        package_path = os.path.join(temp_dir, filename)
        with open(package_path, 'wb') as f:
            f.write(response.content)
       
        return package_path
       
    except Exception as e:
        print(f"    Error downloading {package_name}: {e}")
        return None


def extract_package(package_path: str, temp_dir: str) -> Optional[str]:
    """Extract package to temporary directory."""
    try:
        filename = os.path.basename(package_path)
        extract_dir = os.path.join(temp_dir, f"extracted_{filename}")
        os.makedirs(extract_dir, exist_ok=True)
       
        print(f"    Extracting {filename}...")
       
        if filename.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(package_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif filename.endswith('.tar.bz2'):
            with tarfile.open(package_path, 'r:bz2') as tar:
                tar.extractall(extract_dir)
        elif filename.endswith('.zip') or filename.endswith('.whl'):
            with zipfile.ZipFile(package_path, 'r') as zip_file:
                zip_file.extractall(extract_dir)
        else:
            print(f"    Unsupported archive format: {filename}")
            return None
       
        return extract_dir
       
    except Exception as e:
        print(f"    Error extracting {package_path}: {e}")
        return None


def find_license_in_extracted_package(extract_dir: str, package_name: str) -> Optional[str]:
    """Find and read license file(s) in extracted package root directory only."""
    license_files = [
        'LICENSE', 'LICENSE.txt', 'LICENSE.md', 'LICENSE.rst', 'LICENCE.rst',
        'COPYING', 'COPYRIGHT', 'NOTICE', 'LICENCE', 'LICENCE.txt',
        'License', 'license', 'license.txt', 
        'APACHE-LICENSE-2.0.txt', 'GPL-LICENSE-2.txt',
    ]
   

    license_directories = [
        'LICENSE', 'LICENSES', 'license', 'licenses',
        'License', 'Licenses', 'LICENCE', 'LICENCES'
    ]


   
    print(f"    Searching for license files in root directory only...")
   
    # Find the actual package root (sometimes there's a nested directory)
    package_root = extract_dir
    dist_info_dir = None
   
    # Check if there's a single directory in extract_dir that contains the package
    try:
        items = os.listdir(extract_dir)
       
        # Look for .dist-info directory (wheel format)
        for item in items:
            if item.endswith('.dist-info') and os.path.isdir(os.path.join(extract_dir, item)):
                dist_info_dir = os.path.join(extract_dir, item)
                print(f"    Found .dist-info directory: {item}")
                break
       
        # If no .dist-info, check for nested package directory
        if not dist_info_dir and len(items) == 1 and os.path.isdir(os.path.join(extract_dir, items[0])):
            potential_root = os.path.join(extract_dir, items[0])
            # Check if this looks like a package directory
            if any(f in os.listdir(potential_root) for f in ['setup.py', 'pyproject.toml', 'setup.cfg'] + license_files + license_directories):
                package_root = potential_root
                print(f"    Using package root: {items[0]}/")
    except Exception:
        pass  # Use extract_dir as root if we can't determine package structure
   
    # Function to search for licenses in a given directory
    def search_licenses_in_directory(search_dir: str, dir_description: str) -> Optional[str]:
        try:
            search_items = os.listdir(search_dir)
           
            # First, check for LICENSE directories
            for item in search_items:
                if item in license_directories:
                    license_dir_path = os.path.join(search_dir, item)
                    if os.path.isdir(license_dir_path):
                        print(f"    Found license directory in {dir_description}: {item}")
                       
                        # Collect all license files from the directory
                        license_texts = []
                        try:
                            license_files_in_dir = os.listdir(license_dir_path)
                            license_files_in_dir.sort()  # Sort for consistent ordering
                           
                            for license_file in license_files_in_dir:
                                license_file_path = os.path.join(license_dir_path, license_file)
                               
                                # Skip directories within the license directory
                                if os.path.isdir(license_file_path):
                                    continue
                                   
                                # Read all files in the LICENSE directory
                                if os.path.isfile(license_file_path):
                                    try:
                                        with open(license_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                            content = f.read().strip()
                                            if content and len(content) > 20:  # Basic validation
                                                print(f"      Found license file: {license_file}")
                                                license_texts.append(f"--- {license_file} ---\n{content}")
                                    except Exception as e:
                                        print(f"      Error reading license file {license_file_path}: {e}")
                                        continue
                                       
                        except Exception as e:
                            print(f"    Error reading license directory {license_dir_path}: {e}")
                            continue
                       
                        # If we found license files in the directory, combine them
                        if license_texts:
                            combined_license = "\n\n" + "="*50 + "\n\n".join(license_texts)
                            return combined_license
           
            # If no LICENSE directory found, search for individual license files
            for file in search_items:
                if file in license_files:
                    license_path = os.path.join(search_dir, file)
                    if os.path.isfile(license_path):
                        try:
                            with open(license_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read().strip()
                                if content and len(content) > 50:  # Basic validation
                                    print(f"    Found license file in {dir_description}: {file}")
                                    return content
                        except Exception as e:
                            print(f"    Error reading license file {license_path}: {e}")
                            continue
                           
        except Exception as e:
            print(f"    Error accessing {dir_description}: {e}")
       
        return None
   
    # If we found a .dist-info directory (wheel format), search there first
    if dist_info_dir:
        result = search_licenses_in_directory(dist_info_dir, ".dist-info directory")
        if result:
            return result
   
    # Search in the main package root
    result = search_licenses_in_directory(package_root, "package root")
    if result:
        return result
   
    return None


def write_combined_licenses(license_contents: List[Dict], output_path: str):
    """Write all licenses to a combined file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("COMBINED THIRD-PARTY SOFTWARE LICENSES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"This file contains the licenses of {len(license_contents)} packages.\n")
        f.write(f"Generated automatically on {os.path.basename(__file__) if '__file__' in globals() else 'license downloader'}.\n")
        f.write(f"Output location: {output_path}\n\n")
        f.write("=" * 70 + "\n\n")
       
        for i, license_info in enumerate(license_contents, 1):
            f.write(f"{i}. {license_info['package']} (v{license_info['version']})\n")
            f.write("=" * 70 + "\n")
            f.write(license_info['license'])
            f.write("\n\n" + "=" * 70 + "\n\n")


def print_results_summary(results: Dict[str, str]):
    """Print a nice summary of the results."""
    print("\n" + "=" * 50)
    print("📊 PROCESSING SUMMARY")
    print("=" * 50)
   
    success_count = sum(1 for status in results.values() if status == "License found")
    total_count = len(results)
   
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
   
    if success_count < total_count:
        print("\nFailed packages:")
        for package, status in results.items():
            if status != "License found":
                print(f"  • {package}: {status}")
   
    print("=" * 50)




if __name__ == "__main__":
    print("Welcome to License Collector!")
    print("=" * 50)
    print("This tool identifies external packages, collects their corresponding license files, and aggregates them into a single report.")

    print("Please enter the path to the root of your project folder.")
    print("For example, if your project looks like this:")

    print(r"""
    my_project/
    ├── src/
    │   └── main.py
    ├── data/
    │   └── input.csv
    └── README.md
    """)

    print("These are examples of project root path in different operating systems:")
    print(r"In Windows => C:\Users\YourName\Documents\my_project\ ")  
    print(r"In Linux/Mac => /home/yourname/projects/my_project/")     

    project_root_path = input("\nPath: ")
    print(project_root_path)

    license_file_name = input("\nPlease enter the name of the aggregated license file (e.g., LICENSES.txt): ")

    results = download_and_extract_licenses(cpp(project_root_path), f"{project_root_path}{license_file_name}")

    print(results)
    print_results_summary(results)
