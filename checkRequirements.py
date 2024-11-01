import importlib.metadata

def check_Requirements(requirements_file):
    # Read the required packages from the requirements file
    with open(requirements_file, 'r') as file:
        required_packages = [line.strip() for line in file if line.strip()]

    # Get a set of installed package names (case-insensitive)
    installed_packages = {pkg.metadata['Name'].lower() for pkg in importlib.metadata.distributions()}

    for package in required_packages:
        package_name = package.split('==')[0].lower()  # Extract package name and convert to lowercase
        if package_name in installed_packages:
            print(f"{package_name} is INSTALLED")
        else:
            print(f"{package_name} is not installed")

if __name__ == "__main__":
    check_Requirements('requirements.txt')

