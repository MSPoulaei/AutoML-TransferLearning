from setuptools import setup, find_packages
from pathlib import Path


# Read requirements from requirements.txt
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path, encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                requirements.append(line)
        return requirements


setup(
    name="transfer-learning-orchestrator",
    version="1.0.0",
    description="Multi-Agent Transfer Learning Orchestration System with LLM",
    author="AI Research Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "tl-orchestrator=main:app",
        ],
    },
)
