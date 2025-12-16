from setuptools import setup, find_packages

setup(
    name="transfer-learning-orchestrator",
    version="1.0.0",
    description="Multi-Agent Transfer Learning Orchestration System with LLM",
    author="AI Research Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.5.0",
        "pydantic-ai>=0.0.24",
        "openai>=1.12.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.12",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "sqlalchemy>=2.0.0",
        "rich>=13.7.0",
        "loguru>=0.7.2",
        "typer>=0.9.0",
        "tenacity>=8.2.3",
    ],
    entry_points={
        "console_scripts": [
            "tl-orchestrator=main:app",
        ],
    },
)
