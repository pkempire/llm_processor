from setuptools import setup, find_packages
import pathlib

# Get requirements from requirements.txt
here = pathlib.Path(__file__).parent.resolve()
requirements = (here / 'requirements.txt').read_text(encoding='utf-8').splitlines()


setup(
    name="llm_processor",
    version="0.1.0",
    packages=find_packages(include=['llm_processor', 'llm_processor.*']),
    package_dir={"llm_processor": "llm_processor"},
    install_requires=[
        "openai>=1.0.0",
        "tqdm>=4.65.0", 
        "pandas>=2.0.0",
        "tiktoken>=0.5.0",
        "requests>=2.28.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="High-performance batch processing library for LLM operations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm_processor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    package_data={
        "llm_processor": ["*.json", "*.yaml"]
    }
)
