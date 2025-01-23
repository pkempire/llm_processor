from setuptools import setup, find_packages

setup(
    name="llm_processor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "tiktoken>=0.5.0"
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
) 