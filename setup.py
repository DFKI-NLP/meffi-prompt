from setuptools import setup, find_packages

setup(
    name="meffi_prompt",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    version="1.0.0",
    license="MIT",
    description="Implementation of Multilingual Relation Classification via Efficient and Effective Prompting",
    author="Yuxuan Chen",
    author_email="yuxuan.chen@dfki.de",
    url="https://github.com/DFKI-NLP/meffi-prompt",
    keywords=[
        "artificial intelligence",
        "natural language processing",
        "prompt learning",
        "relation extraction",
        "information extraction",
        "multilinguality"
    ],
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "sklearn",
        "tqdm",
        "hydra-core",
        "omegaconf",
        "sentencepiece",
        "protobuf",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
