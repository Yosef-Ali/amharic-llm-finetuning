from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="amharic-hnet",
    version="0.1.0",
    author="Amharic H-Net Contributors",
    author_email="your.email@example.com",
    description="Improved transformer model for Amharic language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Amharic-Hnet-Qwin",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: Amharic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "amharic-hnet-train=amharic_hnet.cli.train:main",
            "amharic-hnet-generate=amharic_hnet.cli.generate:main",
            "amharic-hnet-evaluate=amharic_hnet.cli.evaluate:main",
            "amharic-hnet-optimize=amharic_hnet.cli.optimize:main",
            "amharic-hnet-deploy=amharic_hnet.cli.deploy:main",
        ],
    },
    include_package_data=True,
)