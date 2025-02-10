from setuptools import setup, find_packages

setup(
    name="autocut",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "scikit-image",
        "imageio",
        "huggingface_hub",
        "transformers>=4.39.1",
    ],
    entry_points={
        'console_scripts': [
            'autocut=src.autocut:main',
        ],
    },
    author="Your Name",
    description="AI powered image background removal tool",
    python_requires=">=3.12",
)
