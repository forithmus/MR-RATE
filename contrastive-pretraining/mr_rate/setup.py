from setuptools import setup, find_packages

setup(
    name='mr-rate',
    packages=find_packages(exclude=[]),
    include_package_data=True,
    version='0.1',
    description='MR-RATE: Contrastive vision-language model for brain MRI',
    install_requires=[
        'torch>=2.0',
        'torchvision',
        'einops>=0.6',
        'transformers>=4.45',
        'beartype',
        'ftfy',
        'regex',
        'numpy',
        'nltk',
    ],
)
