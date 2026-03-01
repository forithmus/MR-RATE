from setuptools import setup, find_packages

setup(
    name='vision-encoder',
    packages=find_packages(exclude=[]),
    include_package_data=True,
    version='0.1',
    description='Vision encoder for MR-RATE (VJEPA2, CTViT)',
    install_requires=[
        'torch>=2.0',
        'torchvision',
        'einops>=0.6',
        'transformers>=4.45',
        'peft>=0.7',
        'accelerate>=0.25',
        'beartype',
        'ema-pytorch>=0.2.2',
        'torchtyping',
        'numpy',
        'nibabel',
        'opencv-python',
        'Pillow',
        'sentencepiece',
        'tqdm',
        'openpyxl',
    ],
)
