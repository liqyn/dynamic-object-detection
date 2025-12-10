from setuptools import setup, find_packages

setup(
    name="dynamic_object_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy<2",
        "opencv-python",
        "torch>=2.8.0",
        "git+https://github.com/facebookresearch/segment-anything.git"
        "torchvision",
        "matplotlib",
        "tensorboard",
        "scipy",
        "opencv-python",
        "dataclasses",
        "open3d",
        "tqdm"
    ],
)