import pathlib

from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="dad_torch",
    version="0.5",
    description="Easy Neural Network Experiments with pytorch",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/trendscenter/dad-torch",
    author="Aashis Khana1",
    author_email="sraashis@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    packages=['dad_torch', 'dad_torch.config', 'dad_torch.data', 'dad_torch.distrib', 'dad_torch.metrics',
              'dad_torch.utils', 'dad_torch.vision'],
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'scikit-learn', 'scikit-image',
                      'pillow', 'matplotlib', 'opencv-python-headless', 'pandas', 'seaborn']
)
