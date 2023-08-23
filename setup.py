#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Setup denoisenet libarary."""

import os
import pip
import sys

from distutils.version import LooseVersion
from setuptools import find_packages
from setuptools import setup

if LooseVersion(sys.version) < LooseVersion("3.6"):
    raise RuntimeError(
        "denoisenet requires Python>=3.6, "
        "but your Python is {}".format(sys.version))

requirements = {
    "install": [
        "torch>=1.6",
        "torchaudio",
        "setuptools>=38.5.1",
        "PyYAML>=3.12",
        "numpy",
        "scipy",
        "soundfile",
        "librosa>=0.10",
    ],
    "setup": [
    ],
    "test": [
    ]
}
entry_points = {
    "console_scripts": [
        "denoisenet-preprocess=denoisenet.bin.preprocess:main",
        "denoisenet-train=denoisenet.bin.train:main",
        "denoisenet-infer=denoisenet.bin.infer:main",
        "denoisenet-export=denoisenet.bin.export:main",
    ]
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {k: v for k, v in requirements.items()
                  if k not in ["install", "setup"]}

dirname = os.path.dirname(__file__)
exec(open(os.path.join(dirname, "denoisenet/version.py")).read())
setup(name="denoisenet",
      version=__version__,
      url="http://github.com/wwyuan2023/DenoiseNet.git",
      author="wuwen.yww",
      author_email="yuanwuwen@126.com",
      description="DenoiseNet implementation",
      long_description=open(os.path.join(dirname, "README.md"),
                            encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      license="MIT License",
      packages=find_packages(include=["denoisenet*"]),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      entry_points=entry_points,
      include_package_data=True,
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Intended Audience :: Science/Research",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: MIT License",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      )
