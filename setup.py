import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dotaservice",
    version="0.0.2",
    author="Tim Zaman",
    author_email="timbobel@gmail.com",
    description="DotaService is a service to play Dota 2 through gRPC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TimZaman/DotaService",
    packages=setuptools.find_packages(),
    package_data={'dotaservice': ['lua/*.lua']},
    python_requires='>=3.7',
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: Freeware",
        "License :: Other/Proprietary License",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development",
    ],
    install_requires=[
        'grpclib',
        'protobuf',
    ],
    extras_require={
        'dev': [
            'grpcio-tools',
        ],
    },
)
