import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="finoptions",
    version="0.1.4",
    author="Ben Cho",
    license="MIT License",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    author_email="ben.cho@gmail.com",
    description="Energy derivatives (futures, options etc...)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.csv", "*.json", "*.geojson"]},
    keywords=[
        "Energy",
        "Risk",
        "Crude",
        "Trading",
        "Petroleum",
        "Oil",
        "Refinery",
        "Refined Products",
        "Products",
    ],
    url="https://github.com/bbcho/finoptions-dev",
    # download_url="https://github.com/bbcho/finoptions-dev/archive/refs/heads/main.zip",
    install_requires=["scipy>=1.7", "numpy", "numdifftools", "matplotlib"],
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
    ],
    python_requires=">=3.6",
)
