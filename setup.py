from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name                = 'alias_free_torch',
    version             = '0.0.6',
    description         = 'alias free torch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author              = 'junjun3518',
    author_email        = 'junjun3518@gmail.com',
    url                 = 'https://github.com/junjun3518/alias-free-torch',
    install_requires    =  [],
    packages            = find_packages(where = "src"),
    package_dir         = {"": "src"},
    keywords            = ['alias','torch','pytorch','filter'],
    python_requires     = '>=3',
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

