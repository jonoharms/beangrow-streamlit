import setuptools

setuptools.setup(
    name="beangrow",
    packages=setuptools.find_packages(exclude=["beangrow_tests"]),
    entry_points={
        'console_scripts': [
            'beangrow-configure=beangrow.configure:main',
            'beangrow-returns=beangrow.compute_returns:main',
        ],
    }
)
