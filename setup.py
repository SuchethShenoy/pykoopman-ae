from setuptools import setup, find_packages

setup(
    name='pykoopman_ae',
    version='0.1.2',
    description='Python package PyKoopman-AE for the data-driven identification of the Koopman Operator-based models for dynamical systems using autoencoders.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sucheth Shenoy',
    author_email='sucheth17@gmail.com',
    url='https://github.com/SuchethShenoy/pykoopman-ae',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
)
