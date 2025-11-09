from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                # Remove inline comments
                if ';' in line:
                    # Keep platform-specific markers
                    requirements.append(line)
                else:
                    requirements.append(line)
    return requirements

setup(
    name='eye-contact-cnn',
    version='0.1.0',
    description='A deep learning-based system for real-time gaze redirection in video streams',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/eye-contact-cnn',
    packages=find_packages(exclude=['tests', 'docs']),
    install_requires=read_requirements(),
    python_requires='>=3.6,<3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Video',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
    keywords='gaze-redirection eye-contact cnn deep-learning computer-vision',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/eye-contact-cnn/issues',
        'Source': 'https://github.com/yourusername/eye-contact-cnn',
    },
    entry_points={
        'console_scripts': [
            'eye-contact-cnn=gaze_correction_system.regz_socket_MP_FD:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
