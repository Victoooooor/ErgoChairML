from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'ErgoChairML'
LONG_DESCRIPTION = 'Scrap Image from Google, Preprocess and generate dataset, train pix2pix'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="ErgoChairML",
    version=VERSION,
    author="Victor Zhang",
    author_email="<victorzh716@email.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'imgaug',
        'tensorflow',
        'matplotlib',
        'pandas',
        'tqdm',
        'Pillow',
        'python-magic',
        'progressbar',
        'IPython',
        'cv2'
        'keras'

    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python', 'tensorflow', 'pix2pix', 'MoveNet', 'Mask R-CNN', 'Image Segmentation', 'GAN'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux :: ArchLinux",
        "Operating System :: Microsoft :: Windows",
    ]
)