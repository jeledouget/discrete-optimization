from setuptools import setup, find_packages

DISTNAME = 'discopt'
DESCRIPTION = 'Discrete optimization'
AUTHOR = 'Jean-Eudes Le Douget'
AUTHOR_EMAIL = 'jeaneudes.ledouget+discopt@gmail.com'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL
DOWNLOAD_URL = 'https://github.com/jeledouget/discrete-optimization'

MIN_PYTHON_VERSION = '3.7'

INSTALL_REQUIRES = [
    'timeout_decorator',
    'numpy',
    'pandas',
    'pydantic',
    'matplotlib'
]

setup(
    name=DISTNAME,
    description=DESCRIPTION,
    url=DOWNLOAD_URL,
    download_url=DOWNLOAD_URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    keywords='coursera discrete optimization',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    platforms='any',
    python_requires='>={}'.format(MIN_PYTHON_VERSION)
)
