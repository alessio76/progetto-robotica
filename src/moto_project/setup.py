from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    #name of your package (and also the name you used for the folder which contains the Python module)
    packages=['moto_project'],
    #the folder where the module to install is located
    package_dir={'': 'src'}
)

setup(**d)
