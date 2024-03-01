from setuptools import setup
import os
import glob


package_name = 'percept_wmodel'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib/python3.8/site-packages/'+ package_name),glob.glob(package_name+'/*.torch')) # this is needed to be able to read the .torch files
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='',
    maintainer_email='',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'imseg_to_cobj = percept_wmodel.imseg_to_cobj:main',
        	'add_frame = percept_wmodel.add_frame:main',
        	'listener_to_csv = percept_wmodel.listener_to_csv:main',
        ],
    },
)
