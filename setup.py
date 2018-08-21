from setuptools import setup

setup(name='implementations_bachelor_thesis_vr',
      version='1.0',
      description='Implementations for the bachelor thesis of Vincent Rolfs',
      author='Vincent Rolfs',
      author_email='v.rolfs@studium.uni-hamburg.de',
      license='GNU General Public License Version 3',
      packages=['implementations_bachelor_thesis_vr'],
      install_requires=[
          'numpy',
          'terminaltables',
          'matplotlib',
          'pandas'
      ],
      zip_safe=False)
