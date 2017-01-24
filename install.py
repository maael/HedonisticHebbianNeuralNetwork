"""
Basic install script.
Using a script instead of requirements.txt to avoid having
to install numpy again on CI, which takes forever.
"""

import os
import pip

ALL = [
  'pytest == 3.0.6',
  'pylint == 1.6.5'
]

NONCI = [
  'numpy == 1.11.2'
]

def install(packages):
  for package in packages:
    pip.main(['install', package])

if __name__ == '__main__':
  install(ALL)
  if 'CI' not in os.environ or os.environ['CI'] != 'true':
    install(NONCI)
