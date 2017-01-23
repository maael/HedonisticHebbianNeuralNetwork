import os
import pip

_all_ = [
  'pytest == 3.0.6'
]

_nonci_ = [
  'numpy == 1.11.2'
]

def install(packages):
  for package in packages:
    pip.main(['install', package])

if __name__ == '__main__':
  install(_all_)
  if 'CI' not in os.environ or os.environ['CI'] != 'true':
    install(_nonci_)