python setup.py sdist --formats=zip
python setup.py register -r pypi
twine upload dist/* -r pypi