python setup.py sdist --formats=gztar,zip
python setup.py register -r pypi
twine upload dist/* -r pypi