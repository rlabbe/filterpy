python setup.py sdist --formats=gztar,zip
python setup.py register -r test
twine upload dist/* -r test