python setup.py sdist --formats=gztar,zip
python setup.py register -r test
#python setup.py sdist upload -r test
twine upload dist/* -r test