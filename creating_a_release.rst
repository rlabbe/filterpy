Steps to Create Release
=======================

* run pytest

* run pylint --disable=similarities --disable=R0205 filterpy
  R0205 turns off warning about deriving from object. We still support 2.7, so it is needed

* update filterpy/filterpy/__init__.py with the version number.

* update filterpy/filterpy/changelog.txt with the changes for this release.

* 'rm *' in dist

* If necessary, edit filterpy/docs/index.rst to add any classes. Add .rst file for those new classes to the /docs subdirectories.

* In /docs, run 'make html'. Inspect docs/_build/html/index.html for correctness.

* Once docs are good, commit to git.

* tag with 'git tag -a 0.1.23 -m "version 0.1.23"

* push to origin. That automatically triggers a build on readthedocs.org.

* push tags to origin with git push origin --tags

* Update pypi.org with 'bash pypi-install.sh'

* You need to manually update the documentation code at pythonhosted, PyPi's documentation server.

    cd /docs/_build/html
    zip -r filterpy.zip *.*
    
    add all files to a zip file (index.html must be at base)
    go to https://pypi.python.org/pypi?%3Aaction=pkg_edit&name=filterpy
    scroll to bottom, add the zip file you just made
    click 'Upload Documentation button'

    it usually takes several minutes for the documentation to show up here:
    https://pythonhosted.org/filterpy/
    
    
    

