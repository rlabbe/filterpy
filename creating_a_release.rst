Steps to Create Release
=======================
* update filterpy/filterpy/__init__.py with the version number.

* 'rm *' in dist

* If necessary, edit filterpy/docs/index.rst to add any classes. Add .rst file for those new classes to the /docs subdirectories.

* In /docs, run 'make html'. Inspect docs/_build/html/index.html for correctness.

* Once docs are good, commit to git.

* tag with 'git tag -a v0.1.23 -m "version 0.1.23"

* push to origin. That automatically triggers a build on readthedocs.org.

* Update pypi.org with 'bash pypi-install.sh'



