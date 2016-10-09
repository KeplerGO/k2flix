# Install k2flix
install:
	python setup.py install

# Upload new version to pypi
publish:
	python setup.py publish

# Convert the README from rst to md for use by the GitHub website creator
md: README.rst
	pandoc --from=rst --to=markdown README.rst > README.md

