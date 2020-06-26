PYTHON=python
path=examples
recursive=True

make:
	@echo Installing pyross...
	${PYTHON} setup.py install
	@echo adding githook...
	cp .githooks/pre-push .git/hooks/
	chmod +x .git/hooks/pre-push

clean-local:
	@echo removing local compiled files
	rm pyross/*.c pyross/*.html pyross/*.cpp

clean:
	@echo removing all compiled files
	${PYTHON} setup.py clean
	rm pyross/*.c pyross/*.html pyross/*.cpp
	
env:
	@echo creating conda environment...
	conda env create --file environment.yml
	# conda activate pyross
	@echo use make to install pyross

test:
	@echo testing pyross...
	cd tests && python quick_test.py

nbtest:
	@echo testing example notebooks...
	@echo test $(path)
	cd tests && python notebook_test.py --path $(path) --recursive $(recursive)


pypitest:
	@echo testing pystokes...
	python setup.py sdist bdist_wheel
	python -m twine upload --repository testpypi dist/*

pypi:
	@echo testing pystokes...
	python setup.py sdist bdist_wheel	
	python -m twine upload dist/*

cycov:
	python setup.py build_ext --force --inplace --define CYTHON_TRACE
	pytest tests/quick_test.py --cov=./ --cov-report=xml
