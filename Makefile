PYTHON=python

make:
	@echo Installing pyross...
	${PYTHON} setup.py install
	@echo adding githook...
	cp .githooks/pre-push .git/hooks/
	chmod +x .git/hooks/pre-push


clean:
	@echo removing all compiled files
	${PYTHON} setup.py clean
	rm pyross/*.c pyross/*.html
	rm -r pyross/__pycache_
	
env:
	@echo creating conda environment...
	conda env create --file environment.yml
	conda activate pyross
	@echo use make to install pyross

test:
	@echo testing pyross...
	cd ~ && python -m unittest pyross.test
