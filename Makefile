PYTHON=python
#PYTHON=path to your python installation

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
ifdef path
	cd tests && python notebook_test.py --path $(path)
else
	cd tests && python notebook_test.py --path examples
endif
