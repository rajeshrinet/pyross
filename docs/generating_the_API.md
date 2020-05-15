# How to generate the API


## Setup Sphinx

Follow [this](https://medium.com/@eikonomega/getting-started-with-sphinx-autodoc-part-1-2cebbbca5365)
guide to initialize Sphinx.

## Generate API

First compile pyross in-place:

```bash
python setup.py build_ext --inplace
```

Then, in the `docs/sphinx` folder, run

```bash
sphinx-build -b html . ../API
```

this creates the `docs/API` folder.

## Generate markdown API

Run this instead:

```bash
sphinx-build -b markdown . ../API
```

Add a table of contents in VSCode by first installing *Markdown All in One*
and then using the command `Markdown: Create Table of Contents`.

