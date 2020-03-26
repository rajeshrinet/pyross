## PyRoss: Infectious diseases models in Python [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rajeshrinet/pyross/master?filepath=examples)

[About](#about) | [News](#news) | [Installation](#installation) | [Examples](#examples) | [Publications ](#publications)| [Support](#support) | [License](#license)

![Imagel](examples/banner.png)


## About

[PyRoss](https://gitlab.com/rajeshrinet/pyross) is a Cython library for simulating infectious diseases. Currently the library supports structured compartment models formulated as systems of differential equations. 

The library was developed to model the outbreak of the novel coronavirus COVID-19 and to assess the age-structured impact of social distancing measures in India. 

The library is named after [Sir. Ronald Ross](https://en.wikipedia.org/wiki/Ronald_Ross), doctor, mathematician and poet. In 1898 he made "the great discovery" in his laboratory in Calcutta "that malaria is conveyed by the bite of a mosquito".  He won the Nobel Prize in 1902 and laid the foundations of the mathematical modelling of infectious diseases. 


## News


## Installation
Clone (or download) the repository and use a terminal to install using 

```bash
>> git clone https://github.com/rajeshrinet/pyross.git
>> cd pyross
>> python setup.py install
```

PyRoss requires the following software 

- Python 2.6+ or Python 3.4+
- [Cython 0.25.x+](http://docs.cython.org/en/latest/index.html) |  [Matplotlib 2.0.x+](https://matplotlib.org) | [NumPy 1.x+](http://www.numpy.org) |  [Pandas](https://pandas.pydata.org/) | [SciPy 1.1.x+](https://www.scipy.org/) 

## Data sources

The age and social contact data that is needed to construct structured compartment models can be found at the following sources:

**Age structure:** [Population Pyramid](https://www.populationpyramid.net/) website. 

**Contact structure:** *Projecting social contact matrices in 152 countries using contact surveys and demographic data*, Kiesha Prem, Alex R. Cook, Mark Jit, PLOS Computational Biology, (2017) [DOI]( https://doi.org/10.1371/journal.pcbi.1005697), [Supporting Information Text](https://doi.org/10.1371/journal.pcbi.1005697.s001)  and [Supporting Information Data](https://doi.org/10.1371/journal.pcbi.1005697.s001).


**COVID-19 cases:** [Worldometers](https://www.worldometers.info/coronavirus) website.


## Examples
```Python
#Ex1: M=1
import numpy as np
import pyross

M = 1                  # age groups
Ni = 1000*np.ones(M)   # population in each group
N = np.sum(Ni)

Ia0, Is0, R0 = 0, 1, 0
S0 = N - Ia0 - R0 - Is0

alpha, beta, gamma = 0, 0.2, 1./10

Tf = 160;   C = np.zeros((2*Tf, M*M))

for i in range(2*Tf):
    C[i, :] = np.identity(M).reshape(M*M)

fsa=1
model = pyross.models.SIR(S0, Ia0, Is0, alpha, beta, gamma, fsa, M, Ni, Tf)

#run simulation and save data
Nf=160; filename='this.mat'
model.simulate(C, Nf, filename)
```


## Publications

* *Age-structured impact of social distancing on the COVID-19 epidemic in India*, Rajesh Singh and R. Adhikari, 2020 (working paper).


## Support

* For help with and questions about PyRoss, please post to the [pyross-users](https://groups.google.com/forum/#!forum/pyross) group.
* For bug reports and feature requests, please use the [issue tracker](https://github.com/rajeshrinet/pyross/issues) on GitHub.

## License
We believe that openness and sharing improves the practice of science and increases the reach of its benefits. This code is released under the [MIT license](http://opensource.org/licenses/MIT). Our choice is guided by the excellent article on [Licensing for the scientist-programmer](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1002598). 
