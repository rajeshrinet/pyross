![Imagel](https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/others/banner.jpg)


## PyRoss: inference, forecasts, and optimised control for epidemiological models in Python [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rajeshrinet/pyross/master?filepath=binder) ![CI](https://github.com/rajeshrinet/pyross/workflows/CI/badge.svg) ![Notebooks](https://github.com/rajeshrinet/pyross/workflows/Notebooks/badge.svg)  [![PyPI](https://img.shields.io/pypi/v/pyross.svg)](https://pypi.python.org/pypi/pyross) [![Python Version](https://img.shields.io/pypi/pyversions/pyross)](https://pypi.org/project/pyross) [![Downloads](https://pepy.tech/badge/pyross)](https://pepy.tech/project/pyross)  ![stars](https://img.shields.io/github/stars/rajeshrinet/pyross) ![forks](https://img.shields.io/github/forks/rajeshrinet/pyross) ![License](https://img.shields.io/github/license/rajeshrinet/pyross) 


[About](#about) | 
[Documentation](https://pyross.readthedocs.io/en/latest/) | 
[Examples](#examples) | 
[Installation](#installation) | 
[Publications ](#publications) | 



## About

[PyRoss](https://github.com/rajeshrinet/pyross) is a numerical library that offers an integrated platform for **inference**, **forecasts** and **non-pharmaceutical interventions** in structured epidemiological compartment models. 

**Compartment models** of arbitrary complexity can be  **user-defined** through Python dictionaries. The most common epidemiological models, and several less common ones, come pre-defined with the library. Models can include **stages** to allow for non-exponentially distributed compartmental residence times. Currently,  [pre-defined models](https://github.com/rajeshrinet/pyross/blob/master/docs/models.pdf) include ones with multiple disease states (exposed, asymptomatic, symptomatic, etc) and may be further divided by age, and by objective medical states (hospitalized, in ICU, etc). The compartment framework supports models for **disease surveillance** and **quarantine** and a variety of other processes of epidemiological relevance. 

**Generative processes** can be formulated **stochastically** (as Markov population processes) or **deterministically** (as systems of differential equations). Population processes are sampled exactly by the Doob-Gillespie algorithm or approximately by the tau-leaping algorithm while differential equations are integrated by both fixed and adaptive time-stepping. A **hybrid algorithm** transits dynamically between these  depending on the magnitude of the compartmental fluctuations. 

**Bayesian inference** on pre-defined or user-defined models is performed using model-adapted **Gaussian processes**  derived from **functional limit theorems** for Markov  population process. Generative models are fitted to data through the surveillance model allowing for possibily **unobserved** compartments. The **MAP** estimates of parameters and their standard errors can be obtained rapidly by optimising, obviating the need for expensive Markov chain Monte Carlo. This enables the fast evaluation of the **model evidence**, through which competing models may be **objectively compared** and their forecasts combined by **Bayesian model averaging**. Forecasts of disease progression, then, can be **fully Bayesian**, convolving uncertainties in data, parameters and models. The sensitivity of these forecasts is estimated through the **Fisher information matrix**. 

**Non-pharmaceutical interventions** are implemented as modifications of the **contact structures** of the model. **Optimised control** of these structures, given **cost functions**, is possible. 

[PyRossGeo](https://github.com/lukastk/PyRossGeo) is a companion library that supports **spatially resolved compartment models** with explicit **commuting networks**. 

The libraries are named after [Sir Ronald Ross](https://en.wikipedia.org/wiki/Ronald_Ross), doctor, mathematician and poet. In 1898 he made "the great discovery" in his laboratory in Calcutta "that malaria is conveyed by the bite of a mosquito".  He won the Nobel Prize in 1902 and laid the foundations of the mathematical modelling of infectious diseases.


The library was developed as a part of [The Rapid Assistance in Modelling the Pandemic (RAMP)](https://royalsociety.org/news/2020/03/urgent-call-epidemic-modelling/) taskforce at the **University of Cambridge**. In alphabetical order, the authors are:
[Ronojoy Adhikari](https://github.com/ronojoy),
[Austen Bolitho](https://github.com/TakodaS),
[Erik Brorson](https://github.com/erikbrorson),
[Fernando Caballero](https://github.com/fcaballerop), 
[Michael Cates](http://www.damtp.cam.ac.uk/person/mec22),
[Jakub Dolezal](https://github.com/JakubJDolezal),
[Tim Ekeh](https://github.com/tekeh),
[Jules Guioth](https://orcid.org/0000-0001-5644-3044), 
[Robert Jack](https://github.com/rljack2002),
[Julian Kappler](https://github.com/juliankappler),
[Lukas Kikuchi](https://github.com/lukastk),
[Hideki Kobayashi](https://github.com/hidekb),
[Irene Li](https://github.com/Irene-Li),
[Joseph Peterson](https://github.com/jdpeterson3/),
[Patrick Pietzonka](https://github.com/ppietzonka),
[Benjamin Remez](https://github.com/BenjaminRemez),
[Paul Rohrbach](https://github.com/prohrbach),
[Rajesh Singh](https://github.com/rajeshrinet), 
and [Günther Turk](https://github.com/phi6GTurk). 
PyRoss development was also partially supported by a [Microsoft Research Award](https://www.microsoft.com/en-us/research/collaboration/studies-in-pandemic-preparedness/#!projects) for "Building an open platform for pandemic modelling".

Please read the  [PyRoss paper](https://arxiv.org/abs/2005.09625) and [PyRoss Wiki](https://github.com/rajeshrinet/pyross/wiki/) before you use PyRoss for your research. [Open an issue](https://github.com/rajeshrinet/pyross/issues), in preference to emailing us with queries. Join our [Slack channel](https://join.slack.com/t/pyross/shared_invite/zt-e8th6kcz-S4b_oJIZWPsGLruSPl3Zuw) for discussion. Please follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/0/code_of_conduct/) in all PyRoss fora. Thank you!

## Installation
You can take PyRoss for a spin **without installation**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rajeshrinet/pyross/master?filepath=binder). Please be patient while [Binder](https://mybinder.org/v2/gh/rajeshrinet/pyross/master?filepath=binder) loads.

### From a checkout of  PyRoss GitHub repository 
This is the recommended way as it downloads a whole suite of examples along with the package. 

#### Install PyRoss and an extended list of dependencies using 

```bash
>> git clone https://github.com/rajeshrinet/pyross.git
>> cd pyross
>> pip install -r requirements.txt
>> python setup.py install
```

#### Install PyRoss and an extended list of dependencies, via [Anaconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html), in an [environment](https://github.com/rajeshrinet/pyross/blob/master/environment.yml) named `pyross`:

```bash
>> git clone https://github.com/rajeshrinet/pyross.git
>> cd pyross
>> make env
>> conda activate pyross
>> make
```

### Via pip

Install the latest [PyPI](https://pypi.org/project/pyross) version

```bash
>> pip install pyross
```




### Testing
Short test of initialisation and running

```bash
>> make test
```

Long test of all example notebooks. Optionally can specify path and recursion
to test a certain subset of notebooks

```bash
>> make nbtest -e path=examples/deterministic/
```



## Examples

PyRoss has model-agnostic, formulation-agnostic intuitive interface. Once a model is instantiated, stochastic, deterministic and hybrid simulations can be performed through the same interface. The example below shows how to set up a deterministic SIR simulation with three age-groups. 

```Python 
# SIR with three age-groups (M=3)

import numpy as np
import pyross
import matplotlib.pyplot as plt


model_spec = { "classes" : ["S", "I"],

             "S" : {"infection" : [ ["I","S", "-beta"] ]},  ## the I class passes infection to S class
             "I" : { "linear"    : [ ["I", "-gamma"] ],     ## this is recovery process for I class
                    "infection" : [ ["I", "S", "beta"]]}    
             
              ## the recovered class R is internally determined by number conservation
             }


parameters = {'beta'  : 0.1,
              'gamma' : 0.1, 
              }

M=3;  Ni=1000*np.ones(M);  N=np.sum(Ni) 


# Initial conditions as an array
x0 = np.array([
    980, 980, 980,    # S
    20,   20,  20,    # I
])

# Or initial conditions as a dictionary 
x0 = {'S': [n-20 for n in Ni], 'I':  [20, 20, 20]  }


CM = np.array( [[1,   0.5, 0.1],
               [0.5, 1,   0.5],
               [0.1, 0.5, 1  ]], dtype=float)

def contactMatrix(t):
    return CM


# duration of simulation and data file
Tf = 160;  Nf=Tf+1; 

det_model = pyross.deterministic.Model(model_spec, parameters, M, Ni)

# simulate model 
data = det_model.simulate(x0, contactMatrix, Tf, Nf)


# plot the data and obtain the epidemic curve
S = np.sum(det_model.model_class_data('S', data), axis=1)
I = np.sum(det_model.model_class_data('I', data), axis=1)
R = np.sum(det_model.model_class_data('R', data), axis=1)
t = data['t']

fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})

plt.fill_between(t, 0, S/N, color="#348ABD", alpha=0.3)
plt.plot(t, S, '-', color="#348ABD", label='$S$', lw=4)

plt.fill_between(t, 0, I/N, color='#A60628', alpha=0.3)
plt.plot(t, I, '-', color='#A60628', label='$I$', lw=4)

plt.fill_between(t, 0, R/N, color="dimgrey", alpha=0.3)
plt.plot(t, R, '-', color="dimgrey", label='$R$', lw=4)

plt.legend(fontsize=26); plt.grid() 
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel('Compartment value')
plt.xlabel('Days');
```
Read more in the [examples](https://github.com/rajeshrinet/pyross/tree/master/examples) folders.


## Publications
* **Bayesian inference across multiple models suggests a strong increase in lethality of COVID-19 in late 2020 in the UK**. Patrick Pietzonka, Erik Brorson, William Bankes, Michael E. Cates, Robert L. Jack, R. Adhikari [medRxiv, 2021](https://doi.org/10.1101/2021.03.10.21253311)

* **Efficient Bayesian inference of fully stochastic epidemiological models with applications to COVID-19**. Yuting I. Li, Günther Turk, Paul B. Rohrbach, Patrick Pietzonka, Julian Kappler, Rajesh Singh, Jakub Dolezal, Timothy Ekeh, Lukas Kikuchi, Joseph D. Peterson, Hideki Kobayashi, Michael E. Cates, R. Adhikari, Robert L. Jack, [arXiv:2010.11783, 2020](https://arxiv.org/abs/2010.11783)  | [ResearchGate](https://www.researchgate.net/publication/344828080_Efficient_Bayesian_inference_of_fully_stochastic_epidemiological_models_with_applications_to_COVID-19)

* **Efficient and flexible methods for time since infection models**, Joseph D. Peterson, R. Adhikari, [arXiv:2010.10955, 2020](https://arxiv.org/abs/2010.10955)

* **Inference, prediction and optimization of non-pharmaceutical interventions using compartment models: the PyRoss library**. R. Adhikari, Austen Bolitho, Fernando Caballero, Michael E. Cates,
  Jakub Dolezal, Timothy Ekeh, Jules Guioth, Robert L. Jack, Julian Kappler,
  Lukas Kikuchi, Hideki Kobayashi, Yuting I. Li, Joseph D. Peterson, Patrick
  Pietzonka, Benjamin Remez, Paul B. Rohrbach, Rajesh Singh, and Günther Turk, [arXiv:2005.09625, 2020](https://arxiv.org/abs/2005.09625) | [ResearchGate](https://www.researchgate.net/publication/341496170_Inference_prediction_and_optimization_of_non-pharmaceutical_interventions_using_compartment_models_the_PyRoss_library).  
  
* **Age-structured impact of social distancing on the COVID-19 epidemic in India**. Rajesh Singh and R. Adhikari,  [arXiv:2003.12055, 2020](https://arxiv.org/abs/2003.12055) | [ResearchGate](https://www.researchgate.net/publication/340209224_Age-structured_impact_of_social_distancing_on_the_COVID-19_epidemic_in_India_Updates_at_httpsgithubcomrajeshrinetpyrossa).


## License
We believe that openness and sharing improves the practice of science and increases the reach of its benefits. This code is released under the [MIT license](http://opensource.org/licenses/MIT). Our choice is guided by the excellent article on [Licensing for the scientist-programmer](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1002598).
	
