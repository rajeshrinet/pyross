## PyRoss: inference, forecasts, and optimised control for epidemiological models in Python [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rajeshrinet/pyross/master?filepath=binder) ![Build](https://github.com/rajeshrinet/pyross/workflows/build/badge.svg) ![Notebooks](https://github.com/rajeshrinet/pyross/workflows/notebooks/badge.svg)  [![PyPI](https://img.shields.io/pypi/v/pyross.svg)](https://pypi.python.org/pypi/pyross) [![Python Version](https://img.shields.io/pypi/pyversions/pyross)](https://pypi.org/project/pyross) [![Downloads](https://pepy.tech/badge/pyross)](https://pepy.tech/project/pyross)  ![stars](https://img.shields.io/github/stars/rajeshrinet/pyross) ![forks](https://img.shields.io/github/forks/rajeshrinet/pyross) ![License](https://img.shields.io/github/license/rajeshrinet/pyross) 


[About](#about) | [Installation](#installation) | [Examples](#examples) | [Publications ](#publications) | [News](#news) |  [License](#license) | [Documentation](https://pyross.readthedocs.io/en/latest/) | [Events](https://github.com/rajeshrinet/pyross/wiki/Upcoming-and-past-events) | [Wiki](https://github.com/rajeshrinet/pyross/wiki)

![Imagel](https://raw.githubusercontent.com/rajeshrinet/pyross/master/examples/banner.jpg)


## About

[PyRoss](https://github.com/rajeshrinet/pyross) is a numerical library that offers an integrated platform for **inference**, **prediction** and **non-pharmaceutical interventions** in age- and contact-structured epidemiological compartment models.

The library is designed to be model-agnostic and **allows the user to define models** in a JSON format. The most common epidemiological models, and several less common ones, come pre-defined with the library. Models can include **stages** to allow for non-exponentially distributed compartmental residence times. Currently,  [pre-defined models](https://github.com/rajeshrinet/pyross/blob/master/docs/models.pdf) include ones with multiple disease states (exposed, asymptomatic, symptomatic, etc) and may be further divided by age, and by objective medical states (hospitalized, in ICU, etc). 

The library supports models formulated **stochastically** (as chemical master equations) or **deterministically** (as systems of differential equations). A **hybrid algorithm** transits dynamically between these  depending on the magnitude of the compartmental fluctuations.

**Inference** on pre-defined or user-defined models is performed using model-adapted **Gaussian processes** on the epidemiological manifold or its tangent space. This method allows for **latent** variable inference and fast computation of the **model evidence** and the **Fisher information matrix**. These estimates are convolved with the instrinsic stochasticty of the dynamics to provide **Bayesian forecasts** of the progress of the epidemic.

**Non-pharmaceutical interventions** are implemented as modifications of the **contact structures** of the model. **Optimised control** of these structures, given **cost functions**, is possible. This feature is being actively developed to be better integrated with the library.

[PyRossGeo](https://github.com/lukastk/PyRossGeo) is a companion library that supports **spatially resolved compartment models** with explicit **commuting networks**. [PyRossTSI](https://github.com/rajeshrinet/pyrossTSI) is another companion library for **time since infection models**. 

The libraries are named after [Sir Ronald Ross](https://en.wikipedia.org/wiki/Ronald_Ross), doctor, mathematician and poet. In 1898 he made "the great discovery" in his laboratory in Calcutta "that malaria is conveyed by the bite of a mosquito".  He won the Nobel Prize in 1902 and laid the foundations of the mathematical modelling of infectious diseases.


The authors are part of [The Rapid Assistance in Modelling the Pandemic (RAMP)](https://royalsociety.org/news/2020/03/urgent-call-epidemic-modelling/) taskforce at the **University of Cambridge**. In alphabetical order, we are:
[Ronojoy Adhikari](https://github.com/ronojoy),
[Austen Bolitho](https://github.com/TakodaS),
Fernando Caballero, Michael Cates,
[Jakub Dolezal](https://github.com/JakubJDolezal),
[Tim Ekeh](https://github.com/tekeh),
Jules Guioth, Robert Jack,
[Julian Kappler](https://github.com/juliankappler),
[Lukas Kikuchi](https://github.com/lukastk),
[Irene Li](https://github.com/Irene-Li),
Joseph Peterson,
[Patrick Pietzonka](https://github.com/ppietzonka),
[Benjamin Remez](https://github.com/BenjaminRemez),
[Paul Rohrbach](https://github.com/prohrbach),
[Rajesh Singh](https://github.com/rajeshrinet), 
and [Günther Turk](https://github.com/phi6GTurk).

Please read the  [PyRoss paper](https://arxiv.org/abs/2005.09625) and [PyRoss Wiki](https://github.com/rajeshrinet/pyross/wiki/) before you use PyRoss for your research. [Open an issue](https://github.com/rajeshrinet/pyross/issues), in preference to emailing us with queries. Join our [Slack channel](https://join.slack.com/t/pyross/shared_invite/zt-e8th6kcz-S4b_oJIZWPsGLruSPl3Zuw)  for discussion. Please follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/0/code_of_conduct/) in all PyRoss fora. Thank you!

## Installation
You can take PyRoss for a spin **without installation**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rajeshrinet/pyross/master?filepath=binder). Please be patient while [Binder](https://mybinder.org/v2/gh/rajeshrinet/pyross/master?filepath=binder) loads.


### Via [Anaconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html)

Install PyRoss and its dependencies in a `pyross` [environment](https://github.com/rajeshrinet/pyross/blob/master/environment.yml):

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

### From a checkout of this repository

Install PyRoss and required dependencies using

```bash
>> git clone https://github.com/rajeshrinet/pyross.git
>> cd pyross
>> pip install -r requirements.txt
>> python setup.py install
```


#### Testing
Short test of initialisation and running
```
make test
```

Long test of all example notebooks. Optionally can specify path and recursion
to test a certain subset of notebooks.
```
make nbtest -e path=examples/deterministic/
```

**Age structure data:** [Population Pyramid](https://www.populationpyramid.net/) website

**Contact structure data:** *Projecting social contact matrices in 152 countries using contact surveys and demographic data*, Kiesha Prem, Alex R. Cook, Mark Jit, [PLOS Computational Biology, (2017)]( https://doi.org/10.1371/journal.pcbi.1005697)




## Examples

PyRoss has model-agnostic, formulation-agnostic intuitive interface. Once a model is instantiated, stochastic, deterministic and hybrid simulations can be performed through the same interface. The example below shows how to set up a deterministic SIR simulation. [PyRoss Documentation](https://pyross.readthedocs.io/en/latest/) contains links for more examples.

```Python
# Ex1: M=1, SIR
import numpy as np
import pyross

M     = 1                  # the SIR model has no age structure
Ni    = 1000*np.ones(M)    # so there is only one age group
N     = np.sum(Ni)         # and the total population is the size of this age group

beta  = 0.2                # infection rate
gIa   = 0.1                # recovery rate of asymptomatic infectives
gIs   = 0.1                # recovery rate of symptomatic infectives
alpha = 0                  # fraction of asymptomatic infectives
fsa   = 1                  # the self-isolation parameter

Ia0   = np.array([0])      # the SIR model has only one kind of infective
Is0   = np.array([1])      # we take these to be symptomatic
R0    = np.array([0])      # and assume there are no recovered individuals initially
S0    = N-(Ia0+Is0+R0)     # initial susceptibles are obtained from S + Ia + Is + R = N

# there is no contact structure
def contactMatrix(t):   
    return np.identity(M)

# instantiate model
parameters = {'alpha':alpha, 'beta':beta, 'gIa':gIa, 'gIs':gIs, 'fsa':fsa}
model      = pyross.deterministic.SIR(parameters, M, Ni)

# simulate model
Tf, Nt = 160,  160           # duration of simulation and data points
data = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nt)

# time series of S, Ia, Is, R
S  = model.S(data)
Ia = model.Ia(data)
Is = model.Is(data)
R  = model.R(data)
```

## Publications
* **Inference, prediction and optimization of non-pharmaceutical interventions using compartment models: the PyRoss library**, R. Adhikari, Austen Bolitho, Fernando Caballero, Michael E. Cates,
  Jakub Dolezal, Timothy Ekeh, Jules Guioth, Robert L. Jack, Julian Kappler,
  Lukas Kikuchi, Hideki Kobayashi, Yuting I. Li, Joseph D. Peterson, Patrick
  Pietzonka, Benjamin Remez, Paul B. Rohrbach, Rajesh Singh, and Günther Turk, [arXiv:2005.09625](https://arxiv.org/abs/2005.09625), [Research Gate](https://www.researchgate.net/publication/341496170_Inference_prediction_and_optimization_of_non-pharmaceutical_interventions_using_compartment_models_the_PyRoss_library).  <br/> *(Please cite this paper if you use PyRoss in your research)*.

* **Age-structured impact of social distancing on the COVID-19 epidemic in India**, Rajesh Singh and R. Adhikari,  [arXiv:2003.12055](https://arxiv.org/abs/2003.12055), [Research Gate](https://www.researchgate.net/publication/340209224_Age-structured_impact_of_social_distancing_on_the_COVID-19_epidemic_in_India_Updates_at_httpsgithubcomrajeshrinetpyrossa).


## News
* A partial list of news articles on PyRoss-based research:  [Zee News](https://zeenews.india.com/india/3-week-coronavirus-covid-19-lockdown-not-enough-sustained-periods-of-shutdown-with-periodic-relaxation-will-work-research-2272313.html)
| [Varsity](https://www.varsity.co.uk/news/19284)
|  [The Weekend Leader](http://www.theweekendleader.com/Headlines/54963/49-day-lockdown-necessary-to-stop-covid-19-resurgence-in-india-study.html) | [The Weather Channel](https://weather.com/en-IN/india/coronavirus/news/2020-03-29-india-needs-49-day-lockdown-prevent-resurgence-covid-19-study) | [The Tribune](https://www.tribuneindia.com/news/49-day-lockdown-necessary-to-stop-coronavirus-resurgence-in-india-study-62790) | [The Times of India](https://timesofindia.indiatimes.com/readersblog/viewofac/a-hard-headed-look-can-india-fight-covid-19-only-through-lock-down-for-how-long-11163/) | [The Telegraph](https://www.telegraphindia.com/india/coronavirus-outbreak-a-case-for-evidence-based-lockdowns-after-april-14/cid/1760155) | [The Northlines](http://www.thenorthlines.com/21-day-covid-19-lockdown-not-enough-sustained-shutdown-with-periodic-relaxation-will-work-research/) | [The New Indian Express](https://www.newindianexpress.com/nation/2020/mar/28/21-day-lockdown-not-enough-to-contain-coronavirus-outbreak-study-2122803.html) | [The Federal](https://thefederal.com/covid-19/data-impact-to-decide-extension-of-covid-19-lockdown-experts/) | [Swarajya](https://swarajyamag.com/insta/49-day-lockdown-required-to-prevent-return-of-covid-19-in-india-cambridge-university-study-suggests) | [Snoop Tales](https://snooptales.com/2020/03/30/coronavirus-lockdown-cambridge-model-predicts-what-india-needs-to-contain-covid-19-spread/) | [Scroll.in](https://scroll.in/article/958277/the-political-fix-will-covid-19-crisis-slow-down-modis-centralising-tendencies) | [Rediff.com](https://www.rediff.com/news/interview/coronavirus-in-india-india-may-need-a-49-day-lockdown/20200409.htm) | [The Quint](https://www.thequint.com/news/india/scientists-mathematical-modeling-warn-sustained-lockdowns-to-beat-covid-19) | [People's Reporter](https://www.peoplesreporter.in/science-technology/5801-minimum-49-days-lockdown-necessary-to-prevent-covid-19-says-cambridge-researcher.html) | [News Bundle](https://newsbunddle.com/%E0%A4%95%E0%A5%8B%E0%A4%B0%E0%A5%8B%E0%A4%A8%E0%A4%BE%E0%A4%B5%E0%A4%BE%E0%A4%AF%E0%A4%B0%E0%A4%B8-%E0%A4%B2%E0%A5%89%E0%A4%95%E0%A4%A1%E0%A4%BE%E0%A4%89%E0%A4%A8-%E0%A4%95%E0%A5%88%E0%A4%AE/) | [Newsgram](https://www.newsgram.com/49-day-national-lockdown-prevent-coronavirus-resurgence-india) | [Nature News](https://www.nature.com/articles/d41586-020-01058-5) | [Mathrubhumi](https://www.mathrubhumi.com/news/india/49-day-lockdown-necessary-to-stop-coronavirus-resurgence-in-india-study-1.4652600)  | [liveMint](https://www.livemint.com/news/india/49-day-lockdown-necessary-to-stop-coronavirus-resurgence-in-india-study-11585473979844.html) |  [India New England News](https://indianewengland.com/2020/03/49-day-lockdown-necessary-to-stop-covid-19-resurgence-in-india-study/) | [India Today](https://www.indiatoday.in/india/story/coronavirus-lockdown-india-cambridge-mathematical-model-extension-1661321-2020-03-30) | [Indian Express: Bangla](https://bengali.indianexpress.com/opinion/21-days-lock-down-not-enough-exponential-curve-research-206782/) | [Indian Blooms](https://indiablooms.com/health-details/H/5782/india-needs-49-day-lockdown-to-combat-covid-19-cambridge-study.html) | [Dainik Bhaskar](https://f87kg.app.goo.gl/epaper) | [CNN](https://edition.cnn.com/2020/04/28/india/india-coronavirus-outbreak-explained-intl-hnk/index.html) | [CCN](https://www.ccn.com/indias-total-coronavirus-lockdown-isnt-enough-and-its-faltering/) | [Business Standard](https://www.business-standard.com/article/current-affairs/49-day-lockdown-necessary-to-stop-coronavirus-resurgence-in-india-study-120032900487_1.html) | [Asiaville](https://www.asiavillenews.com/article/experts-on-whether-india-will-flatten-covid-19-curve-effectively-37658)

* Python Trending has tweeted about PyRoss <blockquote class="twitter-tweet"><p lang="en" dir="ltr">PyRoss - Mathematical modelling of infectious disease in Python. <a href="https://t.co/VBOmT5kaVT">https://t.co/VBOmT5kaVT</a> <a href="https://t.co/FRvEqWMlPL">https://t.co/FRvEqWMlPL</a></p>&mdash; Python Trending (@pythontrending) <a href="https://twitter.com/pythontrending/status/1244918005731033088?ref_src=twsrc%5Etfw">March 31, 2020</a></blockquote>  

* Python Weekly has tweeted about PyRoss <blockquote class="twitter-tweet"><p lang="en" dir="ltr">PyRoss - Mathematical modelling of infectious disease in Python. <a href="https://t.co/MyTWTM1ItP">https://t.co/MyTWTM1ItP</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/Coronavirus?src=hash&amp;ref_src=twsrc%5Etfw">#Coronavirus</a> <a href="https://twitter.com/hashtag/Covid19?src=hash&amp;ref_src=twsrc%5Etfw">#Covid19</a> <a href="https://t.co/gzPNpWf7mK">pic.twitter.com/gzPNpWf7mK</a></p>&mdash; Python Weekly (@PythonWeekly) <a href="https://twitter.com/PythonWeekly/status/1248324915788492807?ref_src=twsrc%5Etfw">April 9, 2020</a></blockquote>



## License
We believe that openness and sharing improves the practice of science and increases the reach of its benefits. This code is released under the [MIT license](http://opensource.org/licenses/MIT). Our choice is guided by the excellent article on [Licensing for the scientist-programmer](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1002598).

