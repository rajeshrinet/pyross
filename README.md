## PyRoss: Infectious disease models in Python [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rajeshrinet/pyross/master?filepath=examples)

[About](#about) | [Contact](#contact) | [Publications ](#publications) | [News](#news) | [Installation](#installation) | [Examples](#examples) | [License](#license) | [Wiki](https://github.com/rajeshrinet/pyross/wiki)

![Imagel](examples/banner.png)

## About

[PyRoss](https://gitlab.com/rajeshrinet/pyross) is a numerical library for mathematical modelling of infectious disease in Python. The library supports structured compartment models formulated deterministically (as systems of differential equations) or stochastically (as chemical master equations). The library can move smoothly from stochastic to deterministic formulations as the compartmental fluctuations become small. 

Currently implemented [models](https://github.com/rajeshrinet/pyross/blob/master/docs/models.pdf) are  **SIR**, **SIRS**, **SEIR**, **SEAIR**, **SEAIRQ**, **SIkR**, and **SEkIkR**.

PyRoss was developed to model the outbreak of the novel coronavirus COVID-19 and to assess the age-structured impact of social distancing measures in India. 

The library is named after [Sir Ronald Ross](https://en.wikipedia.org/wiki/Ronald_Ross), doctor, mathematician and poet. In 1898 he made "the great discovery" in his laboratory in Calcutta "that malaria is conveyed by the bite of a mosquito".  He won the Nobel Prize in 1902 and laid the foundations of the mathematical modelling of infectious diseases. 

Please read the [PyRoss Wiki](https://github.com/rajeshrinet/pyross/wiki/) before you use PyRoss for your research. 


## Contact

The authors are part of [The Rapid Assistance in Modelling the Pandemic (RAMP)](https://royalsociety.org/news/2020/03/urgent-call-epidemic-modelling/) taskforce at the University of Cambridge. 

Please first [read the wiki](https://github.com/rajeshrinet/pyross/wiki) and  then only [open an issue](https://github.com/rajeshrinet/pyross/issues), in preference to emailing us with queries. Issues can be shared with others with similar queries and you help the user community by communicating through issues. Thank you!

When all else fails, we can be contacted at: ra413@cam.ac.uk (R. Adhikari), jk762@cam.ac.uk (Julian Kappler) and rs2004@cam.ac.uk (Rajesh Singh).

## Publications

* [Age-structured impact of social distancing on the COVID-19 epidemic in India](https://github.com/rajeshrinet/pyross/blob/master/draft/covid19.pdf) ([arXiv:2003.12055](https://arxiv.org/abs/2003.12055), [Research Gate](https://www.researchgate.net/publication/340209224_Age-structured_impact_of_social_distancing_on_the_COVID-19_epidemic_in_India_Updates_at_httpsgithubcomrajeshrinetpyrossa)). Rajesh Singh and R. Adhikari (2020). 

This paper is amongst the  **top five most read COVID-19 papers** on [Research Gate 
](https://www.researchgate.net/community/COVID-19). The authors have received (literally) hundreds of emails with inquiries and apologise that they can no longer respond individually to these. Please read the [Frequently Asked Questions](https://github.com/rajeshrinet/pyross/wiki/FAQ-on-COVID-19-India-paper) in the first instance. The authors promise to include questions not covered in the FAQ as soon as possible. 

The figures below show how the lockdown forecasts from the paper are altered by a change in the epidemiological model. The SIR model provides a lower bound on the rapidity with which the epidemic decreases. All other models exceed this bound, as shown for an SEIR model. Both models are fitted to the 
number of infectives using case data for India till 25-03-2020. 

![SIR and SEIR](draft/sir-seir.gif)


## News


* A partial list of news articles on PyRoss-based research:  [Zee News](https://zeenews.india.com/india/3-week-coronavirus-covid-19-lockdown-not-enough-sustained-periods-of-shutdown-with-periodic-relaxation-will-work-research-2272313.html) |  [The Weekend Leader](http://www.theweekendleader.com/Headlines/54963/49-day-lockdown-necessary-to-stop-covid-19-resurgence-in-india-study.html) | [The Weather Channel](https://weather.com/en-IN/india/coronavirus/news/2020-03-29-india-needs-49-day-lockdown-prevent-resurgence-covid-19-study) | [The Tribune](https://www.tribuneindia.com/news/49-day-lockdown-necessary-to-stop-coronavirus-resurgence-in-india-study-62790) | [The Times of India](https://timesofindia.indiatimes.com/readersblog/viewofac/a-hard-headed-look-can-india-fight-covid-19-only-through-lock-down-for-how-long-11163/) | [The Telegraph](https://www.telegraphindia.com/india/coronavirus-outbreak-a-case-for-evidence-based-lockdowns-after-april-14/cid/1760155) | [The Northlines](http://www.thenorthlines.com/21-day-covid-19-lockdown-not-enough-sustained-shutdown-with-periodic-relaxation-will-work-research/) | [The New Indian Express](https://www.newindianexpress.com/nation/2020/mar/28/21-day-lockdown-not-enough-to-contain-coronavirus-outbreak-study-2122803.html) |  [Swarajya](https://swarajyamag.com/insta/49-day-lockdown-required-to-prevent-return-of-covid-19-in-india-cambridge-university-study-suggests) | [Snoop Tales](https://snooptales.com/2020/03/30/coronavirus-lockdown-cambridge-model-predicts-what-india-needs-to-contain-covid-19-spread/) | [Scroll.in](https://scroll.in/article/958277/the-political-fix-will-covid-19-crisis-slow-down-modis-centralising-tendencies) | [Rediff.com](https://www.rediff.com/news/interview/coronavirus-in-india-india-may-need-a-49-day-lockdown/20200409.htm) | [People's Reporter](https://www.peoplesreporter.in/science-technology/5801-minimum-49-days-lockdown-necessary-to-prevent-covid-19-says-cambridge-researcher.html) | [News Bundle](https://newsbunddle.com/%E0%A4%95%E0%A5%8B%E0%A4%B0%E0%A5%8B%E0%A4%A8%E0%A4%BE%E0%A4%B5%E0%A4%BE%E0%A4%AF%E0%A4%B0%E0%A4%B8-%E0%A4%B2%E0%A5%89%E0%A4%95%E0%A4%A1%E0%A4%BE%E0%A4%89%E0%A4%A8-%E0%A4%95%E0%A5%88%E0%A4%AE/) | [Newsgram](https://www.newsgram.com/49-day-national-lockdown-prevent-coronavirus-resurgence-india) | [Nature News](https://www.nature.com/articles/d41586-020-01058-5) | [Mathrubhumi](https://www.mathrubhumi.com/news/india/49-day-lockdown-necessary-to-stop-coronavirus-resurgence-in-india-study-1.4652600)  | [liveMint](https://www.livemint.com/news/india/49-day-lockdown-necessary-to-stop-coronavirus-resurgence-in-india-study-11585473979844.html) |  [India New England News](https://indianewengland.com/2020/03/49-day-lockdown-necessary-to-stop-covid-19-resurgence-in-india-study/) | [India Today](https://www.indiatoday.in/india/story/coronavirus-lockdown-india-cambridge-mathematical-model-extension-1661321-2020-03-30) | [Indian Express: Bangla](https://bengali.indianexpress.com/opinion/21-days-lock-down-not-enough-exponential-curve-research-206782/) | [Indian Blooms](https://indiablooms.com/health-details/H/5782/india-needs-49-day-lockdown-to-combat-covid-19-cambridge-study.html) | [Dainik Bhaskar](https://f87kg.app.goo.gl/epaper) | [CCN](https://www.ccn.com/indias-total-coronavirus-lockdown-isnt-enough-and-its-faltering/) | [Business Standard](https://www.business-standard.com/article/current-affairs/49-day-lockdown-necessary-to-stop-coronavirus-resurgence-in-india-study-120032900487_1.html) | [Asiaville](https://www.asiavillenews.com/article/experts-on-whether-india-will-flatten-covid-19-curve-effectively-37658)

* Python Trending has tweeted about PyRoss <blockquote class="twitter-tweet"><p lang="en" dir="ltr">PyRoss - Mathematical modelling of infectious disease in Python. <a href="https://t.co/VBOmT5kaVT">https://t.co/VBOmT5kaVT</a> <a href="https://t.co/FRvEqWMlPL">https://t.co/FRvEqWMlPL</a></p>&mdash; Python Trending (@pythontrending) <a href="https://twitter.com/pythontrending/status/1244918005731033088?ref_src=twsrc%5Etfw">March 31, 2020</a></blockquote>  


* Python Weekly has tweeted about PyRoss <blockquote class="twitter-tweet"><p lang="en" dir="ltr">PyRoss - Mathematical modelling of infectious disease in Python. <a href="https://t.co/MyTWTM1ItP">https://t.co/MyTWTM1ItP</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/Coronavirus?src=hash&amp;ref_src=twsrc%5Etfw">#Coronavirus</a> <a href="https://twitter.com/hashtag/Covid19?src=hash&amp;ref_src=twsrc%5Etfw">#Covid19</a> <a href="https://t.co/gzPNpWf7mK">pic.twitter.com/gzPNpWf7mK</a></p>&mdash; Python Weekly (@PythonWeekly) <a href="https://twitter.com/PythonWeekly/status/1248324915788492807?ref_src=twsrc%5Etfw">April 9, 2020</a></blockquote>



## Installation
Clone (or download) the repository and use a terminal to install using 

```bash
>> git clone https://github.com/rajeshrinet/pyross.git
>> cd pyross
>> python setup.py install
```

PyRoss requires the following software 

- Python 2.6+ or Python 3.4+
- [Cython 0.25.x+](http://docs.cython.org/en/latest/index.html) |  [Matplotlib 2.0.x+](https://matplotlib.org) | [NumPy 1.x+](http://www.numpy.org) |  [Pandas](https://pandas.pydata.org/) | [SciPy 1.1.x+](https://www.scipy.org/) | [xlrd](https://xlrd.readthedocs.io/en/latest/)

## Data sources

**Case data:** The data for COVID-19 cases is obtained from the [Worldometer website](https://www.worldometers.info/coronavirus).

**Age structure:** [Population Pyramid](https://www.populationpyramid.net/) website. 

**Contact structure:** *Projecting social contact matrices in 152 countries using contact surveys and demographic data*, Kiesha Prem, Alex R. Cook, Mark Jit, PLOS Computational Biology, (2017) [DOI]( https://doi.org/10.1371/journal.pcbi.1005697), [Supporting Information Text](https://doi.org/10.1371/journal.pcbi.1005697.s001)  and [Supporting Information Data](https://doi.org/10.1371/journal.pcbi.1005697.s001).


## Examples

PyRoss has a formulation-agnostic and  intuitive interface. Once a model is formulated, stochastic, deterministic and hybrid simulations can performed through the same interface. The example below shows how to set up a deterministic SIR simulation. See the [examples folder](https://github.com/rajeshrinet/pyross/tree/master/examples) for more Jupyter notebook examples.

```Python
#Ex1: M=1, SIR
import numpy as np
import pyross
M = 1                  # the SIR model has no age structure
Ni = 1000*np.ones(M)   # so there is only one age group 
N = np.sum(Ni)         # and the total population is the size of this age group

beta  = 0.2            # infection rate 
gamma = 0.1            # recovery rate 
alpha = 0              # fraction of asymptomatic infectives 
fsa   = 1              # the self-isolation parameter 


Ia0 = np.array([0])     # the SIR model has only one kind of infective 
Is0 = np.array([1])     # we take these to be symptomatic 
R0  = np.array([0])     # and assume there are no recovered individuals initially 
S0  = N-(Ia0+Is0+R0)    # so that the initial susceptibles are obtained from S + Ia + Is + R = N

# there is no contact structure
def contactMatrix(t):   
    return np.identity(M) 


# instantiate model
parameters = {'alpha':alpha, 'beta':beta, 'gamma':gamma,'fsa':fsa}

model = pyross.deterministic.SIR(parameters, M, Ni)

# duration of simulation and data file
Tf = 160;  Nt=160; 

# simulate model
data = model.simulate(S0, Ia0, Is0, contactMatrix, Tf, Nt)
```


## License
We believe that openness and sharing improves the practice of science and increases the reach of its benefits. This code is released under the [MIT license](http://opensource.org/licenses/MIT). Our choice is guided by the excellent article on [Licensing for the scientist-programmer](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1002598). 
