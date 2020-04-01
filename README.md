# pysimulitis
A python translation of COVID-19 simulation code in MATLAB by [Joshua Gafford](https://www.mathworks.com/matlabcentral/fileexchange/74610-simulitis-a-coronavirus-simulation) 

This is based on the [Washington Post](https://www.washingtonpost.com/graphics/2020/world/corona-simulator/) simulation that illustrates the impact social distancing can have on spread of COVID-19. As Joshua Gafford indicates, he incorporated a probability of mortality that allows simulation of a more deadly disease like Ebola but also there are dials to control infection probability, and other variables. This is illustrative and I can't guarantee 100% that this fully duplicates the MATLAB version. Also, it should go without saying that quantitative conclusions should not be drawn from this, but definitely it highlights the power of pyhsical distancing in "flattening the curve".

# Prerequisites
This depends on the following modules obtainable from `pip`:
```
matplotlib 
numpy 
imageio 
imageio-ffmpeg
```

# or run in binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mnfienen/pysimulitis/master?urlpath=lab)

# Example output
<img src="model0.0.gif " width="200"><img src="model0.5.gif " width="200"><img src="model0.75.gif " width="200">
