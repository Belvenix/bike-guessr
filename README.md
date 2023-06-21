# bike-guessr
This repository contains the code for the master thesis "Link prediction in urban network" at Wrocław University of Science and Technology. It includes code for data extraction from OpenStreetMap (OSM) using the OSMnx library, as well as code for the various link prediction methods discussed in the thesis. The goal of this project is to improve the prediction of links in urban networks using machine learning techniques.

# TODO:
- [ ] Add to loss function a part for graph connectedness. Include information that f1 will suffer from this.
- [ ] Visualize how many of each type of link is downloaded from OSM.
- [ ] Calculate the statistics from Thesis - AD, CC, AC1, PLE, EDE, AC2, Largest component, number of components
- [ ] Divide the graphs into specific level of bike network need. those are b. hostile, b. ignorant, b. emerging, b. friendly, b. dominant. That means preferably one of each for test, two each for validation and the rest for training - preferably 70 cities in total. 
- [ ] Describe that the graphs are linegraphs

## 3rd party code used:

### Link prediction methods:
Resource Allocation (RA) - adapted from: [Github - LPM](https://github.com/whxhx/Link-Prediction-Methods)
Local Random Walk (LRW) - adapted from: [Github - LPM](https://github.com/whxhx/Link-Prediction-Methods)
Stochastic Block Model (SBM) - adapted from: [Github - LPM](https://github.com/whxhx/Link-Prediction-Methods)
Naive Bayes (NB) - ???
Matrix Factorization (MF) - adapted from: [Github - LPM](https://github.com/whxhx/Link-Prediction-Methods)
Struct2vec - adapted from: [Github - Struc2vec](https://github.com/leoribeiro/struc2vec)
SEAL - adapted from: [Github - SEAL](https://github.com/muhanzhang/SEAL)
VERSE - adapted from: [Github - VERSE](https://github.com/xgfs/verse)

### Urban application methods:
GrowBike - adapted from: [Github - GrowBike](https://github.com/mszell/bikenwgrowth)
LinkBike - adapted from: [Github - LinkBike](https://github.com/nateraluis/bicycle-network-growth)
FixBike - adapted from: [Github - FixBike](https://github.com/anastassiavybornova/bikenwgaps)
EgalitarianBike - adapted from: [Github - EgalitarianBike](https://github.com/Hussein-Mahfouz/cycle-networks)
