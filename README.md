# bike-guessr
This repository contains the code for the master thesis "Link prediction in urban network" at Wrocław University of Science and Technology. It includes code for data extraction from OpenStreetMap (OSM) using the OSMnx library, as well as code for the various link prediction methods discussed in the thesis. The goal of this project is to improve the prediction of links in urban networks using machine learning techniques.

# TODO:
- [x] Add to loss function a part for graph connectedness. Include information that f1 will suffer from this.
- [x] Visualize how many of each type of link is downloaded from OSM.
- [x] Calculate the statistics from Thesis - AD, CC, AC1, PLE, EDE, AC2, Largest component, number of components
- [ ] Divide the graphs into specific level of bike network need. those are b. hostile, b. ignorant, b. emerging, b. friendly, b. dominant. That means preferably one of each for test, two each for validation and the rest for training - preferably 70 cities in total. 
- [ ] Improve the documentation of the code.
- [ ] Add instruction to run the code in the README.md file.

\begin{itemize}
    \item MGCN
    \item MGCN + encoding
    \item MGCN + CEC loss
    \item MGCN + encoding + CEC loss
    \item GCN
    \item GCN + encoding
    \item GCN + CEC loss
    \item GCN + encoding + CEC loss
    \item Trivial + encoding
    \item Trivial + encoding + CEC loss
\end{itemize}