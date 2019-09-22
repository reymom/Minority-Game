# Minority-Game
The Minority Game is a widely used model to reproduce market features.

In the basic Minority Game, an odd number N of agents compete in successive rounds where they can choose between two options, say −1 or 1, wanting to be in the minority side each round. The motivation is that, for example, if you want to sell an asset in the next time step, you want that the most part of people were willing to buy assets, so to increase its price when finally all perform an action. So, if you succeed you get a reward, otherwise you get punished. At the beginning of the game each agent is assigned a number of random strategies, which will govern the agent’s behaviour. To choose what strategy to use each round, each is assigned a score based on how well it has performed (virtually) so far,the one with leading score is used at a time step.

Project still in development.

## Material
 - **mygame_functions.py** : Basic program where all necessary functions for the first part are defined. Two main groups of functions are defined, one for the basic analysis of the MG, with different functions in order to extract different kind of results. The second group includes modifications in order to study basic mechanisms in market dynamics, such as having different kinds of agents (producers and speculators) and the inclusion of privileged agents.
 - **MinorityGameComplex.py** : Basic program for the second part. It basically integrates a second dynamic in the dynamics of the minority game of the first part. The dynamics is imitation through a social layer. Different kinds of integration are performed, it allows to adapt the timescale coupling between the two dynamics.

 - **.ipynb** and **.py** files to gather statistics and obtain results and figures using the dynamics defined in the functions of the previous presented programs.
 - **figures**, important results.



## Tasks

- [x] Basic naming game; volatility, phase transition, polarization, frozen agents and predictability.
- [x] Basic naming game with change of variables; speeded up.
- [x] Check the good convergence of the simulation in order to take good statistics.
- [x] Analytical calculations to check numerical ones.
- [x] Modeling market mechanism with minority game: Speculators and Producers, Noise traders, Quitters, multiple strategies, spies.
- [x] Creative part: Imitation of successful strategies in an interwined dynamics scope, through a social layer defined as a complex graph with different kinds of topology (random, Small-World and Scale-free graph)
- [ ] Final manuscript explaining in great detail all the theory and results.

