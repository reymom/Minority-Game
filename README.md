# Minority-Game
The Minority Game is a widely used model to reproduce market features.

In the basic Minority Game, an odd number N of agents compete in successive rounds where they can choose between two options, say −1 or 1, wanting to be in the minority side each round. The motivation is that, for example, if you want to sell an asset in the next time step, you want that the most part of people were willing to buy assets, so to increase its price when finally all perform an action. So, if you succeed you get a reward, otherwise you get punished. At the beginning of the game each agent is assigned a number of random strategies, which will govern the agent’s behaviour. To choose what strategy to use each round, each is assigned a score based on how well it has performed (virtually) so far,the one with leading score is used at a time step.

Project still in development.

## Material
 - **mygame_functions.py** : Basic program where all necessary functions are defined. The last function performs the game in the specific case of having 2 strategies, with changes of variable the computational time needed to do simulations are more than 3 times less than in the other functions.

 - **.ipynb** and **.py** files to gather statistics and obtain results and figures
 - **some figures**



## Tasks

- [x] Basic naming game; volatility, phase transition, polarization, frozen agents and predictability.
- [x] Basic naming game with change of variables; speeded up.
- [x] Check the good convergence of the simulation in order to take good statistics.
- [ ] Analytical calculations to check numerical ones.
- [ ] Modeling market mechanism with minority game: Speculators and Producers, Noise traders, Quitters, multiple strategies, spies.
- [ ] Creative part: Network topology exploration, spreading of ideas and attitudes.

