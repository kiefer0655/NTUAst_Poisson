# Final Project

## General Guidelines and Requirements

1. **Must parallelize your program by at least one of the following methods**
a. **OpenMP (minimum requirement)**
b. **MPI (get bonus point)**
c. **GPU (get extra bonus point)**
2. **Must provide convincing demonstration of the accuracy and performance of your program**
3. **Must use GitHub for collaborative development**
a. **Tutorial:** [https://gitbook.tw/](https://gitbook.tw/)
b. **Use the Fork $\rightarrow$ Pull $\rightarrow$ Push $\rightarrow$ Pull Request $\rightarrow$ Merge workflow**
c. **Do NOT just upload the final code $\rightarrow$ must keep the development history on GitHub**
4. **Students per group: 3-4**
a. **Inform the TA before May 1**
5. **Final presentation: June 5, 12**
a. **25 mins presentation + 10 mins questions**
b. **All students are encouraged to ask ANY questions**
6. **Bonus points: *Surprise Me!!!***
7. **Provide the following materials after the presentation: slides, a GitHub link (including usernames and corresponding Chinese names), and a task allocation table**

## Project-II: Multigrid Poisson Solver

1. **Implement the multigrid method to solve the 2D Poisson equation**
a. **Compare the V and W cycles**
b. **Measure the performance scaling (i.e., wall-clock time vs. number of cells)**
c. **Compare the performance with SOR**
2. **Bonus points**
a. **Implement the *full multigrid method* (FMG) and compare with MG**
3. **Reference**
a. **Enzo:**
[https://github.com/enzo-project/enzo-dev/blob/master/src/enzo/MultigridSolver.C](https://www.google.com/search?q=https://github.com/enzo-project/enzo-dev/blob/master/src/enzo/MultigridSolver.C)
b. **GAMER:**
[https://github.com/gamer-project/gamer/blob/master/src/SelfGravity/CPU_Poisson/CPU_PoissonSolver_MG.cpp](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgamer-project%2Fgamer%2Fblob%2Fmaster%2Fsrc%2FSelfGravity%2FCPU_Poisson%2FCPU_PoissonSolver_MG.cpp)
c. **Numerical Recipes, Chapter 20.6**
d. **Multigrid Techniques: 1984 Guide with Applications to Fluid Dynamics**
[https://epubs.siam.org/doi/book/10.1137/1.9781611970753](https://www.google.com/search?q=https%3A%2F%2Fepubs.siam.org%2Fdoi%2Fbook%2F10.1137%2F1.9781611970753)
