# dynamic_aca_tsp
Visualize the dynamic ant colony algorithm(ACA).

Imagine a situation like this. A delivery person needs to deliver items to targeted customers one by one every day. In the process, a customer changes his mind and cancels the order. At the same time, a new customer shows up. Then the problem changes from a traditional TSP problem to a dynamic TSP problem. We need to explore whether the original optimal path is still valid when the goal changes. How to reach convergence again after the change.

This program is designed using the ACO algorithm. I created a region of 50 points. The coordinates of each point are randomly generated. When the number of iterations reaches 50 and 150, the pheromone is reset(dynamic_aca_tsp.py) after randomly replacing the position and number of some of these points. The total number of iterations is 300. And another version dynamic_aca_tsp_without_reset does not reset the pheromone.

⬇️ This is the total distance to iteration.
![ACA_TSP_10](https://github.com/GDBNPIO/dynamic_aca_tsp/assets/25226269/c690f478-77a2-4fc4-ae50-4aab52ec8f08)

⬇️ This is the route changing with iteration.
![ACA_TSP_15](https://github.com/GDBNPIO/dynamic_aca_tsp/assets/25226269/c2b1b014-bd94-4f5d-98d1-9e4030dd5a7f)

I save the result charts and gifs when  5 points change.

# Reference
[1] Guo, Fei. scikit-opt. Available at: https://github.com/guofei9987/scikit-opt. Accessed on 2024.2.6.

Thanks to Guo, Fei. your work is really helpful!
