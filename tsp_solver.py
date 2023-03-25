from math import sqrt
from typing import List, Tuple
import math
import numpy as np
import heapq

#Import necessary libraries

def MST_heuristic(cities : List[Tuple[float, float]]) -> float:
    '''
    Implement the MST algorithm and calculate the optimal cost for connecting cities

    @param cities: a vector of tuples containing the (x,y) co-ordinates of the cities
    @return: Optimal cost for connecting all the cities with MST
    '''
    INF = float('inf')
    N = len(cities)
    selected_node = [0]*N
    no_edge = 0
    selected_node[0] = True
    adjacency_matrix = np.zeros((N,N))
    cost=0
    
    for i in range(N):
        for j in range(0,i):
            adjacency_matrix[i][j] = math.dist(cities[i],cities[j])
            adjacency_matrix[j][i] = math.dist(cities[i],cities[j])

    while (no_edge < N - 1):
        minimum = INF
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and adjacency_matrix[m][n]):  
                        if minimum > adjacency_matrix[m][n]:
                            minimum = adjacency_matrix[m][n]
                            a = m
                            b = n
        cost+=adjacency_matrix[a][b]
        selected_node[b] = True
        no_edge += 1
    return cost


def tsp_search(cities : List[Tuple[float, float]]) -> Tuple[float, int, float]:
    '''
    Implement the A* start algorithm and calculate the optimal cost for reaching goal

    @param cities: a vector of tuples containing the (x,y) co-ordinates of the cities
	@return: A structure of type AstarReturn with three elements where
		first element (float type): optimal travel distance for the TSP problem
		second element (int type): number of nodes expanded
		third element (float type): the runtime for the A* search (in seconds)
    '''

    def heuristic(path, remaining_cities):
        """Calculate the lower bound of the cost to complete the path by adding an MST."""
        if not remaining_cities:
            return 0
        edges = minimum_spanning_tree(remaining_cities)
        return sum(edge[0] for edge in edges) + min(
            math.dist(path[-1], city) for city in remaining_cities
        )
    remaining_cities = set(cities)
    remaining_cities.remove(cities[0])
    paths = [(0, [cities[0]])]
    while True:
        if not paths:
            print(cost)
            return (cost,0,0)
        cost, path = heapq.heappop(paths)
        if len(path) == len(cities):
            return path + [cities[0]]
        for city in remaining_cities:
            new_path = path + [city]
            new_cost = cost + math.dist(path[-1], city)
            heapq.heappush(paths, (new_cost + heuristic(new_path, remaining_cities - {city}), new_path))
        remaining_cities.remove(city)


def minimum_spanning_tree(cities):
    """Calculate the minimum spanning tree of a set of cities using Prim's algorithm."""
    start_city = cities[0]
    visited_cities = [start_city]
    edges = []
    for i in range(len(cities) - 1):
        new_edges = [
            (math.dist(city, visited_city), city, visited_city)
            for city in cities
            for visited_city in visited_cities
            if city not in visited_cities
        ]
        edges.append(min(new_edges))
        visited_cities.append(edges[-1][1])
    print(edges)
    return edges

def solve_problem():
    # Complete function to produce the necessary data for future plotting
    pass

def check_mst_heuristic_for_case(cities : List[Tuple[float, float]], true_cost : float):
    cost = MST_heuristic(cities)
    print( "Current case: Returned cost for MST "+str(cost)+"; true cost "+str(true_cost))
    if abs(true_cost - cost) > 1e-9:
        raise Exception("Error: True cost doesn't match cost returned by MST implementation")
	
def check_mst_heuristic():
    cities1 = [(1,1), (2,2)]
    cities2 = [(1,1), (2,2), (3,3)]
    cities3 = [(1,1), (0,1), (0, 0), (1, 0)]
    cities4 = [(1,1), (2,1), (0, 0), (3, 0)]
    sqrt2 = sqrt(2.0)
    check_mst_heuristic_for_case(cities1, sqrt2)
    check_mst_heuristic_for_case(cities2, 2*sqrt2)
    check_mst_heuristic_for_case(cities3, 3)
    check_mst_heuristic_for_case(cities4, 1 + 2*sqrt2)


def check_tsp_for_case(cities : List[Tuple[float, float]], true_cost : float):
    cost = tsp_search(cities)[0]
    print("Current case: Returned cost for TSP "+str(cost)+"; true cost "+str(true_cost))
    if abs(true_cost - cost) > 1e-9 :
        raise Exception("Error: True cost doesn't match cost returned by TSP implementation")

def check_tsp():
    cities1 = [(1,1), (2,2)]
    cities2 = [(1,1), (2,2), (3,3)]
    cities3 = [(1,1), (0,1), (0, 0), (1, 0)]
    cities4 = [(1,1), (2,1), (0, 0), (3, 0)]
    sqrt2 = sqrt(2.0)
    check_tsp_for_case(cities1, 2*sqrt2)
    check_tsp_for_case(cities2, 4*sqrt2)
    check_tsp_for_case(cities3, 4.0)
    check_tsp_for_case(cities4, 4 + 2*sqrt2)

def run_tests():
    try:
        check_mst_heuristic()
        check_tsp()
        print("All tests passed!")
    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    run_tests()
    solve_problem()
