from django.shortcuts import render, render_to_response, redirect
#import tsp_backtracking as tsp
from django.template import RequestContext

from tsp_backtracking import *
from tsp_backtracking import check_shortest, shortest, expanded_nodes
# Create your views here.
from models import TSP_Solution

def home(request):
    # TODO: Mostrar el grafico del recorrido
    coordenadas=[[288, 149], [288, 129], [280, 133], [270, 133], [260, 129], [252, 125], [256, 141], [246, 141], [236, 145], [228, 145], [220, 145], [212, 145], [204, 145], [196, 145], [188, 145], [180, 125], [180, 117], [180, 109], [180, 101], [180, 93], [188, 93], [196, 101], [204, 109], [212, 117], [220, 125], [228, 125], [228, 117], [228, 109], [228, 101], [236, 101], [236, 93], [228, 93], [228, 85], [236, 85], [260, 85], [260, 93], [252, 101], [260, 109], [268, 97], [276, 101], [280, 109], [288, 109], [284, 101], [284, 93], [276, 93], [276, 85], [284, 85], [284, 77], [284, 69], [284, 61], [284, 53], [276, 53], [276, 61], [276, 69], [276, 77], [260, 77], [260, 69], [260, 61], [260, 53], [260, 45], [260, 37], [260, 29], [252, 21], [236, 21], [228, 21], [228, 29], [236, 29], [236, 37], [228, 37], [228, 45], [236, 45], [236, 53], [228, 53], [228, 61], [236, 61], [236, 69], [236, 77], [228, 77], [228, 69], [220, 73], [212, 65], [204, 57], [196, 49], [188, 41], [180, 37], [180, 45], [172, 45], [172, 37], [172, 29], [180, 29], [180, 21], [172, 21], [156, 25], [162, 9], [148, 9], [136, 9], [128, 9], [120, 9], [124, 21], [132, 21], [124, 29], [124, 37], [124, 45], [124, 53], [124, 61], [132, 61], [140, 65], [124, 69], [104, 57], [104, 49], [104, 41], [104, 33], [104, 25], [104, 17], [92, 9], [80, 9], [72, 9], [64, 21], [72, 25], [80, 25], [80, 25], [80, 41], [88, 49], [104, 65], [104, 73], [104, 81], [104, 89], [104, 97], [104, 105], [104, 113], [104, 121], [124, 125], [124, 117], [124, 109], [124, 101], [124, 93], [124, 85], [124, 77], [132, 81], [148, 85], [164, 81], [172, 77], [172, 69], [172, 61], [172, 53], [180, 53], [180, 61], [180, 69], [180, 77], [180, 85], [172, 85], [172, 93], [172, 101], [172, 109], [172, 117], [172, 125], [164, 137], [172, 145], [164, 145], [156, 145], [156, 137], [148, 137], [148, 145], [140, 145], [140, 137], [132, 137], [132, 145], [124, 145], [116, 145], [104, 145], [104, 137], [104, 129], [56, 113], [56, 105], [56, 97], [56, 89], [48, 83], [56, 81], [56, 73], [56, 65], [56, 57], [56, 49], [72, 49], [72, 41], [64, 41], [56, 41], [56, 33], [56, 25], [56, 17], [56, 9], [44, 11], [32, 17], [24, 17], [16, 17], [16, 25], [24, 25], [32, 25], [44, 27], [44, 35], [44, 43], [48, 51], [40, 51], [40, 63], [48, 63], [48, 73], [40, 73], [40, 83], [32, 81], [32, 73], [32, 65], [32, 57], [32, 49], [32, 41], [24, 45], [8, 41], [8, 49], [16, 57], [8, 57], [8, 65], [8, 73], [8, 81], [8, 89], [8, 97], [8, 109], [16, 109], [16, 97], [24, 89], [32, 89], [32, 97], [40, 99], [48, 99], [40, 113], [32, 113], [32, 121], [32, 129], [32, 137], [32, 145], [32, 153], [32, 161], [32, 169], [40, 169], [40, 161], [40, 153], [40, 145], [40, 137], [40, 129], [40, 121], [56, 121], [56, 129], [56, 137], [56, 145], [56, 153], [56, 161], [56, 169], [64, 165], [64, 157], [80, 157], [90, 165], [104, 169], [104, 161], [104, 153], [116, 161], [124, 169], [132, 169], [140, 169], [148, 169], [156, 169], [164, 169], [172, 169], [188, 169], [196, 161], [196, 169], [204, 169], [212, 169], [220, 169], [228, 161], [228, 169], [236, 169], [246, 157], [256, 157], [288, 149]]
    return render(request,'index.html', {"coordinates_django":coordenadas})

def run_tsp(request):
    global check_shortest
    global shortest
    global expanded_nodes
    set_check_shortest(100000000000)
    set_shortest([])
    set_expanded_nodes(0)
    coordenadas=[[288, 149], [288, 129], [280, 133], [270, 133], [260, 129], [252, 125], [256, 141], [246, 141], [236, 145], [228, 145], [220, 145], [212, 145], [204, 145], [196, 145], [188, 145], [180, 125], [180, 117], [180, 109], [180, 101], [180, 93], [188, 93], [196, 101], [204, 109], [212, 117], [220, 125], [228, 125], [228, 117], [228, 109], [228, 101], [236, 101], [236, 93], [228, 93], [228, 85], [236, 85], [260, 85], [260, 93], [252, 101], [260, 109], [268, 97], [276, 101], [280, 109], [288, 109], [284, 101], [284, 93], [276, 93], [276, 85], [284, 85], [284, 77], [284, 69], [284, 61], [284, 53], [276, 53], [276, 61], [276, 69], [276, 77], [260, 77], [260, 69], [260, 61], [260, 53], [260, 45], [260, 37], [260, 29], [252, 21], [236, 21], [228, 21], [228, 29], [236, 29], [236, 37], [228, 37], [228, 45], [236, 45], [236, 53], [228, 53], [228, 61], [236, 61], [236, 69], [236, 77], [228, 77], [228, 69], [220, 73], [212, 65], [204, 57], [196, 49], [188, 41], [180, 37], [180, 45], [172, 45], [172, 37], [172, 29], [180, 29], [180, 21], [172, 21], [156, 25], [162, 9], [148, 9], [136, 9], [128, 9], [120, 9], [124, 21], [132, 21], [124, 29], [124, 37], [124, 45], [124, 53], [124, 61], [132, 61], [140, 65], [124, 69], [104, 57], [104, 49], [104, 41], [104, 33], [104, 25], [104, 17], [92, 9], [80, 9], [72, 9], [64, 21], [72, 25], [80, 25], [80, 25], [80, 41], [88, 49], [104, 65], [104, 73], [104, 81], [104, 89], [104, 97], [104, 105], [104, 113], [104, 121], [124, 125], [124, 117], [124, 109], [124, 101], [124, 93], [124, 85], [124, 77], [132, 81], [148, 85], [164, 81], [172, 77], [172, 69], [172, 61], [172, 53], [180, 53], [180, 61], [180, 69], [180, 77], [180, 85], [172, 85], [172, 93], [172, 101], [172, 109], [172, 117], [172, 125], [164, 137], [172, 145], [164, 145], [156, 145], [156, 137], [148, 137], [148, 145], [140, 145], [140, 137], [132, 137], [132, 145], [124, 145], [116, 145], [104, 145], [104, 137], [104, 129], [56, 113], [56, 105], [56, 97], [56, 89], [48, 83], [56, 81], [56, 73], [56, 65], [56, 57], [56, 49], [72, 49], [72, 41], [64, 41], [56, 41], [56, 33], [56, 25], [56, 17], [56, 9], [44, 11], [32, 17], [24, 17], [16, 17], [16, 25], [24, 25], [32, 25], [44, 27], [44, 35], [44, 43], [48, 51], [40, 51], [40, 63], [48, 63], [48, 73], [40, 73], [40, 83], [32, 81], [32, 73], [32, 65], [32, 57], [32, 49], [32, 41], [24, 45], [8, 41], [8, 49], [16, 57], [8, 57], [8, 65], [8, 73], [8, 81], [8, 89], [8, 97], [8, 109], [16, 109], [16, 97], [24, 89], [32, 89], [32, 97], [40, 99], [48, 99], [40, 113], [32, 113], [32, 121], [32, 129], [32, 137], [32, 145], [32, 153], [32, 161], [32, 169], [40, 169], [40, 161], [40, 153], [40, 145], [40, 137], [40, 129], [40, 121], [56, 121], [56, 129], [56, 137], [56, 145], [56, 153], [56, 161], [56, 169], [64, 165], [64, 157], [80, 157], [90, 165], [104, 169], [104, 161], [104, 153], [116, 161], [124, 169], [132, 169], [140, 169], [148, 169], [156, 169], [164, 169], [172, 169], [188, 169], [196, 161], [196, 169], [204, 169], [212, 169], [220, 169], [228, 161], [228, 169], [236, 169], [246, 157], [256, 157], [288, 149]]
    if(request.method == 'POST'):
        n = request.POST.get('n-size')
        if n > 0:
            n=int(n)
            # hasta aca ya se crea la matriz de puntos, la de distancia y se ordena.
            matrix = RandomMatrix(n)
            # se genera la matriz de distancias
            matrix.distance(n)
            # se crea un grafo completamente conexo que se utilizara para recorrer el grafo e ir agregando los pesos
            graph = matrix.create_graph()
            # (solution, cost) = find_shortest_path(graph,0,0, matrix.distance_matrix,[],0)
            A = []
            for i in graph:
                A += [i]
            print "Resolving using backtracking"
            start_backtracking = timeit.default_timer()
            backtracking_cost = tsp_backtracking(A, 0, 0, matrix.distance_matrix)
            stop_backtracking = timeit.default_timer()
            backtracking_expanded_nodes = get_expanded_nodes()
            backtracking_path = get_shortest()
            backtracking_coordinates = matrix.get_coordinates(backtracking_path)
            backtracking_result = []
            for each in backtracking_coordinates:
                backtracking_result.append([each[0],each[1]])
            coordinates_backtracking = backtracking_result
            backtracking_runtime = stop_backtracking - start_backtracking
            print "Resolving with heuristic..."
            set_check_shortest(100000000000)
            set_shortest([])
            set_expanded_nodes(0)
            start_heuristic = timeit.default_timer()
            heuristic_cost = tsp_heuristic(graph, matrix.distance_matrix)
            stop_heuristic = timeit.default_timer()
            heuristic_path = get_shortest()
            heuristic_expanded_nodes = get_expanded_nodes()
            heuristic_coordinates = matrix.get_coordinates(heuristic_path)
            heuristic_result = []
            for each in heuristic_coordinates:
                heuristic_result.append([each[0], each[1]])
            coordinates_heuristic = heuristic_result
            heuristic_runtime = stop_heuristic-start_heuristic
            print "Applying 2opt..."
            set_check_shortest(100000000000)
            set_shortest([])
            set_expanded_nodes(0)
            start_2opt = timeit.default_timer()
            two_opt_path = tsp_two_opt(heuristic_path, matrix.distance_matrix)
            two_opt_cost = calculate_cost(two_opt_path, matrix.distance_matrix)
            print "2OPT COST ES:", two_opt_cost
            print "el true costo de 2opt es:", two_opt_cost
            stop_2opt = timeit.default_timer()
            two_opt_expanded_nodes = get_expanded_nodes()
            two_opt_runtime = stop_2opt-start_2opt
            two_opt_coordinates = matrix.get_coordinates(two_opt_path)
            two_opt_result = []
            for each in two_opt_coordinates:
                two_opt_result.append([each[0], each[1]])
            coordinates_two_opt = two_opt_result

            matrix_size = n

            nodes = len(coordinates_backtracking)-1

            #TODO: add and save heuristic and 2opt information
            backtracking_solution = TSP_Solution(matrix_size=matrix_size, solution_cost=backtracking_cost, coordinates=backtracking_coordinates, expanded_nodes=int(backtracking_expanded_nodes),execution_time=backtracking_runtime, nodes=nodes, approach='Backtracking')
            backtracking_solution.save()

            print "lass coordenadas son", two_opt_result
            data = {"coordinates_django":coordinates_backtracking,
                    "matrix_size":n,
                    "coordinates_backtracking":coordinates_backtracking,
                    "coordinates_heuristic":coordinates_heuristic,
                    "coordinates_two_opt":coordinates_two_opt,
                    "backtracking_cost":backtracking_cost,
                    "heuristic_cost":heuristic_cost,
                    "two_opt_cost":two_opt_cost,
                    "backtracking_expanded_nodes":backtracking_expanded_nodes,
                    "heuristic_expanded_nodes": heuristic_expanded_nodes,
                    "two_opt_expanded_nodes": two_opt_expanded_nodes,
                    "backtracking_runtime":backtracking_runtime,
                    "heuristic_runtime":heuristic_runtime,
                    "two_opt_runtime":two_opt_runtime,
                    "nodes":nodes
            }
        return render(request, 'results.html',data)
    else:
      return render(request,'results.html',{"coordinates_django":coordenadas})