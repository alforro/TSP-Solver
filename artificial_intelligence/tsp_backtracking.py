"""Traveling Salesman Sample.

   This is a sample using the routing library python wrapper to solve a
   Traveling Salesman Problem.
   The description of the problem can be found here:
   http://en.wikipedia.org/wiki/Travelling_salesman_problem.
   The optimization engine uses local search to improve solutions, first
   solutions being generated using a cheapest addition heuristic.
   Optionally one can randomly forbid a set of random connections between nodes
   (forbidden arcs).
"""


import timeit
import random
import argparse
import numpy as np
import math
import operator
import networkx as nx
parser = argparse.ArgumentParser()
parser.add_argument('--tsp_size', default = 8, type = int,
                     help='Size of Traveling Salesman Problem instance.')

check_shortest = 100000000000
shortest = []
expanded_nodes = 0

def get_shortest():
  return shortest

def get_expanded_nodes():
  return expanded_nodes

def get_check_shortest():
  return shortest

def set_expanded_nodes(n):
  global expanded_nodes
  expanded_nodes=n

def set_check_shortest(n):
  global check_shortest
  check_shortest=n

def set_shortest(n):
  global shortest
  shortest=n


class RandomMatrix(object):
  """Random Matrix"""
  def create_graph(self):
    self.graph = nx.complete_graph(len(self.coordinates))
    print self.graph
    for edge in self.graph:
        print edge
    return self.graph

  def euclidean_distance(self, (x1,y1), (x2,y2)):
    """This function calculates the euclidean distance between the nodes"""
    xdiff=x2-x1
    ydiff=y2-y1
    return math.sqrt(xdiff*xdiff + ydiff*ydiff)

  def normalize(self,a):
      """ All the values below 0.5 should be 0 and in the other case will be 1"""
      if (a > 0.5):
          return 1
      else:
          return 0

  def distance(self, size):
    n=len(self.coordinates)
    self.distance_matrix=np.zeros((n,n),dtype=np.float)
    for coord, dist in self.sorted_d:
      (row,column)=coord
      self.distance_matrix[row, column]=dist

  def get_coordinates(self, solution):
      coord=[]
      for node in solution:
          coord_temp =  self.coordinates[node]
          coord+=[coord_temp]
      return coord

  def get_coord_test(self, solution):
      coord = []
      for node in solution:
          coord_temp = self.coordinates[node]
          coord += [coord_temp]
      return np.asarray(coord)

  def __init__(self, size):
    """Create the matrix that will store the coordinates of the nodes"""
    vectorial_function = np.vectorize(self.normalize)
    self.matrix = vectorial_function(np.random.random((size, size)))
    print self.matrix
    self.coordinates = np.nonzero(self.matrix)
    self.coordinates = np.transpose(self.coordinates)

    n = len(self.coordinates)
    D = {}  # dictionary to hold n times n matrix
    for i in range(n - 1):
          for j in range(i + 1, n):
            (x1, y1) = self.coordinates[i]
            (x2, y2) = self.coordinates[j]
            D[i, j] = self.euclidean_distance((x1,y1),(x2,y2))
            D[j, i] = D[i, j]
    # sort the values of D
    self.sorted_d = sorted(D.items(),key=operator.itemgetter(0))


def tsp_backtracking(A,l,length_so_far,distance_matrix):
    """TSP implementation with a backtracking approach"""
    n = len(A)
    global expanded_nodes,check_shortest, shortest
    if expanded_nodes<30000000:
      if l==n-1:
        if length_so_far + distance_matrix[A[l]][A[0]]<check_shortest:
          check_shortest=length_so_far + distance_matrix[A[l]][A[0]]
          shortest=A+[0]
      else:
        for i in range(l+1,n):
          expanded_nodes+=1
          A[l+1], A[i] = A[i], A[l+1]
          new_length = length_so_far + distance_matrix[A[l]][A[l+1]]
          if new_length<=check_shortest:
            check_shortest=min(check_shortest,tsp_backtracking(A,l+1,new_length,distance_matrix))
          A[l+1], A[i] = A[i], A[l+1]
      return check_shortest
    else:
      return check_shortest

def tsp_heuristic(graph,distance_matrix):
    global shortest, check_shortest, expanded_nodes
    a = [0]
    b = range(1,len(graph))
    check_shortest = 0
    expanded_nodes=0
    while b:
        # last node placed in a
        min = 100000000
        last = a[-1]
        neighbors = graph[last]
        for n in neighbors:
            if n not in a:
                if distance_matrix[a[-1]][n] < min:
                    min = distance_matrix[a[-1]][n]
                    next = n
        b.remove (next)
        expanded_nodes += 1
        check_shortest += distance_matrix[next][a[-1]]
        a.append (next)
    a.append(0)
    shortest = a
    check_shortest += distance_matrix[shortest[len(shortest)-2]][0]
    return check_shortest

def two_opt_swap(route, i, k):
  if i<>0 and k<>0:
  #print "LA SOLUCION QUE SE ENVIO DENTRO DEL SWAP ES:",route
    new_route = route[0:i]
    temp = route[i:k]
    temp.reverse()
    new_route+=temp
    new_route+=route[k:]
    return new_route
  else:
    return route

def tsp_two_opt(solution, distance_matrix):
    global expanded_nodes
    changes = 0
    number_of_nodes=len(solution)

    while(changes<3):
      changes+=1
      best_distance = calculate_cost(solution, distance_matrix)
      for i in range(number_of_nodes - 1):
        for k in range(i+1,number_of_nodes):
          expanded_nodes+=1
          new_route = two_opt_swap(solution, i, k)
          new_distance = calculate_cost(new_route, distance_matrix)
          if (new_distance < best_distance):
            solution = new_route
    return solution

def calculate_cost(solution, distance_matrix):
  cost = 0
  #print distance_matrix
  for i in range(len(solution)-1):
    cost += distance_matrix[solution[i]][solution[i+1]]
  return cost

def main(args):
  # Create routing model
  if args.tsp_size > 0:
      # hasta aca ya se crea la matriz de puntos, la de distancia y se ordena.
      matrix = RandomMatrix(args.tsp_size)
      # se genera la matriz de distancias
      matrix.distance(args.tsp_size)
      # se crea un grafo completamente conexo que se utilizara para recorrer el grafo e ir agregando los pesos
      graph = matrix.create_graph()

      A=[]
      for i in graph:
        A+=[i]
      print A
      start = timeit.default_timer()
      print "LA LONGITUD DE A ES:",len(A)
      #solution =tsp(A,0,0, matrix.distance_matrix)
      solution = tsp_heuristic(graph, matrix.distance_matrix)
      stop = timeit.default_timer()
      print "El tiempo de ejecucion fue de: ",stop - start
      print "El costo de la solucion es de: ", solution
      print "La cantidad de nodos expandidos fue de: ", expanded_nodes
      print "Y el path es:", shortest
      #opt2 = other_two_otp(shortest,matrix.distance_matrix)
      opt2 = tsp_two_opt(shortest, matrix.distance_matrix)
      print "el camino original fue", shortest
      print "la optimizacion es:", opt2
      new_cost = calculate_cost(opt2, matrix.distance_matrix)
      print "El costo original:", solution
      print "El costo optimizado", new_cost
      print "La solucion original fue:", shortest
      print "El camino optimizado fue", opt2
      #coordenadas = matrix.get_coordinates(shortest)
      coordenadas = matrix.get_coord_test(shortest)



if __name__ == '__main__':
  main(parser.parse_args())
