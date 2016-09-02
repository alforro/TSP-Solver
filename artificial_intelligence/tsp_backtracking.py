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
def find_shortest_path(graph, current, parent, distance_matrix, path,cost):
    global shortest
    global check_shortest
    path = path + [current]
    #print "ahora el path es:",path
    if cost>check_shortest:
        return None, None

    #add the cost of the final node to the origin node
    if len(path)==len(distance_matrix) and cost+distance_matrix[current][0]< check_shortest:
        path = path + [0]
        cost += distance_matrix[current][0]
        return path, cost
    #shortest = None
    for edge in graph[current]:
        if edge not in path:
            (newpath, newcost) = find_shortest_path(graph, edge, current, distance_matrix,path,cost+distance_matrix[parent,current])

            if path:
                trash = path.pop()

            if newpath:
                if newcost<check_shortest:
                    shortest = newpath
                    check_shortest=newcost
                    print "el shortest es", check_shortest

    return shortest, cost



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
      if (a > 0.7):
          return 1
      else:
          return 0

  def distance(self, size):
    n=len(self.coordinates)
    print n
    print "Coordenadas:"
    print self.coordinates
    self.distance_matrix=np.zeros((n,n),dtype=np.float)
    for coord, dist in self.sorted_d:
      (row,column)=coord
      self.distance_matrix[row, column]=dist

  def get_coordinates(self, solution):
      coord=[]
      dist=[]
      print "LA SOLUCION ES ", solution
      for node in solution:
          coord_temp =  self.coordinates[node]
          coord+=[coord_temp]

      #[coord for coord, dist in self.sorted_d]
      print "las coordenadas son: ", coord
      return coord
  def get_coord_test(self, solution):
      coord = []
      dist = []
      for node in solution:
          coord_temp = self.coordinates[node]
          coord += [coord_temp]
      return np.asarray(coord)
  def __init__(self, size):
    """Create the matrix that will store the coordinates of the nodes"""
    vectorial_function = np.vectorize(self.normalize)
    self.matrix = vectorial_function(np.random.random((size, size)))
    print self.matrix
    #extract the coordinates equals to 1
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

def tsp_backtrack(A,current,path,cost,distance_matrix):
    global check_shortest
    global shortest
    print "el valor de path es:", path
    if len(path)==len(int(A)):
      print "el valor de path es:", path
      if cost+distance_matrix[path[len(path)-1]][0] < check_shortest:
        check_shortest=cost+distance_matrix[path[len(path)-1]][0]
        shortest=path+[0]
        print "ahora el camino mas corto es de:", check_shortest
        print "y el camino es: ", shortest
    else:
      for i in range(len(path)+1,len(distance_matrix)):
        if i not in path:
          print "el cost + distance es:", cost+distance_matrix[current][i]
          if cost+distance_matrix[current][i]<check_shortest:
            check_shortest = min(check_shortest, tsp_backtrack(graph, i, path + [i],cost+distance_matrix[current][i], distance_matrix))
            check_shortest=cost
          if path:
            trash=path.pop()
    return check_shortest


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
    return check_shortest

def two_opt(solution, distance_matrix):
    minchange = 0;
    while(True):
      for i in range(len(solution)-2):
        print i
        for j in range(i+2,len(solution)-1):
          print "(i, j) = (",i,",",j,")"
          change = distance_matrix[i][j]+distance_matrix[i+1][j+1]-distance_matrix[i][i+1]-distance_matrix[j][j+1]
          change = 0
          if(minchange>change):
            minchange = change
            mini=i
            minj=j
        #solution[i+1], solution[i+2] = solution[minj], solution[mini]
        print solution
      break
    return solution

def other_two_otp(solution, distance_matrix):
    n=len(solution)-1
    print "fuera de i:", solution
    for i in range(n-3):
      for j in range(i+1,n-1):
        print "Dentro de j:", solution
        print "(",i,",",j,")"
        newdistance = distance_matrix[i][j]+distance_matrix[i+1][j+1]-distance_matrix[i][i+1]-distance_matrix[j][j+1]
        print "i=>j (",i,",",j,")"
        print "i+1=>j+1 (",i + 1,",",j + 1,")"
        print "i=>i+1 (",i,",",i + 1,")"
        print "j=>j+1 (",j,",",j + 1,")"
        print "before", newdistance
        print solution
        if newdistance<0:
          print "ENTRO ACA CARAJO"
          #old_path = solution
          #solution[i + 1]= old_path[i + j]
          #solution[i + j] = old_path[i + 1]
          print "solution[i+1]", solution[i + 1], i+1
          print "solution[i+j]", solution[i + 1], j+ i
          print "solution[i+j]", solution[i + 1]
          print "solution[i+1]", solution[i + 1]
          solution[i+1], solution[i+j] = solution[i+j], solution[i+1]
    return solution
          #for k in range(j):
            #solution[i + k] = old_path[i + j - k]



def get_shortest():
    return shortest

def get_expanded_nodes():
    return expanded_nodes

def set_expanded_nodes(n):
    global expanded_nodes
    expanded_nodes=n

def main(args):
  # Create routing model
  if args.tsp_size > 0:
      # hasta aca ya se crea la matriz de puntos, la de distancia y se ordena.
      matrix = RandomMatrix(args.tsp_size)
      # se genera la matriz de distancias
      matrix.distance(args.tsp_size)
      # se crea un grafo completamente conexo que se utilizara para recorrer el grafo e ir agregando los pesos
      graph = matrix.create_graph()

      #(solution, cost) = find_shortest_path(graph,0,0, matrix.distance_matrix,[],0)
      print "ACA EMPIEZA EL KILOMBO"
      #solution= tsp_backtrack(graph,0,[0],0,matrix.distance_matrix)
      A=[]
      for i in graph:
        A+=[i]
      print A
      start = timeit.default_timer()
      print "LA LONGITUD DE A ES:",len(A)
      #solution =tsp(A,0,0, matrix.distance_matrix)
      solution = tsp_heuristic(graph)
      stop = timeit.default_timer()
      print "El tiempo de ejecucion fue de: ",stop - start
      print "El costo de la solucion es de: ", solution
      print "La cantidad de nodos expandidos fue de: ", expanded_nodes
      print "Y el path es:", shortest
      opt2 = other_two_otp(solution,matrix.distance_matrix)
      print "la optimizacion es:", opt2
      #coordenadas = matrix.get_coordinates(shortest)
      coordenadas = matrix.get_coord_test(shortest)
      for each in coordenadas:
          print each[0]


if __name__ == '__main__':
  main(parser.parse_args())
