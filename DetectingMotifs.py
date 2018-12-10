# CS224W Fall 2018
# Yardena Hirsch and Shelby Marcus
# Motif Detection
# using Directed, Unweighted Graph

import snap
import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt
import random
from sets import Set

class DetectMotifs:
	def __init__(self, graphName):
		self.graphName = graphName
		self.directed_3 = self.load_3_subgraphs()
		self.motif_counts = [0]*len(self.directed_3)
		self.detectMotifs()

	def detectMotifs(self):
		zScoresGrid = self.getZScores()
		self.plotZScores(zScoresGrid)
	
	def getMotifMeanStd(self):
		motifs = np.zeros((10,13))
		for i in range(10):
			graph = self.load_graph(self.graphName)
			config_graph, clustering_coeffs = self.gen_config_model_rewire(graph, 8000)
			self.enumerate_subgraph(config_graph)
			motifs[i, :] = list(motif_counts)
		motifMean = list(np.mean(motifs, 0))
		motifStd = list(np.std(motifs, 0))
		return motifMean, motifStd

	def getZScores(self):
		motifMean, motifStd = self.getMotifMeanStd()
		graph = self.load_graph(self.graphName)
		self.enumerate_subgraph(graph)
		zScores = []
		for i in range(len(motifMean)):
			if motifStd[i] == 0: 
				zScore = 0.0
			else: 
				zScore = float(motif_counts[i] - motifMean[i])/motifStd[i]
			print "i: ", str(i+1), " : ", zScore
			zScores.append(zScore)
		print zScores

		return zScores

	def plotZScores(self, zScores):
		plt.plot(range(1, len(zScores) + 1), zScores)
		plt.xlabel('Motif Index')
		plt.ylabel('Z Score')
		plt.title('Z Scores for ' + "Stanford Memes Graph")
		plt.show()
		plt.savefig(self.graphName + '.png', format='png')
	
	def load_graph(self, name):
	    '''
	    Helper function to load graphs.
	    Check that the respective .txt files are in the same folder as this script;
	    if not, change the paths below as required.
	    '''
	    mapping = snap.TStrIntSH()
	    G = snap.LoadEdgeListStr(snap.PNGraph, self.graphName, 0, 1, mapping)
	    return G

	def load_3_subgraphs(self):
		'''
		Loads a list of all 13 directed 3-subgraphs.
		The list is in the same order as the figure in the HW pdf, but it is
		zero-indexed 
		'''
		return [snap.LoadEdgeList(snap.PNGraph, "./subgraphs/{}.txt".format(i), 0, 1) for i in range(13)]

	def plot_q3_1(self, clustering_coeffs):
		'''
		Helper plotting code for question 3.1 Feel free to modify as needed.
		'''
		plt.plot(np.linspace(0,8000,len(clustering_coeffs)), clustering_coeffs)
		plt.xlabel('Iteration')
		plt.ylabel('Average Clustering Coefficient')
		plt.title('Random Edge Rewiring: Clustering Coefficient')
		plt.savefig('q3_1.png', format='png')
		plt.show()

	def gen_config_model_rewire(self, graph, iterations=8000):
		'''
		Note that this will modify the graph
		'''
		clustering_coeffs = []
		##########################################################################
		edges = [[edge.GetSrcNId(), edge.GetDstNId()] for edge in graph.Edges()]
		numIterations = 0
		while numIterations < iterations:
			# if numIterations % 100 == 0:
			# 	ccf = snap.GetClustCf(graph, 1000)
			# 	clustering_coeffs.append(ccf)
			# Choose distinct edges
			while True:
				firstEdge = random.choice(edges)
				secondEdge = random.choice(edges)
				if firstEdge != secondEdge:
					break
			# Get random start and end points of each edge
			u = random.choice(firstEdge)
			v = firstEdge[0] if u == firstEdge[1] else firstEdge[1]
			w = random.choice(secondEdge)
			x = secondEdge[0] if w == secondEdge[1] else secondEdge[1]
			# If there are no self edges or multi edges, remove old edges from graph and set, add new
			# edges to graph and set, and update number of iterations
			if u != w and v != x and not graph.IsEdge(u, w) and not graph.IsEdge(v, x):
				# Remove edges
				edges.remove(firstEdge)
				edges.remove(secondEdge)
				graph.DelEdge(firstEdge[0], firstEdge[1])
				graph.DelEdge(secondEdge[0], secondEdge[1])
				# Add edges
				edges.append([u, w])
				edges.append([v, x])
				graph.AddEdge(u, w)
				graph.AddEdge(v, x)
				# Update iterations count
				numIterations += 1

		##########################################################################
		return graph, clustering_coeffs

	def match(self, G1, G2):
		'''
		This function compares two graphs of size 3 (number of nodes)
		and checks if they are isomorphic.
		It returns a boolean indicating whether or not they are isomorphic
		You should not need to modify it, but it is also not very elegant...
		'''
		if G1.GetEdges() > G2.GetEdges():
			G = G1
			H = G2
		else:
			G = G2
			H = G1
		# Only checks 6 permutations, since k = 3
		for p in permutations(range(3)):
			edge = G.BegEI()
			matches = True
			while edge < G.EndEI():
				if not H.IsEdge(p[edge.GetSrcNId()], p[edge.GetDstNId()]):
					matches = False
					break
				edge.Next()
			if matches:
				break
		return matches

	def count_iso(self, G, sg, verbose=False):
		'''
		Given a set of 3 node indices in sg, obtains the subgraph from the
		original graph and renumbers the nodes from 0 to 2.
		It then matches this graph with one of the 13 graphs in
		directed_3.
		When it finds a match, it increments the motif_counts by 1 in the relevant
		index

		IMPORTANT: counts are stored in global motif_counts variable.
		It is reset at the beginning of the enumerate_subgraph method.
		'''
		#if verbose:
			#print(sg)
		nodes = snap.TIntV()
		for NId in sg:
			nodes.Add(NId)
		# This call requires latest version of snap (4.1.0)
		SG = snap.GetSubGraphRenumber(G, nodes)
		for i in range(len(self.directed_3)):
			if self.match(self.directed_3[i], SG):
				motif_counts[i] += 1

	def enumerate_subgraph(self, G, k=3, verbose=False):
		'''
		This is the main function of the ESU algorithm.
		Here, you should iterate over all nodes in the graph,
		find their neighbors with ID greater than the current node
		and issue the recursive call to extend_subgraph in each iteration

		A good idea would be to print a progress report on the cycle over nodes,
		So you get an idea of how long the algorithm needs to run
		'''
		global motif_counts
		motif_counts = [0]*len(self.directed_3) # Reset the motif counts (Do not remove)
		##########################################################################
		counter = 0
		numNodes = G.GetNodes()
		for v in G.Nodes():
			counter += 1
			v_id = v.GetId()
			v_degree = v.GetDeg()
			neighbor_ids = [v.GetNbrNId(i) for i in range(0, v_degree)]
			v_ext = Set([x for x in neighbor_ids if x > v_id])
			sg = Set([v_id])
			self.extend_subgraph(G, k, sg, v_ext, v_id, verbose)
		##########################################################################


	def extend_subgraph(self, G, k, sg, v_ext, node_id, verbose=False):
		'''
		This is the recursive function in the ESU algorithm
		The base case is already implemented and calls count_iso. You should not
		need to modify this.

		Implement the recursive case.
		'''
		# Base case (you should not need to modify this):
		if len(sg) is k:
			self.count_iso(G, sg, verbose)
			return
		# Recursive step:
		##########################################################################
		v_ext_copy = Set(v_ext)
		while v_ext:
			w = v_ext.pop()
			w_node = G.GetNI(w)
			w_degree = w_node.GetDeg()
			w_neighbors = [w_node.GetNbrNId(i) for i in range(0, w_degree)]
			# Keep Ids greater than node_id and convert to set
			w_neighbors = Set([x for x in w_neighbors if x > node_id])
			# Get rid of nodes in subgraph and v_ext and remove w 
			w_neighbors = w_neighbors.difference(sg)
			w_neighbors = w_neighbors.difference(v_ext_copy)
			# Add filtered neighbors to v_ext_new
			v_ext_new = Set(v_ext).union(w_neighbors)
			sg_new = Set(sg)
			sg_new.add(w)
			self.extend_subgraph(G, k, sg_new, v_ext_new, node_id, verbose)
		return 
		##########################################################################
