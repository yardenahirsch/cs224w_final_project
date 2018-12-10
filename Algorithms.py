# CS224W Fall 2018
# Yardena Hirsch and Shelby Marcus
# Algorithms

import networkx as nx
from networkx.algorithms import approximation
from networkx.algorithms import community as comm
from DetectingMotifs import DetectMotifs
import snap
import numpy as np
import itertools
from itertools import permutations
from matplotlib import pyplot as plt
import random
from sets import Set
import sys
import collections
from collections import defaultdict
import community 
from collections import Counter 


reload(sys)
sys.setdefaultencoding('utf-8')

# Load entire users graph 
unweighted_graph_file = "graphs/unweighted_users_graph.gph"
weighted_graph_file = "graphs/weighted_users_graph.gph"
undir_unw_graph = nx.read_edgelist(unweighted_graph_file)
undir_w_graph = nx.read_edgelist(weighted_graph_file, data = (('weight', float),))
dir_w_graph = nx.read_edgelist(weighted_graph_file, create_using = nx.DiGraph(), data = (('weight', float),))

# Load training graphs 
unweighted_graph_file_train = "graphs/unweighted_train_graph.gph"
weighted_graph_file_train = "graphs/weighted_train_graph.gph"
undir_unw_graph_train = nx.read_edgelist(unweighted_graph_file_train)
undir_w_graph_train = nx.read_edgelist(weighted_graph_file_train, data = (('weight', float),))
dir_w_graph_train = nx.read_edgelist(weighted_graph_file_train, create_using = nx.DiGraph(), data = (('weight', float),))

# Load testing graphs 
unweighted_graph_file_test = "graphs/unweighted_test_graph.gph"
weighted_graph_file_test = "graphs/weighted_test_graph.gph"
undir_unw_graph_test = nx.read_edgelist(unweighted_graph_file_test)
undir_w_graph_test = nx.read_edgelist(weighted_graph_file_test, data = (('weight', float),))
dir_w_graph_test = nx.read_edgelist(weighted_graph_file_test, create_using = nx.DiGraph(), data = (('weight', float),))



def getStats():
	# Basic stats: 
	print "Stats for directed, Weighted Graph:"
	print "Number of Nodes: ", dir_w_graph.number_of_nodes()
	print "Number of Edges: ", dir_w_graph.number_of_edges()
	
	wccs = sorted(list(nx.weakly_connected_components(dir_w_graph)), key = lambda x: len(x), reverse = True)
	wcc_g = dir_w_graph.subgraph(wccs[0])
	print "Number of nodes in largest WCC: ", wcc_g.number_of_nodes()
	print "Number of edges in largest WCC: ", wcc_g.number_of_edges()

	sccs = sorted(list(nx.strongly_connected_components(dir_w_graph)), key = lambda x: len(x), reverse = True)
	scc_g = dir_w_graph.subgraph(sccs[0])
	print "Number of nodes in largest SCC: ", scc_g.number_of_nodes()
	print "Number of edges in largest SCC: ", scc_g.number_of_edges()


	print "--------------------------------------------"

	print "Stats for Undirected, Unweighted Graph:"
	print "Number of Nodes: ", undir_unw_graph.number_of_nodes()
	print "Number of Edges: ", undir_unw_graph.number_of_edges()
	print "CC: ", nx.average_clustering(undir_unw_graph)
	print "Number of connected components: ", nx.number_connected_components(undir_unw_graph)
	print "Number of Bridges: ", len(list(nx.bridges(undir_unw_graph)))

	print "--------------------------------------------"

	print "Stats for Undirected, Weighted Graph:"
	print "Number of Nodes: ", undir_w_graph.number_of_nodes()
	print "Number of Edges: ", undir_w_graph.number_of_edges()
	print "CC: ", nx.average_clustering(undir_w_graph, weight = 'weight')


def plotDegreeDist(g):
	degree_sequence = list(g.degree())  # degree sequence
	degrees = Counter()
	for n, d in degree_sequence:
		degrees[d] += 1
	y_values = degrees.values()
	x_values = degrees.keys()

	y_values = degrees.values()
	plt.loglog(x_values, y_values, linestyle = 'dotted', color = 'b')

	plt.xlabel('Node Degree (log)')
	plt.ylabel('Proportion of Nodes with a Given Degree (log)')
	plt.title('Degree Distribution of the Stanford Memes Network')
	plt.savefig("DegDist", format='eps', dpi=1000)
	plt.show()


def getMotifs():
	# Get 3-node motifs on unweighted graph 
	motifz = DetectMotifs(unweighted_graph_file)

def linkPrediction(num_predicted_links):
	# Construct core graph with nodes that have edge weights greater than one and degree greater than one. 
	# Use undirected, weighted graph for train and test
	g_train = createUndirectedWeightedGraph(dir_w_graph_train)
	g_test = createUndirectedWeightedGraph(dir_w_graph_test)
	nodes_in_core = []
	print "Number of nodes in undir_w_graph_train: ", g_train.number_of_nodes()
	print "Number of edges in undir_w_graph_train: ", g_train.number_of_edges()
	print "Number of nodes in undir_w_graph_test: ", g_test.number_of_nodes()
	print "Number of edges in undir_w_graph_test: ", g_test.number_of_edges()
	for u in g_train.nodes():
		if g_train.degree(u) > 2:
			nodes_in_core.append(u)
	core_g = g_train.subgraph(nodes_in_core)
	print "Number of nodes in core: ", core_g.number_of_nodes()
	print "Number of edges in core: ", core_g.number_of_edges()

	scores = list(nx.adamic_adar_index(core_g))
	# Sort edges in descending order to use for link prediction
	scores = sorted(scores, key= lambda x: x[2], reverse = True)
	preds = [(u, v) for u, v, w in scores][0:num_predicted_links]

	num_correct_predictions = 0.0
	for pred in preds:
		if g_test.has_edge(pred[0], pred[1]):
			num_correct_predictions +=1

	print "Number of correct predictions: ", num_correct_predictions
	print "Number of predictions: ", num_predicted_links

def communityDetection(g):
	# greed modularity community detection
	greedy_communities = list(comm.greedy_modularity_communities(g))
	greedy_communities_dict = {}
	for i, c in enumerate(greedy_communities):
		for node_id in c:
			greedy_communities_dict[node_id] = i
	greedy_score = community.modularity(greedy_communities_dict, g)
	print "Greedy Number of Communities: ", len(set(greedy_communities_dict.values()))
	print "Greedy Modularity: ", greedy_score
	greedy_communities_sorted = sorted(greedy_communities, key = lambda x: len(x), reverse = True)
	for i in range(1, 6):
		print "Size of", i,  "Community: ", len(greedy_communities_sorted[i])


	louvain_communities = community.best_partition(g)
	louvain_score = community.modularity(louvain_communities, g)
	print "Louvain Number of Communities: ", len(set(louvain_communities.values()))
	print "Louvain Modularity: ", louvain_score
	louvain_score_dict = defaultdict(list)
	for node_id, comm_id in louvain_communities.items():
		louvain_score_dict[comm_id].append(node_id)
	louvain_communities_list = louvain_score_dict.values()
	louvain_communities_list = sorted(louvain_communities_list, key = lambda x: len(x), reverse = True)
	for i in range(1, 6):
		print "Size of ", i,  " Community: ", len(louvain_communities_list[i])

def removeTies(isReverse):
	# Use undirected weighted graph to detect weak ties and remove them 
	g = createUndirectedWeightedGraph(dir_w_graph)
	original_sz = float(max(nx.connected_component_subgraphs(g), key=len).number_of_nodes())
	wcc_szs = []
	while True:
		# Get largest WCC
		wcc_g = max(nx.connected_component_subgraphs(g), key=len)
		if wcc_g.number_of_edges() == 0:
			break
		wcc_szs.append(wcc_g.number_of_nodes()) 
		# Get the edge weights in the weakly connected component 
		edge_weights = sorted(wcc_g.edges(data = True), key = lambda x: x[2], reverse = isReverse)
		# Remove the first weak tie
		u, v = edge_weights[0][0], edge_weights[0][1]
		wcc_g.remove_edge(u, v)
		g = wcc_g
	print wcc_szs
	wcc_szs = [x/original_sz for x in wcc_szs]
	return wcc_szs


def plotRemoveWeakStrongTies():
	wcc_szs = removeTies(False)
	plt.plot(wcc_szs, label = "Weak Ties", color = "r")

	scc_szs = removeTies(True)
	plt.plot(scc_szs, label = "Strong Ties", color = "b")

	plt.legend()
	plt.xlabel("Number of Nodes Removed")
	plt.ylabel("Size of Largest Connected Component")
	plt.title("Link Removal by Tie Strength")
	plt.show()

def createUndirectedWeightedGraph(directedWeightedGraph): 
	g = nx.Graph()
	for u, v, w in directedWeightedGraph.edges(data = True):
		weight = w["weight"]
		if g.has_edge(u, v):
			g[u][v]["weight"] += weight
		else:
			g.add_edge(u, v, weight = weight)
	return g 


def testUndirectedGraph():
	counter = 0
	g = createUndirectedWeightedGraph()
	for u, v, w in g.edges(data = True):
		if dir_w_graph.has_edge(u, v) and dir_w_graph.has_edge(v, u):
			counter += 1
			print "Directed: ", u, v, dir_w_graph[u][v]["weight"], dir_w_graph[v][u]["weight"]
			print "Undirected: ", u, v, w["weight"]
			print "----------------------------------"
	print counter 


def main():
	g = createUndirectedWeightedGraph(dir_w_graph)
	getStats()
	plotDegreeDist(undir_unw_graph)
	linkPrediction(1000)
	getMotifs()
	communityDetection(g)
	plotRemoveWeakStrongTies()
	


if __name__ == '__main__':
	main()


