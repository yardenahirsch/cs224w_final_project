# CS224W Fall 2018
# Yardena Hirsch and Shelby Marcus
# Plots

from Antigraph import AntiGraph 
import networkx as nx
from matplotlib import pyplot as plt
import json
from collections import Counter
from community import community_louvain
from networkx.algorithms import community as mod_com
import community 
from matplotlib.lines import Line2D


# https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
cnames = {
'aliceblue':            '#F0F8FF',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightsteelblue':       '#B0C4DE',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

weighted_graph_file = "graphs/weighted_users_graph.gph"
unweighted_graph_file = "graphs/unweighted_users_graph.gph"

dir_w_graph = nx.read_edgelist(weighted_graph_file, create_using = nx.DiGraph(), data = (('weight', float),))
undir_unw_graph = nx.read_edgelist(unweighted_graph_file)
undir_w_graph = nx.read_edgelist(weighted_graph_file, data = (('weight', float),))

# constants for graphing
LAYOUT_TYPE = 'spring_layout'
SPRING_LAYOUT = 'spring_layout'
KK_LAYOUT = 'kamada_kawai_layout'
LOUVAIN = 'Louvain'
GREEDY = 'Greedy'


# load Node attributes from Dictionary Files
with open("attributes/nameAttributes.json") as json_file:  
	idsToNames = json.load(json_file)
with open("attributes/majorAttributes.json") as json_file:  
	idsToMajors= json.load(json_file)
with open("attributes/gradAttributes.json") as json_file:  
	idsToGradStatus= json.load(json_file)

def plotNodeColorMap(g):
	pos = nx.spring_layout(g, iterations=200)
	nx.draw(g, pos, node_color=range(24), node_size=800, cmap=plt.cm.Blues)
	plt.show()


def plotAntigraph(g):
	Anp = AntiGraph(nx.complement(g))
	nx.draw(g)
	plt.show()

def plotGrad(g):
	for node, grad_status in idsToGradStatus.items():
		if not grad_status: 
			idsToGradStatus[node] = 'Unknown'
		else:
			idsToGradStatus[node] = str(grad_status).strip('\n').strip()

	graduate_programs = set(['Graduate, Business', 'Graduate, Law', 'Graduate', 'Graduate, Medicine'])

	only_undergrad_and_grad_d = {}
	for node in g.nodes():
		if node in idsToGradStatus:
			if idsToGradStatus[node] in graduate_programs:
				only_undergrad_and_grad_d[node] = 'Graduate'
			elif idsToGradStatus[node] == 'Undegraduate':
				only_undergrad_and_grad_d[node] = 'Undegraduate'
			else:
				only_undergrad_and_grad_d[node] = 'Non-Student'
		else:
			only_undergrad_and_grad_d[node] = 'Non-Student'

	statuses = list(set(only_undergrad_and_grad_d.values()))
	# print statuses
	colors = ['red', 'green', 'grey']
	status_to_color_d = {status:colors[i] for i, status in enumerate(statuses)}
	print status_to_color_d
	values = [status_to_color_d[only_undergrad_and_grad_d[node]] for node in g.nodes()]

	nx.draw(g, cmap=plt.get_cmap('jet'), with_labels=False, node_color=values)
	# nx.draw_networkx_edges(g, alpha=0.5, width=0.5)
	pos=nx.spring_layout(g,k=0.9,iterations=20)
	plt.savefig('new_plots/grad_plot.png')
	plt.show()

def plotMajors(g):
	cleanUpMajors(g)

	majors = list(set(idsToMajors.values()))

	top_majors = getTopMajors()
	num_top_majors = len(top_majors)
	
	removeBottomMajors(top_majors)

	# coloring nodes
	major_to_color_d = {major:colors[i] for i, major in enumerate(top_majors)}	
	# colorBottomMajors(majors, top_majors, major_to_color_d)

	values = [major_to_color_d[idsToMajors[node]] for node in g.nodes()]

	weights = [g[u][v]['weight'] for u,v in g.edges()]

	legend_ish = zip(top_majors, colors)
	print legend_ish
	pos=nx.kamada_kawai_layout(g, dim=2)

	# pos=nx.circular_layout(g)
	# pos=nx.spring_layout(g, scale = 100, iterations = 15)

	nx.draw(g, pos, cmap=plt.get_cmap('jet'), with_labels=False, arrows=True, node_color=values, width=weights)
	plt.savefig('new_plots/cs_majors_plot_2.png')
	# plt.show()


def simplePlot(g):
	pos=nx.spring_layout(g, scale = 25, iterations = 5)
	nx.draw(g,pos)
	# labels = nx.get_edge_attributes(g,'weight')
	# nx.draw_networkx(g, pos, with_labels=False, arrows=True)
	# plt.draw()
	plt.show()

#------------------------------------------------------------------------------------------------
# Helper functions
def colorBottomMajors(majors, top_majors, major_to_color_d):
	for major in majors:
		if major not in top_majors:
			major_to_color_d[major] = colors[-1]


def removeBottomMajors(g, majors, top_majors):
	remove_nodes = set()
	for node in g.nodes():
		if idsToMajors[node] not in top_majors:
			remove_nodes.add(node)

	g.remove_nodes_from(remove_nodes)

def getTopMajors():
	majors = list(set(idsToMajors.values()))
	major_count_d = {major:0 for major in majors}
	for node, major in idsToMajors.items():
		major_count_d[major] += 1
	top_majors = [major for major, count in major_count_d.items() if count >= 45 and major != 'Unknown' and major != 'School of Engineering' and major != 'Vice Provost for Undergraduate Education']
	return top_majors

def cleanUpMajors(g):
	for node, majors in idsToMajors.items():
		if not majors: 
			idsToMajors[node] = 'Unknown'
		else:
			idsToMajors[node] = str(majors[0]).strip('\n').strip()

	for node in g.nodes():
		if node not in idsToMajors:
			idsToMajors[node] = 'Unknown'

def removeNonUndergrads(g):
	non_majors = set(['Ethics In Society', 'Office of Technology Licensing (OTL)','Undergrad Housing Front Desks',
	'Department of Energy Resources Engineerin', 'University Communications','Law School','Medicine - Med/Immunology & Rheumatology',
	'Arrillaga Family Dining Common - AFDC', 'Graduate School of Business', 'Ophthalmology', 'Medicine - Med/Blood and Marrow Transplantation',
	'Unknown', 'Stanford Management Company','Continuing Studies','Health Research and Policy - Epidemiology','Department of Energy Resources Engineering',
	'Art and Architecture Library','Institute for Computational and Mathematical Engineering (ICME)','Surgery - Anatomy','Surgery - General Surgery',
	'Psychiatry and Behavioral Sciences','Cancer Clinical Trials Office','Department of Medicine, Center for Biomedical Informatics Research','History Department'])

	undegrad_nodes_majors = {node:idsToMajors[node] for node in g.nodes() if idsToMajors[node] not in non_majors}
	majors = list(set(undegrad_nodes_majors.values()))
	nodes_w_majors = undegrad_nodes_majors.keys()
	return nodes_w_majors, majors


def reduceGraph(g, partition, comm_size_min):
	community_d = {}
	for node in g.nodes():
		community_d[partition[node]] = community_d.get(partition[node], 0) + 1

	comm_nums = sorted([(comm_size,comm_id) for comm_id, comm_size in community_d.items()], reverse=True)
	comm_to_plot = set([comm_id for comm_size, comm_id in comm_nums if comm_size > comm_size_min])
	nodes_to_draw = [node for node, comm_id in partition.items() if comm_id in comm_to_plot]

	return community_d, comm_to_plot, nodes_to_draw

#------------------------------------------------------------------------------------------------
def plotCommunitiesByLouvainForChosenMajor(g, major_name):
	majors = list(set(idsToMajors.values()))

	removeBottomMajors(g, majors, [major_name])

	# partition by Louvain Algorithm for community detection
	partition = community_louvain.best_partition(g)
	weights = [g[u][v]['weight'] for u,v in g.edges()]

	fig, ax = plt.subplots()

	pos = community_layout(g, partition)
	nx.draw(g,pos,cmap=plt.get_cmap('jet'), with_labels=False, arrows=True, node_color=major_to_color_d[major_name], width=weights)
	# legend info 	
	g.add_node(major_name)
	ax.plot([0],[0],color=major_to_cname[major_name],label=major_name,linewidth=5.0)

	plt.legend(numpoints=1,loc='lower left',fontsize='x-small',bbox_to_anchor=(-0.15,-0.1))

	cc_weighted = nx.average_clustering(g, weight = 'weight')

	plt.title(major_name+" Network \n " + "Clustering Coefficient: " + str(round(cc_weighted,5)))
	
	# plt.show()
	plt.savefig('new_plots/community_plot_by_louvain_' + major_name + '.png')


def detectCommunity(g, algorithm, layout_type):
	global LAYOUT_TYPE
	LAYOUT_TYPE = layout_type
	print "LAYOUT TYPE: ", LAYOUT_TYPE, ", algorithm: ", algorithm

	# partition based on algorithm given
	if algorithm == 'Louvain':
		partition = community_louvain.best_partition(g)
	else: # greedy
		greedy_communities = list(mod_com.greedy_modularity_communities(g))
		partition = {}
		for i, c in enumerate(greedy_communities):
			for node_id in c:
				partition[node_id] = i


	community_d, comm_to_plot, nodes_to_draw = reduceGraph(g, partition, comm_size_min=100)
	subgraph = g.subgraph(nodes_to_draw)

	# draw graphs
	drawNonInducedGraph(g, subgraph, partition, algorithm, comm_to_plot)
	drawInducedGraph(g, subgraph, partition, algorithm, comm_to_plot, community_d)

def drawNonInducedGraph(g, subgraph, partition, algorithm, comm_to_plot):	
	# colors
	comm_to_color_zip = zip(COLORS, comm_to_plot)
	comm_to_color_d = {com:color for color, com in comm_to_color_zip}
	node_to_color = [comm_to_color_d[partition[node]] for node in subgraph.nodes()]

	fig, ax = plt.subplots()
	# draw non-induced graph
	new_er_partition = {node:comm for node, comm in partition.items() if comm in comm_to_plot}
	pos = community_layout(subgraph, new_er_partition, comm_scale=2000, node_scale=200)
	nx.draw(subgraph,pos, cmap=plt.get_cmap('jet'), with_labels=False, arrows=True, node_color=node_to_color,node_size=150.0, width=.1)
	plt.title("Communities Formed From " + algorithm + " Algorithm")
	plt.savefig('new_plots/non_induced_community_' + algorithm + '_' + LAYOUT_TYPE + '.png')

def drawInducedGraph(g, subgraph, partition, algorithm, comm_to_plot, community_d):
	# colors
	comm_to_color_zip = zip(COLORS, comm_to_plot)
	comm_to_color = [color for color, com in comm_to_color_zip]

	fig, ax = plt.subplots()
	# induced graph scoring
	score = community.modularity(partition, subgraph)
	print "Induced Community Score: ", score
	# draw induced graph
	new_er_partition = {node:comm for node, comm in partition.items() if comm in comm_to_plot}
	comm_graph = community.induced_graph(new_er_partition, subgraph)  

	new_partition = {comm:comm for comm in comm_graph.nodes()}
	new_pos = community_layout(comm_graph, new_partition, comm_scale=2000,node_scale=50)	
	weights = [comm_graph[u][v]['weight']/15.0 for u,v in comm_graph.edges()]
	# node_degs = [val*35 for node,val in nx.degree(comm_graph)]
	community_self_loops = [community_d[comm] for comm in comm_graph.nodes()]
	nx.draw(comm_graph, new_pos, cmap=plt.get_cmap('jet'), with_labels=False, arrows=True, node_color=comm_to_color,node_size=community_self_loops,width=weights)
	plt.title("Supernodes from Community Detection Using " + algorithm + " Algorithm \n Modularity Score: " + str(score))
	plt.savefig('new_plots/induced_community_' + algorithm + '_' + LAYOUT_TYPE + '.png')

def plotUndergradsByMajor(g):
	nodes_w_majors, majors = removeNonUndergrads(g)
	subgraph = g.subgraph(nodes_w_majors)

	major_d = {major:int(i) for i, major in enumerate(majors)}

	values = [major_to_color_d[idsToMajors[node]] for node in subgraph.nodes()]

	partition = {node: major_d[idsToMajors[node]] for node in subgraph.nodes()}

	cc_weighted = nx.average_clustering(subgraph, weight = 'weight')
	print "cc_weighted: ", cc_weighted

	global LAYOUT_TYPE
	LAYOUT_TYPE = KK_LAYOUT
	pos = community_layout(subgraph, partition,comm_scale=4000,node_scale=300)
	
	fig, ax = plt.subplots()

	labels = [idsToMajors[node] for node in subgraph.nodes()]
	weights = [subgraph[u][v]['weight']/10.0 for u,v in subgraph.edges()]
	nx.draw(subgraph,pos,cmap=plt.get_cmap('jet'), with_labels=False, arrows=True, node_color=values, width=weights, labels=labels, node_size=100)
	
	for major in majors:
		g.add_node(major)
		ax.plot([0],[0],color=major_to_cname[major],label=major,linewidth=5.0)

	# plt.legend(numpoints=1, loc='upper left',fontsize='x-small',bbox_to_anchor=(-0.3,-0.3))
	score = community.modularity(partition, subgraph)

	plt.title('Stanford Undergraduates Clustered by Major \n Clustering Coefficient: '+ str(cc_weighted) +", Modularity: " + str(score))
	plt.savefig('new_plots/cluster_by_undergrad_majors.png')

def isPopularAmericanName(node, pop_names):
	name = idsToNames[node]
	if name in pop_names: return 1
	else: return 0

def plotByNamePopularity(g):
	file = "PopularNames.txt"
	pop_names = set()
	with open(file,"r") as names:
		for line in names: 
			pop_names.add(str(line.strip('\n')))

	print pop_names

	partition = {isPopularAmericanName(node, pop_names) for node in g.nodes()}

	global LAYOUT_TYPE
	LAYOUT_TYPE = KK_LAYOUT
	pos = community_layout(g, partition, comm_scale=2000,node_scale=400)

	fig, ax = plt.subplots()
	
	weights = [g[u][v]['weight']/10.0 for u,v in g.edges()]
	nx.draw(subgraph,pos,cmap=plt.get_cmap('jet'), with_labels=False, node_color=values, width=weights, labels=partition, node_size=100)

	# legend
	g.add_node('Popular American Name')
	ax.plot([0],[0],color='light blue',label='Popular American Name',linewidth=5.0)
	g.add_node('Non-Popular American Name')
	ax.plot([0],[0],color='light orange',label='Non-Popular American Name',linewidth=5.0)

	score = community.modularity(partition, g)
	plt.title("Stanford Undergraduates Clustered by Major \n Modularity: " + str(score))
	plt.savefig('new_plots/cluster_by_undergrad_majors.png')


#------------------------------------------------------------------------------------------------

# https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx
def community_layout(g, partition, comm_scale, node_scale):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=comm_scale)

    pos_nodes = _position_nodes(g, partition, scale=node_scale)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    if LAYOUT_TYPE == SPRING_LAYOUT:
    	pos_communities = nx.spring_layout(hypergraph, **kwargs)
    elif LAYOUT_TYPE == KK_LAYOUT:
    	pos_communities = nx.kamada_kawai_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """
    print "KWARGS IN POSITION NODES: ", kwargs
    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos
#------------------------------------------------------------------------------------------------
def checkNodes():
	for node, major in idsToMajors.items():
		if major == 'Unknown':
			print "id: ", node
			if node in idsToNames:
				print "     name: ", idsToNames[node]



def plotCommunities():
	detectCommunity(undir_w_graph, LOUVAIN, SPRING_LAYOUT)
	detectCommunity(undir_w_graph, LOUVAIN, KK_LAYOUT)
	detectCommunity(undir_w_graph, GREEDY, SPRING_LAYOUT)
	detectCommunity(undir_w_graph, GREEDY, KK_LAYOUT)




if __name__ == '__main__':
	cleanUpMajors(undir_w_graph)

	majorz = list(set(idsToMajors.values()))
	# coloring
	major_to_cname = {major:cname for major, cname in zip(majorz,cnames.keys()[:len(majorz)])}
	major_to_color_d = {major:cnames[major_to_cname[major]] for i, major in enumerate(majorz)}

	COLORS = cnames.values()

	# plotNodeColorMap(undir_unw_graph)
	# plotAntigraph(undir_unw_graph)
	# plotMajors(dir_w_graph)
	# plotGrad(undir_unw_graph)
	# simplePlot(dir_w_graph)

	plotByNamePopularity(undir_w_graph)
	# ------COMMUNITY DETECTION---------------
	# plotUndergradsByMajor(undir_w_graph)
	# plotCommunities()

	# clustering coefficients
	# top_ccs = [(major,getClusteringCoefficientFor(undir_w_graph.copy(), major, majorz)) for major in majorz]

	# # don't work if run all together b/c modify graph or somethings!!! 
	# plotCommunitiesByLouvainForChosenMajor(undir_w_graph, 'Program in Human Biology')
	# plotCommunitiesByLouvainForChosenMajor(undir_w_graph, 'Economics')
	# plotCommunitiesByLouvainForChosenMajor(undir_w_graph, 'Computer Science')

	# getCommunitiesByModularity(undir_w_graph)

