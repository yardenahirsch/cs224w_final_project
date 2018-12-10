from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
import time 
import sys
from urlparse import urlparse
import os
import json
import time
import random
import os.path 

g = nx.read_edgelist("graphs/unweighted_users_graph.gph")
with open("attributes/nameAttributes.json") as json_file:  
	idsToNames = json.load(json_file)

for node in g.nodes():
# underscores between names and dash between NAMES and FACEBOOK ID
	output = "users_htmls/"+node+".html"

	if os.path.isfile(output): continue

	name = idsToNames[node]
	# print "name: ", name
	first_name, last_name = name.split(" ")[0].replace("'","").replace("-",""), name.split(" ")[-1].replace("'","").replace("-","")
	# print "node: ", node, ", name: ", name
	if not all(ord(c) < 128 for c in name): continue

	url = "https://stanfordwho.stanford.edu/SWApp/lookup?search="+last_name+",+"+first_name
	command = "python -m wget -o " + output + " " + url
	os.system(command)
	# time.sleep(random.randint(0,2))
