from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
import time 
import sys
from urlparse import urlparse
import os
import json

g = nx.read_edgelist("graphs/unweighted_users_graph.gph")
idsToMajors = {}
idsToGrad = {}

files = os.listdir("users_htmls")
for file in files: 
	with open("users_htmls/" + file) as fp:
		soup = BeautifulSoup(fp, "lxml")
		nodeId = file[:-5]
		infos = soup.find_all('div', class_="Affiliation")
		majors = []
		for info in infos: 
			if info is None: 
				idsToMajors[nodeId] = None
			else:
				personalInfo = info.find_all('dd', class_ = "public")
				if len(personalInfo) > 1:
					major, grad_status = personalInfo[0].string,  personalInfo[1].string, 
					majors.append(major)
					#print "grad: ", grad_status
					idsToGrad[nodeId] = grad_status
				else:
					majors.append(personalInfo[0].string)
		
		#print "nodeId: ", nodeId, "Major: ", majors
		idsToMajors[nodeId] = majors

		
jsonFile = json.dumps(idsToMajors)
f = open("attributes/majorAttributes.json","w")
f.write(jsonFile)
f.close()

jsonFile = json.dumps(idsToGrad)
f = open("attributes/gradAttributes.json","w")
f.write(jsonFile)
f.close()

