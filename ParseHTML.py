'''
CS224W Fall 2018 Final Project
Yardena Hirsch and Shelby Marcus
Data Scraping
--------------------------------
Given the filename of an html file of the Facebook group Stanford Memes, 
this program parses the html and stores the links between users in a graph.
The graph is saved as both a weighted and unweighted edge list in the "graphs" 
folder.  
'''

from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
import time 
import sys
from urlparse import urlparse
import json
import os
import datetime

# CONSTANTS
NEW_FILENAME = "data/SMFET_1245pm.htm"
OLD_FILENAME = "data/745pm.htm"

# Fixes an error on Shelby's computer
reload(sys)
sys.setdefaultencoding('utf-8')


def extractCommentNewPage(comment, idToName):
	posterBlock = comment.find("a", class_ = "_6qw4", href = True)
	posterName = posterBlock.string
	posterId = getUserId(posterBlock)

	taggedUserNames = []
	taggedUserIds = []
	taggedUsersBlock= comment.find('span', class_ = "_3l3x")
	
	if taggedUsersBlock is not None:
		taggedUsersBlock = taggedUsersBlock.find_all("a", href = True)
		for taggedUser in taggedUsersBlock:
			taggedUserName = taggedUser.string
			if posterName is not None and taggedUserName is not None:
				# if not all(ord(c) < 128 for c in posterName) or not all(ord(c) < 128 for c in taggedUser): continue
				posterName = posterName.encode('utf-8')
				taggedUserName = taggedUserName.encode('utf-8')
				taggedUserId = getUserId(taggedUser)
				taggedUserNames.append(taggedUserName)
				taggedUserIds.append(taggedUserId)
				idToName[posterId] = posterName
				idToName[taggedUserId] = taggedUserName

	return posterName, posterId, taggedUserNames, taggedUserIds


def scrapeHTMLNewPage(filename, verbose, idToName):
	files = os.listdir(filename)
	post_d = {}
	for file in files: 
		edges = []
		with open(filename + "/" + file) as fp:
			soup = BeautifulSoup(fp, "lxml")
		if verbose:
			print "---START OF POST--------------------------"
		originalPosterName = soup.find('span', class_ ="fwb fcg")
		if originalPosterName is not None: 
			originalPosterName = originalPosterName.a.string
		post = soup.find('abbr', class_='_5ptz')
		if post is not None:
			post_date = post.find('span').string
			# print "post date: ", post_date
		else:
			post_date = None
		msg = soup.find('div', class_ = '_5pbx userContent _3576')
		if msg is not None: 
			message = msg.p.string
		else:
			message = None

		if verbose: 
			print "---Poster Name: ", originalPosterName
			print "---Post Date: ", post_date
		# Find all comments 
		comments = soup.find_all("div", class_ = "_72vr")
		for comment in comments:
			posterName, posterId, taggedUserNames, taggedUserIds =  extractCommentNewPage(comment, idToName)
			edges.append((posterId, taggedUserIds))
			if verbose:
				print "-------------Commenter: ", posterName, " Tagged Users: ", taggedUserNames
		if verbose:
			print "---END OF POST--------------------------"
		post_d[(post_date, message, originalPosterName)] = edges

	return post_d


'''
Function: getUserId
-------------------
Given a html block containing the href of the user's profile,
get the users ID. The user ID is either the user's chosen ID or a
numerical ID
'''
def getUserId( user):
	userHref = user["href"]
	# If the user has chosen an ID
	userId = list(urlparse(userHref).path)
	del userId[0]
	userId = ''.join(userId)
	# If the user has not chosen an ID
	if userId == "profile.php":
		userId = urlparse(userHref).query
		startIndex = userId.find("=") + 1
		endIndex = userId.find("&")
		userId = userId[startIndex:endIndex]
	return userId


'''
Function: extractComment
------------------------
Given a directed, weighted graph and a comment, this function adds 
an edge between user A and user B if user A tags user B in a comment.
This function returns the commenter's name and a list of the tagged users 
'''
def extractComment(comment, idToName):
	posterBlock = comment.find('a', class_ = "UFICommentActorName", href = True)
	posterName = posterBlock.string
	posterId =  getUserId(posterBlock)
	taggedUsersBlock= comment.find_all('a', class_ = "profileLink", href = True)
	taggedUserNames = []
	taggedUserIds = []
	for taggedUser in taggedUsersBlock:
		taggedUserName = taggedUser.string
		taggedUserId =  getUserId(taggedUser)
		if posterName is not None and taggedUserName is not None:
			if not all(ord(c) < 128 for c in posterName) or not all(ord(c) < 128 for c in taggedUserName): continue
			posterName = posterName.encode('utf-8')
			taggedUserName = taggedUserName.encode('utf-8')
			idToName[posterId] = posterName
			idToName[taggedUserId] = taggedUserName
			taggedUserNames.append(taggedUserName)
			taggedUserIds.append(taggedUserId)
	return posterId, taggedUserIds


'''
Function: extractHTML
---------------------
Given the name of an HTML file, this function returns a weighted, directed graph 
where Facebook users are nodes and there is an edge between user A and user B 
if user A tags user B in a post
'''
def extractHTML(filename, verbose, idToName):
	# g = nx.DiGraph()
	with open(filename) as fp:
	    soup = BeautifulSoup(fp, "lxml")
	# Get a list of posts 
	posts = soup.find_all('div', class_="_5pcr userContentWrapper")
	# Iterate over posts and extract edges between users 

	# key is (post_date, message, authors name), val is edges of comments on post [(commenter, [tagged people])]
	post_d = {}
	for post in posts:
		edges = []
		if verbose:
			print "---START OF POST--------------------------"
		# Get the original poster name 
		originalPosterName = post.find('span', class_ ="fwb fcg")
		if originalPosterName is not None: 
			originalPosterName = originalPosterName.a.string
		post_date = post.find('abbr', class_='_5ptz').find('span').string
		# print "post_date: ", post_date
		if verbose: 
			print "---Poster Name: ", originalPosterName
		# Get all comments from post and add edges using extractComment
		if post is not None:
			message_obj = post.find('div', class_ = "_5pbx userContent _3576")
			if message_obj: 
				message = message_obj.string
			if verbose:
				print "Message: ", message
			# The first comment does not have the same tag as others 
			firstComment = post.find('div', class_ = "UFIRow _48pi _4204 UFIComment _4oep")
			if firstComment is not None:
				posterId, taggedUserIds =  extractComment(firstComment, idToName)
				edges.append((posterId, taggedUserIds))
				if verbose:
					print "-------------Commenter: ", posterName, " Tagged Users: ", taggedUserNames
			# Add edges for all other comments 
			comments = post.find_all('div', class_ = "UFIRow UFIComment _4oep")
			for comment in comments:	
				posterId, taggedUserIds =  extractComment(comment, idToName)
				edges.append((posterId, taggedUserIds))
				if verbose:
					print "-------------Commenter: ", posterName, " Tagged Users: ", taggedUserNames
		
		post_d[(post_date, message, originalPosterName)] = edges

		if verbose:
			print "---END OF POST--------------------------"
	return post_d


def writeUnweightedGph(g, filename):
    graphFilename = "graphs/unweighted_" + filename
    with open(graphFilename, 'w') as f:
        for edge in g.edges():
            f.write("{} {}\n".format(edge[0], edge[1]))

def writeWeightedGph(g, filename):
    graphFilename = "graphs/weighted_" + filename
    with open(graphFilename, 'w') as f:
        for edge in g.edges():
        	weight = g[edge[0]][edge[1]]['weight']
        	f.write("{} {} {}\n".format(edge[0], edge[1], weight))

def createGraphFromDict(d):
	g = nx.DiGraph()
	for key, edges in d.items():
		post_date, msg, authors_name = key
		for edge in edges: 
			commenter, tagged_ppl = edge
			if commenter is None: continue
			for tagged_person in tagged_ppl:
				if tagged_person is None: continue
				if g.has_edge(commenter, tagged_person):
					g[commenter][tagged_person]['weight'] += 1.0
				else:
					g.add_edge(commenter, tagged_person, weight=1.0)
	return g

def writeDictionary(dictionary, name):
	jsonFile = json.dumps(dictionary)
	f = open("attributes/" + name + ".json","w")
	f.write(jsonFile)
	f.close()

def inTestData(post_date):
	test_months = set(['Yesterday', 'November','December'])
	if post_date is None: return False
	if post_date.split() is None: return False
	if len(post_date.split()) > 1: 
		month = post_date.split()[0]
		day = post_date.split()[1]
		if day is None: return True
		day = day.split(",")[0]
		if month == 'November':
			if int(day) >= 15: return True
			else: return False
		elif month in test_months: return True
		else: return False 

	else: 
		return True

def scrapeData():
	# Extract data and create users to users graph 
	idToName = {}
	
	old_post_d =  extractHTML('data/SMFET_1245pm.htm', False, idToName)

	new_post_d = scrapeHTMLNewPage('data/html_manual', False, idToName)

	# compile 
	compiled_post_d = {}

	# key of post_d is (post_date, message, authors name), val is edges of comments on post [(commenter, [tagged people])]
	for post_d in [old_post_d, new_post_d]:
		for key, edges in post_d.items():
			post_date, msg, authors_name = key
			if key in compiled_post_d:
				if len(compiled_post_d[key]) < len(edges):
					compiled_post_d[key] = edges
			else:
				compiled_post_d[key] = edges

	# add nodes/edges to graph from compiled dictionary
	compiled_g = createGraphFromDict(compiled_post_d)

	# print
	print "num nodes: ", compiled_g.number_of_nodes()
	print "num edges: ", compiled_g.number_of_edges()

	# posts before Nov 15
	train_d = {key:edges for key, edges in compiled_post_d.items() if not inTestData(key[0])}
	train_g = createGraphFromDict(train_d)

	# posts after Nov 15
	test_d = {key:edges for key, edges in compiled_post_d.items() if inTestData(key[0])}
	test_g = createGraphFromDict(test_d)

	writeWeightedGph(train_g, "train_graph.gph")
	writeWeightedGph(test_g, "test_graph.gph")
	writeUnweightedGph(train_g, "train_graph.gph")
	writeUnweightedGph(test_g, "test_graph.gph")

	writeWeightedGph(compiled_g, "users_graph.gph")
	writeUnweightedGph(compiled_g, "users_graph.gph")

	writeDictionary(idToName, "nameAttributes")

if __name__ == "__main__":
	scrapeData()
