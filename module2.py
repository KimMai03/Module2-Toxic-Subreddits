import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 

#Extracting a subset of negative sentiments
df = pd.read_csv("soc-redditHyperlinks-body.tsv", sep="\t")

toxic_subreddits = df[df["LINK_SENTIMENT"] == -1]
print(f"Toxic subreddits: {len(toxic_subreddits)}")

#Graph of Network
G = nx.from_pandas_edgelist (
    toxic_subreddits,
    source="SOURCE_SUBREDDIT",
    target="TARGET_SUBREDDIT",
    create_using=nx.DiGraph()
)

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

#Find the Top 5 toxic subreddits
out_degrees = dict(G.out_degree())
top_5 = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
top_5_nodes = [node[0] for node in top_5]

print(f"Top 5 Toxic subreddits: {top_5_nodes}")

#Create a graph with only edges originating from these Top 5
edges_to_draw = [(u, v) for u, v in G.edges() if u in top_5_nodes]
Top5_Graph = nx.DiGraph(edges_to_draw)

#Top 5 toxic subreddits graph
nx.write_graphml(Top5_Graph, "top_5_toxic.graphml")

for rank, (subreddit, toxic_links) in enumerate(top_5, 1):
    print(f"{rank}. {subreddit}: {toxic_links} negative outbound links")

#Top 5 by Degree Centrality:
degree_centrality = nx.degree_centrality(G)
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 by Degree Centrality:")
for rank, top_degree in enumerate(top_degree, 1):
    print(f"{rank}. {top_degree}")

#Top 5 by Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G)
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 by Betweenness Centrality:")
for rank, top_betweenness in enumerate(top_betweenness, 1):
    print(f"{rank}. {top_betweenness}")

#Top 5 by Eigenvector Centrality:
eigenvector_centrality = nx.eigenvector_centrality(G)
top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 by Eigenvector Centrality:")
for rank, top_eigenvector in enumerate(top_eigenvector, 1):
    print(f"{rank}. {top_eigenvector}")