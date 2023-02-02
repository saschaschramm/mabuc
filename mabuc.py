import networkx as nx
from matplotlib import pyplot as plt


def graph_causal_training():
    graph = nx.grid_2d_graph(2,7, create_using=nx.DiGraph)
    plt.figure(figsize=(7,2))
    plt.axis('off')
    pos = {(x,y):(y,-x) for x,y in graph.nodes()}
    labels = {(0,1): "U0", (1,0): "X0", (1,2): "Y0", (1,4): "X1", (1,6): "Y1", (0,5): "U1"}

    edge_label = "P(Y|do(X),U)\n\n"
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={((1,0), (1,2)): edge_label})

    edge_label = "P(Y|do(X),U)\n\n"
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={((1,4), (1,6)): edge_label})

    edge_list = [((1,0), (1,2)), ((0,1), (1,2)), ((1,0), (1,4)), ((1,4), (1,6)), ((0,5), (1,6))]
    nx.draw_networkx_edges(graph, pos, edgelist=edge_list)

    nx.draw_networkx_nodes(graph, pos, nodelist=[(0,1), (0,5)], node_size=200, node_shape='s', alpha=0.4)
    nx.draw_networkx_nodes(graph, pos, node_color='lightgray', node_size=200, node_shape='s', nodelist=[(1,0), (1,2), (1,4), (1,6)])
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)

    # save graph to file
    #plt.savefig("causal_training.png", bbox_inches='tight', pad_inches=0)

    # save graph to file with high dpi
    plt.savefig("causal_training.png", bbox_inches='tight', pad_inches=0, dpi=300)


    #plt.text(pos[(1,0)][0], pos[(1,0)][1]-0.2, f"a1* = argmax E1[O|do(A=a)]", fontsize=10, horizontalalignment='center', verticalalignment='center')
    #plt.text(pos[(1,4)][0], pos[(1,4)][1]-0.2, f"a2* = argmax E2[O|do(A=a)]", fontsize=10, horizontalalignment='center', verticalalignment='center')

graph_causal_training()