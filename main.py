# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import networkx as nx
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import numpy as np
from sklearn.metrics import adjusted_rand_score


# Încărcăm graful din fișierul GML
def load_graph(gml_path):
    G = nx.Graph()  # Inițializăm un graf simplu
    current_node = {}
    node_id = None
    edge_source = None
    edge_target = None
    reading_edge = False

    with open(gml_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()

            if stripped_line.startswith('node'):
                reading_edge = False
                current_node = {}
            elif stripped_line.startswith('edge'):
                reading_edge = True
                edge_source = None
                edge_target = None
            elif stripped_line.startswith('['):
                continue
            elif stripped_line.startswith(']'):
                if not reading_edge and node_id is not None:
                    G.add_node(node_id, **current_node)
                    node_id = None
                if reading_edge and edge_source is not None and edge_target is not None:
                    if not G.has_edge(edge_source, edge_target):
                        G.add_edge(edge_source, edge_target)
            else:
                if ' ' in stripped_line:
                    key, value = stripped_line.split(' ', 1)
                    value = value.strip('"')
                    if reading_edge:
                        if key == 'source':
                            edge_source = int(value)
                        elif key == 'target':
                            edge_target = int(value)
                    else:
                        if key == 'id':
                            node_id = int(value)
                        else:
                            current_node[key] = value

    return G

def load_graph_from_txt(txt_path):
    G = nx.read_edgelist(txt_path, create_using=nx.Graph())
    return G

# Calea către fișierul GML din setul de date 'football'
gml_path = 'D:/Inteligenta Artificiala/LAB10/real/football/football.gml'
gml_path2 = 'D:/Inteligenta Artificiala/LAB10/real/dolphins/dolphins.gml'
gml_path3 = 'D:/Inteligenta Artificiala/LAB10/real/karate/karate.gml'
gml_path4 = 'D:/Inteligenta Artificiala/LAB10/real/krebs/krebs.gml'
personal_data = 'D:/Inteligenta Artificiala/LAB10/com-amazon.ungraph.txt'

football_graph = load_graph(gml_path)
#football_graph = load_graph_from_txt(personal_data)

# Afișăm informații generale despre graf
print("Informații despre graf:")
print(f"Numărul de noduri: {football_graph.number_of_nodes()}")
print(f"Numărul de muchii: {football_graph.number_of_edges()}")

# Afișăm primele 5 noduri pentru a vedea detaliile
print("\nPrimele 5 noduri:")
for node, data in list(football_graph.nodes(data=True))[:5]:
    print(node, data)

#Preprocesarea datelor

# Încărcăm etichetele de clasă din fișierul de text
def load_class_labels(txt_path):
    return pd.read_csv(txt_path, header=None, names=['label'])

# Generare artificială de etichete pentru noduri
def generate_artificial_labels(graph, num_communities=3):
    nodes = list(graph.nodes())
    labels = np.random.randint(0, num_communities, size=len(nodes))
    label_df = pd.DataFrame(labels, index=nodes, columns=['label'])
    return label_df

# Calea către fișierul de etichete
labels_path = 'D:/Inteligenta Artificiala/LAB10/real/football/classLabelfootball.txt'
labels_path2 = 'D:/Inteligenta Artificiala/LAB10/real/dolphins/classLabeldolphins.txt'
labels_path3 = 'D:/Inteligenta Artificiala/LAB10/real/karate/classLabelkarate.txt'
labels_path4 = 'D:/Inteligenta Artificiala/LAB10/real/krebs/classLabelkrebs.txt'

class_labels = load_class_labels(labels_path)
#class_labels = generate_artificial_labels(football_graph)

#print(list(football_graph.nodes()))

# Asociem etichetele cu nodurile din graf
def assign_labels_to_graph(graph, labels):
    for idx, label in enumerate(labels['label']):
        graph.nodes[idx]['community'] = label

    #Pentru gml3
    #for node in graph.nodes():  # iterează prin noduri direct
        #if node in labels.index:  # verifică dacă nodul este în index
            #graph.nodes[node]['community'] = labels.loc[node, 'label']
        #else:
            #print(f"Warning: No label for node {node}")

    #Pentru setul de date de pe net
    #for node in graph.nodes():
        #graph.nodes[node]['community'] = labels.loc[node, 'label']


assign_labels_to_graph(football_graph, class_labels)

# Verificăm etichetarea nodurilor
print("\nEtichetele nodurilor atribuite:")
for node, data in list(football_graph.nodes(data=True))[:5]:
    print(node, data)

#Functie de fitness diferita
def community_density(graph, communities):
    densities = []
    for community in communities:
        subgraph = graph.subgraph(community)
        if len(community) > 1:
            max_edges = len(community) * (len(community) - 1) / 2
            actual_edges = subgraph.number_of_edges()
            density = actual_edges / max_edges
        else:
            density = 0  # Pentru comunitati cu un singur nod, densitatea este 0
        densities.append(density)

    if not densities:
        return 0

    return sum(densities) / len(densities)  # Media densitatilor

def conductance(graph, communities):
    conductance_values = []
    for community in communities:
        if len(community) > 0:
            subgraph = graph.subgraph(community)
            internal_edges = subgraph.number_of_edges()
            total_community_edges = sum(deg for node, deg in graph.degree(community))
            outgoing_edges = total_community_edges - 2 * internal_edges
            total_edges = graph.number_of_edges()

            if total_community_edges == 0:  # Evită împărțirea la zero
                continue

            community_conductance = outgoing_edges / min(total_community_edges, 2 * total_edges - total_community_edges)
            conductance_values.append(community_conductance)

    if not conductance_values:
        return 1  # Returnează o valoare mare dacă nu sunt comunități

    return sum(conductance_values) / len(conductance_values)  # Media conductance-urilor

#Algoritmul

# Funcția de evaluare
def evaluate(individual, graph):
    community_dict = {}
    for node, community in zip(graph.nodes(), individual):
        if community not in community_dict:
            community_dict[community] = set()
        community_dict[community].add(node)

    communities = list(community_dict.values())

    # Alege funcția de fitness dorită
    # fitness_score = conductance(graph, communities)
    fitness_score = conductance(graph, communities)
    return (fitness_score,)

# Toolset-ul pentru DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Generator de indivizi: Atribuim fiecărui nod o comunitate aleatorie
toolbox.register("attr_comm", random.randint, 0, len(set(class_labels['label'])) - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_comm, n=football_graph.number_of_nodes())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operatori
toolbox.register("evaluate", evaluate, graph=football_graph)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(set(class_labels['label'])) - 1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Algoritmul evolutiv
def main():
    random.seed(42)
    pop = toolbox.population(n=100)  # 100 de indivizi în populație
    hof = tools.HallOfFame(1)  # Păstrăm cel mai bun individ

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof,
                                   verbose=True)

    best_individual = hof[0]
    best_communities = extract_communities(best_individual)

    # Calcularea numărului de comunități unice
    num_communities = len(set(best_communities))

    # Afișarea numărului de comunități și a apartenenței la comunități pentru fiecare nod
    print("Numărul de comunități identificate:", num_communities)
    print("Aparțenența la comunitate pentru fiecare nod:")
    for node, community in zip(football_graph.nodes(), best_communities):
        print(f"Nodul {node} aparține comunității {community}")

    return pop, stats, hof

#4 Testarea si evaluarea

# Functie pentru conversia soluției evolutive în etichete de comunitate
def extract_communities(individual):
    community_map = {node: ind for node, ind in zip(football_graph.nodes(), individual)}
    return [community_map[node] for node in football_graph.nodes()]


# Evaluarea acurateței prin compararea cu etichetele reale
def evaluate_accuracy(community_labels, real_labels):
    return adjusted_rand_score(real_labels, community_labels)


# Testarea și evaluarea algoritmului
def test_algorithm():
    # Rulăm algoritmul evolutiv pentru a obține cel mai bun individ
    _, _, hof = main()
    best_individual = hof[0]

    # Extragerea comunităților detectate
    detected_communities = extract_communities(best_individual)

    # Calcularea modularității pentru cel mai bun individ
    modularity_score = best_individual.fitness.values[0]

    # Calcularea acurateței comparativ cu etichetele reale
    accuracy_score = evaluate_accuracy(detected_communities, class_labels['label'].tolist())

    return modularity_score, accuracy_score

# Rulăm algoritmul
if __name__ == "__main__":
    final_population, statistics, hall_of_fame = main()
    print("Cel mai bun individ are modularitatea:", hall_of_fame[0].fitness.values)

    # Rulăm testarea
    modularity, accuracy = test_algorithm()
    print(f"Modularitatea soluției finale: {modularity}")
    print(f"Acuratețea soluției finale: {accuracy}")
