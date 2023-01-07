import copy
from cProfile import label
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats
from scipy.special import factorial

######################### Matplotlib features #################################
plt.rcParams["figure.figsize"] = [10,10]

######################### Read data #################################

def df_to_dictionary(csv, model, pvalue=1):
    df = csv.loc[csv[ 'P-value' ]<pvalue]
    df.reset_index(inplace=True, drop=True)
    query_dic = {}
    array_dic = {}
    for i in range(len(df)):
        if model == 'mul':
            interaction = df.loc[i, 'Double mutant fitness']-df.loc[i, 'Query single mutant fitness (SMF)']*df.loc[i, 'Array SMF']
        if model == 'add':
            interaction = df.loc[i, 'Double mutant fitness']-(df.loc[i, 'Query single mutant fitness (SMF)']+df.loc[i, 'Array SMF']-1)
        if not (np.isnan(interaction) or np.isinf(interaction)):
            query = df.loc[i, 'Query allele name']
            array = df.loc[i, 'Array allele name']
            if query not in query_dic.keys():
                query_dic[query]={}
            query_dic[query][array]=interaction
            if array not in array_dic.keys():
                array_dic[array]={}
            array_dic[array][query]=interaction
    return query_dic, array_dic

def json_load(file_path):
    with open(file_path) as file:
        arch = json.load(file)
    return arch
def list_load(file_path):
    with open(file_path) as file:
        arch = file.readlines()
    return [line.strip() for line in arch]

def load_dataframe(dataframe):
    ct = pd.read_table(dataframe, low_memory=False)
    ct = ct.drop(index=0)
    ct = ct.drop(columns={'Unnamed: 1'})
    ct.set_index('Unnamed: 0', inplace=True)
    ct = ct.applymap(float)
    return ct

def extract_dictionary(dataframe):
    dic = {}
    for i in range(len(dataframe)):
        if dataframe.index[i] not in dic.keys():
            dic[dataframe.index[i]] = {}
        for j in range(i+1, len(dataframe)):
            value = dataframe.iloc[i,j]
            dic[dataframe.index[i]][dataframe.index[j]] = value
            if dataframe.index[j] not in dic.keys():
                dic[dataframe.index[j]] = {}
            dic[dataframe.index[j]][dataframe.index[i]] = value
    return dic

######################### PCC Calculations #################################

def common_interactors(gene1, gene2, GIN):
    return list(set(GIN[gene1].keys()).intersection(set(GIN[gene2].keys())))

def calculate_pcc(epsilon_dictionary):
    pcc = {}
    genes = sorted(set([key for key in epsilon_dictionary]))
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            if genes[i] not in pcc.keys():
                pcc[genes[i]] = {}
            if genes[j] not in pcc.keys():
                pcc[genes[j]] = {}
            interactors = common_interactors(genes[i], genes[j], epsilon_dictionary)
            x_list = [epsilon_dictionary[genes[i]][k] for k in interactors]
            y_list = [epsilon_dictionary[genes[j]][k] for k in interactors]
            try:
                value = scipy.stats.pearsonr(x_list, y_list)
                if value[1]<0.05:
                    pcc[genes[i]][genes[j]] = value[0]
                    pcc[genes[j]][genes[i]] = pcc[genes[i]][genes[j]]
            except:
                pass
    return pcc

def join_pcc(dic1, dic2):
    pcc = copy.deepcopy(dic1)
    common = set(dic1.keys()).intersection(set(dic2.keys()))
    unique2 = set(dic2.keys()).difference(set(dic1.keys()))
    for key in unique2:
        pcc[key] = dic2[key]
    for key in common:
        commonkeys = set(dic1[key].keys()).intersection(dic2[key].keys())
        for ck in commonkeys:
            pcc[key][ck] = (pcc[key][ck] + dic2[key][ck])/2
        diffkeys = set(dic2[key].keys()).difference(dic1[key].keys())
        for df in diffkeys:
            pcc[key][df] = dic2[key][df]
    return pcc

######################### Save data #################################

def json_save(file, file_path):
    with open(file_path, 'w', encoding='utf8') as arch:
        json.dump(file, arch, ensure_ascii=False)

def list_save(file, file_path):
    with open(file_path, 'w') as arch:
        for element in file:
            arch.writelines(f'{element}\n')

def to_cytoscape(dic, path):
    panel = {'source gene':[],
             'target gene':[],
             'pcc': [],
             '1-pcc': [],
             'weight': [],
             'ORF': []
             }
    gene_ORF = json_load('data/gene_ORF.json')
    for key in dic.keys():
        for gene in dic[key].keys():
            panel['source gene'].append(key)
            panel['target gene'].append(gene)
            panel['pcc'].append(abs(dic[key][gene]))
            try:
                panel['ORF'].append(gene_ORF[key])
            except:
                panel['ORF'].append('NotFound')
            try:
                panel['1-pcc'].append(1-abs(dic[key][gene]))
            except:
                panel['1-pcc'].append(float('nan'))
            panel['weight'].append(1)
    df = pd.DataFrame(panel)
    df.to_csv(path)

######################### Filter data #################################

def select_interactions(dic, value):
    dic = copy.deepcopy(dic)
    for gene in dic:
        try:
            dic[gene] = dict(filter(lambda item: abs(item[1])>value, dic[gene].items()))
        except:
            pass
    dic = dict(filter(lambda item: len(item[1])>0, dic.items()))
    return dic

def select_by_degree(network, value):
    degree_dic = {}
    for k in network.keys():
        degree_dic[k] = len(network[k].keys())
    filtered = list(map(lambda x: x[0], filter(lambda x: x[1]>=value, degree_dic.items())))
    return filtered

def select_extreme_interactions(network, percentage):
    net = copy.deepcopy(network)
    net = reduce_dictionary(net)
    interactions = []
    for k in net.keys():
        for k1 in net[k].keys():
            interactions.append((k,k1,net[k][k1]))
    interactions = sorted(interactions, key=lambda x: x[2])
    cutoff = round(len(interactions)*percentage/100)
    positive = interactions[-cutoff:]
    negative = interactions[:cutoff]
    new = {}
    for tup in positive:
        if tup[0] not in new.keys():
            new[tup[0]] = {}
        if tup[1] not in new.keys():
            new[tup[1]] = {}
        new[tup[0]][tup[1]] = tup[2]
        new[tup[1]][tup[0]] = new[tup[0]][tup[1]]
    for tup in negative:
        if tup[0] not in new.keys():
            new[tup[0]] = {}
        if tup[1] not in new.keys():
            new[tup[1]] = {}
        new[tup[0]][tup[1]] = tup[2]
        new[tup[1]][tup[0]] = new[tup[0]][tup[1]]
    return new

def equal_interactions(net1, net2):
    """ Selects in net2 the same number of interactions that exist in net1:
    the same number of both positive and negative interactions that net1. Returns net2 """
    net1 = copy.deepcopy(net1)
    net2 = copy.deepcopy(net2)
    net1 = reduce_dictionary(net1)
    net2 = reduce_dictionary(net2)
    interactions = []
    for k in net1.keys():
        for k1 in net1[k].keys():
            interactions.append((k,k1,net1[k][k1]))
    interactions = sorted(interactions, key=lambda x: x[2])
    #############################
    neg_int = len(list(filter(lambda x: x[2]<0, interactions)))
    pos_int = len(list(filter(lambda x: x[2]>0, interactions)))
    #############################
    interactions2 = []
    for k in net2.keys():
        for k1 in net2[k].keys():
            interactions2.append((k,k1,net2[k][k1]))
    interactions2 = sorted(interactions2, key=lambda x: x[2])
    #############################
    new_neg = interactions2[:neg_int]
    new_pos = interactions2[-pos_int:]
    new_int = []
    new_int.extend(new_neg)
    new_int.extend(new_pos)
    #############################
    new = {}
    for tup in new_int:
        if tup[0] not in new.keys():
            new[tup[0]] = {}
        if tup[1] not in new.keys():
            new[tup[1]] = {}
        new[tup[0]][tup[1]] = tup[2]
        new[tup[1]][tup[0]] = new[tup[0]][tup[1]]
    return new

def check_mean_std(net):
    net = copy.deepcopy(net)
    values = []
    for k in net.keys():
        values.extend(net[k].values())
    mu, sigma =  scipy.stats.norm.fit(values)
    return mu, sigma

def normalize_dictionary(net):
    net = copy.deepcopy(net)
    mu, sigma =  check_mean_std(net)
    for gene in net.keys():
        for gene1 in net[gene].keys():
                net[gene][gene1] = net[gene][gene1]-mu
    mu, sigma =  check_mean_std(net)
    return net, mu, sigma

def create_network(gene_list, network):
    """
    Takes a list of genes and filters a network in such way that only genes in the list are found in the network
    """
    network = copy.deepcopy(network)
    network = dict(filter(lambda x: x[0] in gene_list, network.items()))
    network = {k:dict(filter(lambda x: x[0] in gene_list, network[k].items())) for k in network.keys()}
    return network
######################### Topology #################################

def is_essential(data, splitted=False):
    essential_list = list_load('data/essential_genes.txt')
    nonessential_list = list_load('data/nonessential_genes.txt')
    if type(data) == str:
        if data in essential_list:
            return True
        elif data in nonessential_list:
            return False
        else:
            return 'Gene Not Found'
    elif type(data) == list:
        data = set(map(lambda x: x.lower(), data))
        essential_list = set(map(lambda x: x.split('-')[0].lower(), essential_list))
        nonessential_list = set(map(lambda x: x.split('-')[0].lower(), nonessential_list))
        if data.issubset(essential_list):
            return 'All genes are essential'
        elif data.issubset(nonessential_list):
            return 'All genes are nonessential'
        elif not data.difference(essential_list.union(nonessential_list)):
            return 'Genes essential and nonessential'
        else:
            not_found = data.difference(essential_list.union(nonessential_list))
            return f'Some genes on data not found on datasets {not_found}'

def gene_degree(dic):
    dic = copy.deepcopy(dic)
    for gene in dic.keys():
        dic[gene] = len(dic[gene].keys())
    return dic

def clus_coeff(dic):
    G = nx.DiGraph(dic)
    G = G.to_undirected()
    return nx.clustering(G)

def centrality_closeness(dic):
    G = nx.DiGraph(dic)
    G = G.to_undirected()
    return nx.closeness_centrality(G)

def centrality_betweenness(dic, normalized = True, endpoints = False):
    # For Graphs with a large number of nodes, the value of betweenness centrality is very high.
    # So, we can normalize the value by dividing with number of node pairs (excluding the current node).
    G = nx.DiGraph(dic)
    G = G.to_undirected()
    bet_closeness = nx.betweenness_centrality(G, normalized = normalized, endpoints = endpoints)
    return bet_closeness

def degree_freq(net):
    if type(net) == nx.classes.graph.Graph:
        deg = dict(net.degree())
    else:
        G = nx.DiGraph(net)
        G.to_undirected()
        deg = {k:int(v/2) for k,v in G.degree()}
    dic = {}
    for k in deg.keys():
        if deg[k] not in dic.keys():
            dic[deg[k]] = 1
        else:
            dic[deg[k]] += 1
    freq = np.array([v for k,v in dic.items()])
    degree = np.array([k for k,v in dic.items()])
    z = list(zip(degree, freq))
    z.sort(key=lambda x: x[0])
    degree = np.array(list(map(lambda x: x[0], z)))
    freq = np.array(list(map(lambda x: x[1], z)))
    return degree, freq

def construct_power_law(dic, title='title', save=False, name='default.png', show_fit=False):
    counter = {}
    for key in dic.keys():
        if dic[key] in counter.keys(): counter[dic[key]]+=1
        else: counter[dic[key]]=1
    tup = [(k,v) for k,v in counter.items()]
    tup.sort(key=lambda x: x[0])
    degree = np.array(list(map(lambda x: x[0], tup)))
    freq = np.array(list(map(lambda x: x[1], tup)))
    #####################################
    def func(x, A, B, c):
        return A * np.exp(-B * x) + c
    popt, pcov = curve_fit(func, degree, freq)
    #####################################
    a = plt.scatter(degree, freq, s=40, alpha=0.8)
    if show_fit:
        b = plt.plot(degree, func(degree, *popt), 'r-', label='Best Fit')
        plt.legend()
    l = plt.xlabel('Grado')
    d  = plt.ylabel('Frecuencia')
    t = plt.title(title)
    plt.grid(True)
    if save:
        plt.savefig(name)
    #### El segundo valor de popt es negativo
    return popt

def generate_power_law(n, m):
    return nx.barabasi_albert_graph(n = n, m = m, seed=10374196)

def generate_random(n, p):
    return nx.erdos_renyi_graph(n, p, seed=10374196)

def fit_poisson(x, deg, freq):
    L = [deg[i] for i in range(len(deg)) for f in range(freq[i])]
    mean = np.mean(L)
    def func(x, A):
        return A*scipy.stats.poisson.pmf(x, mean)
    popt, _ = curve_fit(func, deg, freq)
    return popt[0]*scipy.stats.poisson.pmf(x, mean)

######################### Clustering #################################

# class Cluster():
#     def __init__(self, path, dic):
#         self.path = path
#         self.genes = []
#         with open(path) as file:
#             for i in file.readlines():
#                 self.genes.append(i.replace('\n',''))
#         self.network = {}
#         for gene in self.genes:
#             self.network[gene] = dic[gene]
#             self.network[gene] = dict(filter(lambda x: x[0] in self.genes, self.network[gene].items()))

def obtain_clusters(path, p_value=0.01):
#def obtain_clusters(path, network, p_value=0.01, qual=0.5):
    df = pd.read_csv(path)
    # df = df.loc[(df['P-value']<p_value) & (df['Quality']>qual)]
    df = df.loc[df['P-value']<p_value]
    df.reset_index(inplace=True)
    cluster_tup = []
    for i in range(len(df)):
        cluster_tup.append(set(df.loc[i, 'Members'].split(' ')))
    # Comento estas lineas porque quiero obtener listas de los clusters, no tanto las redes dentro
    # cluster_list = []
    # for cluster in cluster_tup:
    #     nw = dict(filter(lambda x: x[0] in cluster, network.items()))
    #     for key in nw.keys():
    #         nw[key] = dict(filter(lambda x: x[0] in cluster, network[key].items()))
    #     nw = dict(filter(lambda x: len(nw[x[0]])!=0, nw.items()))
    #     nw = reduce_dictionary(nw)
    #     cluster_list.append(nw)
    
    # Introduzco estas lineas para sacar solamente las listas de genes del cluster
    return cluster_tup

def export_clusters(cluster_tup, name):
    if 'Clusters' not in os.listdir():
        os.mkdir('Clusters')
    if name not in os.listdir('Clusters/'):
        os.mkdir('Clusters/'+name)
    for i in range(len(cluster_tup)):
        with open('Clusters/'+name+'/'+str(name)+'_clust'+str(i)+'.txt', 'w') as file:
            for gene in cluster_tup[i]:
                file.writelines(gene.upper().split('-')[0]+'\n')

def reduce_dictionary(dic):
    int_list = [(key, key1) for key in dic.keys() for key1 in dic[key].keys()]
    int_list = [tuple(sorted(tup)) for tup in int_list]
    int_list.sort()
    int_list = list(set(int_list))
    new_dic = {}
    for a,b in int_list:
        if a not in new_dic.keys():
            new_dic[a] = {}
        if b not in new_dic[a].keys():
            try:
                new_dic[a][b] = dic[a][b]
            except:
                new_dic[a][b] = dic[b][a]
    new_dic = dict(filter(lambda x: len(new_dic[x[0]])!=0, new_dic.items()))
    return new_dic

def analyze_cluster(cluster):
    pos = 0
    neg = 0
    for key in cluster.keys():
        for key1 in cluster[key].keys():
            if cluster[key][key1]>0: pos+=1
            else: neg+=1
    try:
        abundance = pos/(pos+neg)*100
        return abundance
    except:
        return -1

######################### Visualize data #################################

def compare_dictionaries(dic1, dic2, xaxis='dic1', yaxis='dic2', c='blue', save=False, name='default.png'):
    list1=[]
    list2=[]
    for key in dic1.keys():
        for key1 in dic1[key]:
            if key in dic2.keys() and key1 in dic2[key].keys():
                list1.append(dic1[key][key1])
                list2.append(dic2[key][key1])
    plt.figure(figsize=(10,10))
    plt.scatter(list1, list2, s=0.2, c=c)
    plt.plot(np.arange(-1,1,0.1), np.arange(-1,1,0.1), c='k', zorder=2, alpha=0.5)
    plt.xlabel(xaxis, fontsize=30)
    plt.ylabel(yaxis, fontsize=30)
    plt.xticks(np.arange(-1,1,0.2), fontsize=20)
    plt.yticks(np.arange(-1,1,0.2), fontsize=20)
    plt.grid(True)
    if save:
        plt.savefig(name)

def histogram_for_values(net, bin_width=0.01, title='Histogram for values', density=0):
    net = copy.deepcopy(net)
    # net = reduce_dictionary(net)
    values = []
    for k in net.keys():
        values.extend(net[k].values())
    ##########################################
    mu, sigma =  scipy.stats.norm.fit(values)
    x = np.arange(-1,1,0.01)
    if density==1:
        plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma), linewidth=3, label='Gaussian')
    ##########################################
    plt.hist(values, bins = np.arange(-1,1, bin_width), edgecolor = 'black',  label='Histogram', density=density)
    plt.grid(axis='y')
    plt.xticks(np.arange(-1,1,0.1))
    plt.xlim(-1,1)
    plt.xlabel('Rango', fontsize=20)
    plt.ylabel('Frecuencia', fontsize=20)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()
    return mu, sigma

def count_genes(net):
    return len(net.keys())

def count_interactions(network):
    c =0
    for k in network.keys():
        c += len(network[k].keys())
    return c

def ontology(df, title='No Title', save=False, path='default.png'):
    plt.style.use('ggplot')
    plt.figure(figsize=(10,8))
    if len(df)>25:
        df.sort_values('p_fdr_bh', inplace=True)
        df = df[:25]    
    # w = plt.scatter(df['ratio_in_study'], df['name'], s=list(map(lambda x: x*4000, df['ratio_in_pop'])), c=df['p_fdr_bh'], cmap='winter_r', edgecolors="black")
    w = plt.scatter(df['ratio_in_study'], df['name'], s=list(map(lambda x: x*10, df['study_count'])), c=df['p_fdr_bh'], cmap='winter_r', edgecolors="black")
    a = plt.xticks(np.arange(0,1.1,0.1))
    plt.xlabel('Razon de genes en estudio')
    plt.ylabel('Terminos GO')
    plt.title(title)
    clb = plt.colorbar()
    plt.clim(0, 0.05)
    clb.set_label('p-valor')
    if save:
        plt.savefig(path, bbox_inches='tight')

######################### Hubs #################################

def find_hubs(network, num_of_hubs):
    """ This method finds the num_of_hubs most connected hubs. It splits the alleles names and takes just the first part in consideration
    """
    dic = {}
    for k in network.keys():
        gene = k.split('-')[0]
        if gene not in dic.keys():
            dic[gene] = {}
        for k1 in network[k].keys():
            gene1 = k1.split('-')[0]
            if gene1 not in dic.keys():
                dic[gene1] = {}
            dic[gene][gene1] = network[k][k1]
            dic[gene1][gene] = dic[gene][gene1]
    counter = {}
    for k in dic.keys():
        counter[k] = len(dic[k].keys())
    dic = [(k,v) for k,v in counter.items()]
    dic.sort(key=lambda x: x[1], reverse=True)
    return list(map(lambda x: x[0], dic))[:num_of_hubs]

def most_connected_alleles(network, num_of_alleles):
    """ This method finds the num_of_hubs most connected hubs. It doesn't split the alleles names like the method find_hubs
    """
    dic = copy.deepcopy(network)
    counter = {}
    for gene in dic.keys():
        counter[gene] = len(dic[gene].keys())
    dic = [(k,v) for k,v in counter.items()]
    dic.sort(key=lambda x: x[1], reverse=True)
    return list(map(lambda x: x[0], dic))[:num_of_alleles]

######################### Differences and STRING API #################################

def find_differences(dic1, dic2):
    """Find interactions in dic1 that are not in dic2"""
    tuple_list1 = []
    for gene in dic1.keys():
        list1 = [(gene.split('-')[0].upper(), i.split('-')[0].upper()) for i in dic1[gene].keys()]
        tuple_list1.extend(list1)
    tuple_list2 = []
    for gene in dic2.keys():
        list2 = [(gene.split('-')[0].upper(), i.split('-')[0].upper()) for i in dic2[gene].keys()]
        tuple_list2.extend(list2)
    tuple_set1 = set(tuple_list1)
    tuple_set2 = set(tuple_list2)
    diff = tuple_set1.difference(tuple_set2)
    diff_nr = []
    for tup in diff:
        if (tup[0], tup[1]) not in diff_nr and (tup[1], tup[0]) not in diff_nr:
            diff_nr.append(tup)
    return diff_nr




######### Additional


def get_genes(dic):
    genes = []
    for k in dic.keys():
        genes.append(k.split('-')[0].upper())
        for k1 in dic[k].keys():
            genes.append(k1.split('-')[0].upper())
    return set(genes)

def find_genes(df, gene_list):
    gene_list = list(map(lambda x: x.lower(), gene_list))
    not_found = []
    for gene in gene_list:
        df_ = df.loc[(df['Query allele name'].str.contains(gene)) | (df['Array allele name'].str.contains(gene))]
        if len(df_)==0:
            not_found.append(gene)
    if not len(not_found):
        print('All found')
    else:
        print('Not found: ', not_found)

def represent_network(gene_list, dic, dist=0):    
    dic0 = dict(filter(lambda x: x[0] in gene_list, dic.items()))
    dic_f = {}
    for k in dic0.keys():
        dic_f[k] = dict(filter(lambda x: x[0] in gene_list, dic0[k].items()))
    if dist==1:
        proc = list(dic0.keys())
        for gene in proc: # gene es un gen del procesoma
            proc = list(dic0.keys())
            interactors = list(dic[gene].keys()) # todos los genes que interactuan con gene
            interactors = list(filter(lambda x: x not in proc, interactors)) # me quedo con los que no estan en proc
            proc.remove(gene)
            for i in interactors: # i es cada interactor de gene
                if len(set(dic[i].keys()).intersection(set(proc)))>0:
                    dic_f[i] = dict(filter(lambda x: x[0] in dic0.keys(), dic[i].items()))
                    for j in dic_f[i].keys():
                        dic_f[j][i] = dic[i][j]
    return dic_f


def splitted_dict(df):
    df = df.dropna()            # Elimino filas con NaN
    df.reset_index(inplace=True, drop=True)
    ##################################
    df['Mul'] = df['Double mutant fitness']-df['Query single mutant fitness (SMF)']*df['Array SMF']         # columna con interacciones mul
    df['Add'] = df['Double mutant fitness']-(df['Query single mutant fitness (SMF)']+df['Array SMF']-1)     # columna con interacciones add
    ###################################
    std_mul = df['Mul'].std()           # Desviaciones estandar de cada columna
    std_add = df['Add'].std()
    ##################################
    df['Query allele name'] = df['Query allele name'].map(lambda x: x.split('-')[0].upper())        # Cambio de nombre
    df['Array allele name'] = df['Array allele name'].map(lambda x: x.split('-')[0].upper())
    ##################################
    df = df.loc[df['P-value']<0.05]             # Filtro P-value

    def df_to_dictionary_splitted(df, model):
        df.reset_index(inplace=True, drop=True)
        query_dic = {}
        array_dic = {}
        for i in range(len(df)):
            if model == 'mul':
                interaction = df.loc[i, 'Double mutant fitness']-df.loc[i, 'Query single mutant fitness (SMF)']*df.loc[i, 'Array SMF']
            if model == 'add':
                interaction = df.loc[i, 'Double mutant fitness']-(df.loc[i, 'Query single mutant fitness (SMF)']+df.loc[i, 'Array SMF']-1)
            if not (np.isnan(interaction) or np.isinf(interaction)):
                query = df.loc[i, 'Query allele name']
                array = df.loc[i, 'Array allele name']
                ########################################################
                if query not in query_dic.keys():
                    query_dic[query]={}
                if array not in query_dic[query].keys():
                    query_dic[query][array]=interaction
                elif abs(interaction) > abs(query_dic[query][array]):
                    query_dic[query][array]=interaction
                ########################################################
                if array not in array_dic.keys():
                    array_dic[array]={}
                if query not in array_dic[array].keys():
                    array_dic[array][query]=interaction
                elif abs(interaction) > abs(array_dic[array][query]):
                    array_dic[array][query]=interaction
                ########################################################
        return query_dic, array_dic

    ########################################################################

    q, a = df_to_dictionary_splitted(df, model='mul')
    mul = join_pcc(q, a)

    q, a = df_to_dictionary_splitted(df, model='add')
    add = join_pcc(q, a)

    ########################################################################

    for gene in mul.keys():
        mul[gene] = dict(filter(lambda x: abs(x[1])>2*std_mul, mul[gene].items()))

    for gene in add.keys():
        add[gene] = dict(filter(lambda x: abs(x[1])>2*std_add, add[gene].items()))

    return mul, add

######################### Notes #################################

#### MUL cuando se seleccionan 2.5% a cada lado
### E -0.13861973501750255 0.21021254676901852
### N -0.06755956498613237 0.08465028446325662
### global -0.06938762744455954 0.09125818791712999

#### ADD cuando se seleccionan 2.5% a cada lado
### E -0.11402645256193826 0.4216637528117716
### N -0.1020362657513211 0.1557469861740041
### global 



######################### Examples #################################

# a ={'a':{'b':1, 'c':1, 'd':1},
#     'b':{'a':1, 'c':1,'d':1, 'e':1},
#     'c':{'a':1, 'b':1, 'e':1, 'f':1},
#     'd':{'b':1, 'f':1, 'a':1, 'e':1},
#     'e':{'b':1, 'c':1, 'd':1},
#     'f':{'d':1, 'c':1}
# }
# b ={'e':{'h':1, 'f':1},
#     'f':{'e':1, 'i':1, 'g':1},
#     'g':{'f':1},
#     'h':{'e':1},
#     'i':{'f':1}
# }

# a = {
#     'a-1':{'b-1':1, 'b-2':1, 'b-3':1, 'c-1':1, 'c-2':1, 'd-8':1},
#     'a-2':{'b-2':1, 'b-3':1, 'c-1':1, 'd-2':1, 'e-8':1},
#     'b-1':{'a-1':1, 'c-1':1, 'c-2':1, 'r-3':1, 'f-2':1},
#     'b-2':{'a-1':1, 'a-2':1},
#     'b-3':{'a-1':1, 'a-2':1, 'c-1':1},
#     'c-1':{'a-1':1, 'a-2':1, 'b-1':1, 'b-3':1, 'd-8':1},
#     'c-2':{'a-1':1, 'b-1':1},
#     'd-2':{'a-2':1},
#     'd-8':{'a-1':1, 'c-1':1},
#     'e-8':{'a-2':1},
#     'f-2':{'b-1':1},
#     'r-3':{'b-1':1}
# }

# example = {
#             "1":{"2":1, "3":1, "4":1},
#             "2":{"1":1, "3":1},
#             "3":{"1":1, "2":1, "4":1},
#             "4":{"1":1, "3":1, "5":1, "6":1},
#             "5":{"4":1, "6":1, "7":1, "8":1},
#             "6":{"4":1, "5":1, "7":1, "8":1},
#             "7":{"5":1, "6":1, "8":1, "9":1},
#             "8":{"5":1, "6":1, "7":1},
#             "9":{"7":1}                                 
#            }