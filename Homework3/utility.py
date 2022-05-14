import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Initialize the initial infected in the graph
# following a specified policy
# random : the infected are initialized in a random way
# neighbors : all the intial infected are neighbors
def set_init_config( G, initial_i, policy):
    x0 = np.zeros([len(G)])
    
    if policy == "random":
        idx = np.random.choice(list(G.nodes), size =initial_i )

        for i in idx:
            x0[idx] = 1
        
        return x0
    
    
    elif policy == "neighbors":
        # Take the first infected randomly
        # until he has at a least a neighbor
        while True:
            idx = np.random.choice(list(G.nodes))
            
            if len(dict(G.adj[idx]).keys()) != 0 :
                break
        
        x0[idx] = 1
        counter = 1
        finish = False
        
        # Take the nieghbors
        while True:
            neig = dict( G.adj[int(idx)] ).keys()
            
            # Put the neighbors to I until the
            # required number is reached
            for n in neig :
                if counter == initial_i :
                    finish = True
                    break
                else :
                    if x0[n] == 0:
                        x0[n] = 1
                        counter += 1
            
            if finish == True:
                break
    
            # Find another neighbor with nodes
            for i in np.argwhere(x0 == 1)[0]:
            
                n_list = list(dict(G.adj[int(i)]).keys())
                if len(n_list) != 0 and len(np.argwhere(x0[n_list]==0)) !=0 :
                    idx = i
                    
                else :
                    idx = np.random.choice(np.argwhere(x0 == 0).squeeze())
    
        return x0
        
    else :
        print("Wrong policy")
        
        
        
# Count the number of nodes in the different states
def count_SIR( states, n_states ):
    cardinalities = np.zeros([states.shape[0],n_states ])
    
    for i in range(states.shape[0]):
        for n_s in range(n_states):
            cardinalities[i, n_s] = int(np.argwhere(states[i] == n_s).shape[0])
    
    return cardinalities



# Count the new individuals in that state respect
# to the previous week
def new_weekly(states, s):
    cardinalities = np.zeros(states.shape[0])
    
    for n_w in range(states.shape[0]):
        
        if n_w == 0 :
            cardinalities[n_w] = int(np.argwhere(states[n_w] == s).shape[0])
        else :
            cardinalities[n_w] = int(np.argwhere( ( (states[n_w]==s) & (states[n_w-1]<states[n_w]) ) ).shape[0])
            
    return cardinalities



# Returns the rates of a node to the next configuration
def single_rate(state, m, beta, ro, vax=False):

    # Considering the current state compute the
    # probabilities of the next possible ones
    if vax == True:
        if state == 0 :
            return np.array([(1-beta)**m,  1 - (1-beta)**m, 0, 0])
    
        elif state == 1 :
            return np.array([0,  1-ro, ro, 0 ])
    
        elif state == 2 :
            return np.array([0,  0, 1, 0 ])
    
        elif state == 3 :
            return np.array([0,  0, 0, 1 ])
    
    else :
        if state == 0 :
            return np.array([(1-beta)**m,  1 - (1-beta)**m, 0])
    
        if state == 1 :
            return np.array([0,  1-ro, ro ])
    
        elif state == 2 :
            return np.array([0,  0, 1 ])
        
        
        
# Comput the rates of the new possible conigurations
def new_conf_rates(G, x, beta, ro, n_agents):
    
    matrix = np.zeros([len(G), n_agents])
    
    # Nodes' loop
    for agent in range(len(G)) :
        
        # if the state is Susceptible the next one
        # influenced by the infected neighbors
        # so compute it
        if x[agent] == 0:
            m=0
            for neig in dict(G.adj[agent]).keys() :
                if x[neig] == 1:
                    m+=1
        else: m=0
        
        # if SIR model
        if n_agents == 3 :
            matrix[agent] = single_rate(x[agent], m, beta, ro, False)
        
        # if SIRV model
        elif n_agents == 4:
            matrix[agent] = single_rate(x[agent], m, beta, ro, True)
            
    return matrix



# Vaccination step 
def vaccination(perc, state, n_nodes):
    # Chose between non vaccinated people
    idx_si = np.where(state < 3 )[0]
    
    # Chose the right percentage of people
    new_n = round((n_nodes * perc) / 100)
    idx_tov = np.random.choice(idx_si, size=new_n, replace=False)
    
    # Set selected people as Vaccianted (3)
    for i in idx_tov:
        state[idx_tov] = 3
        
    return state



# Plot the pandemic evolution on the graph
def graph_evolution(G, states, n_stati):
    colors = ['w', 'r', 'g', 'y']
    pos = nx.spectral_layout(G)
    fig = plt.figure(figsize=(20,20))
    
    for t in range(0,len(states)):
        plt.subplot(round(len(states)/4)+1,4,t+1)
        x = states[t]
        
        for s in range(n_stati):
            nx.draw(G,
                pos = pos,
                with_labels=True,
                nodelist=np.argwhere(x==s).T[0].tolist(),
                node_color = colors[s]
                )
            plt.title('Week = {0}'.format(t+1))
    plt.show()
