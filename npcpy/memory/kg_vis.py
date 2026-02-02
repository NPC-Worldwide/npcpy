


import json
from collections import defaultdict
import pandas as pd
import math
from textwrap import fill
from chroptiks.plotting_utils import *


import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch
import numpy as np
from pyvis.network import Network
import networkx as nx  

def load_kg_with_pandas(generation, path_prefix="kg_state"):
    """Loads the new graph structure from CSV files."""
    kg = {
        "generation": generation, "facts": [], "concepts": [],
        "concept_links": [], "fact_to_concept_links": {}
    }
    try:
        nodes_df = pd.read_csv(f'{path_prefix}_gen{generation}_nodes.csv')
        links_df = pd.read_csv(f'{path_prefix}_gen{generation}_links.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find data files for generation {generation}. {e}")
        return None

    for _, row in nodes_df.iterrows():
        if row['type'] == 'fact':
            kg['facts'].append({'statement': row['id'], 'generation': int(row['generation'])})
        elif row['type'] == 'concept':
            kg['concepts'].append({'name': row['id'], 'generation': int(row['generation'])})

    fact_links = defaultdict(list)
    concept_links = []
    for _, row in links_df.iterrows():
        if row['type'] == 'fact_to_concept':
            fact_links[row['source']].append(row['target'])
        elif row['type'] == 'concept_to_concept':
            concept_links.append((row['source'], row['target']))
    
    kg['fact_to_concept_links'] = dict(fact_links)
    kg['concept_links'] = concept_links
    
    print(f"Successfully loaded KG Generation {generation} with pandas.")
    return kg

def load_changelog_from_json(from_gen, to_gen, path_prefix="changelog"):
    """Loads the detailed changelog JSON file created during a 'kg_sleep_process'."""
    filename = f"{path_prefix}_gen{from_gen}_to_{to_gen}.json"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            changelog = json.load(f)
            print(f"Successfully loaded changelog from {filename}")
            return changelog
    except FileNotFoundError as e:
        print(f"Error: Could not find changelog file: {e}")
        return None


def visualize_knowledge_graph_final_interactive(kg, filename="knowledge_graph.html"):
    """Updated to work with the new KG structure"""
    print(f"Generating interactive graph for Gen {kg['generation']} -> {filename}")
    
    # Extract facts and concepts from the new structure
    facts = kg.get("facts", [])
    concepts = kg.get("concepts", [])
    fact_to_concept_links = kg.get("fact_to_concept_links", {})
    concept_links = kg.get("concept_links", [])
    
    # Create node map for easy lookup
    node_map = {}
    for fact in facts:
        node_map[fact['statement']] = fact
    for concept in concepts:
        node_map[concept['name']] = concept
    
    # Calculate positions in concentric circles
    # Facts in inner circle, concepts in outer circle
    fact_radius = 300
    concept_radius = 600
    
    node_positions = {}
    
    # Position facts
    if facts:
        for i, fact in enumerate(facts):
            angle = (2 * math.pi * i) / len(facts)
            node_id = fact['statement']
            node_positions[node_id] = {
                'x': fact_radius * math.cos(angle), 
                'y': fact_radius * math.sin(angle)
            }
    
    # Position concepts
    if concepts:
        for i, concept in enumerate(concepts):
            angle = (2 * math.pi * i) / len(concepts)
            node_id = concept['name']
            node_positions[node_id] = {
                'x': concept_radius * math.cos(angle), 
                'y': concept_radius * math.sin(angle)
            }
    
    # Create the network
    net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white", directed=True)
    
    # Add fact nodes
    for fact in facts:
        node_id = fact['statement']
        pos = node_positions.get(node_id, {'x': 0, 'y': 0})
        title_text = f"<strong>Fact (Gen: {fact.get('generation', 'N/A')})</strong><br><em>{fill(node_id, 50)}</em>"
        net.add_node(
            node_id, 
            label=fill(node_id, 25), 
            title=title_text, 
            x=pos['x'], 
            y=pos['y'], 
            color='#ff6961',  # Red for facts
            physics=False
        )
    
    # Add concept nodes
    for concept in concepts:
        node_id = concept['name']
        pos = node_positions.get(node_id, {'x': 0, 'y': 0})
        title_text = f"<strong>Concept (Gen: {concept.get('generation', 'N/A')})</strong><br><em>{fill(node_id, 50)}</em>"
        net.add_node(
            node_id, 
            label=fill(node_id, 25), 
            title=title_text, 
            x=pos['x'], 
            y=pos['y'], 
            color='#ffb480',  # Orange for concepts
            physics=False
        )
    
    # Add fact-to-concept edges
    for fact_statement, concept_names in fact_to_concept_links.items():
        for concept_name in concept_names:
            if fact_statement in node_map and concept_name in node_map:
                net.add_edge(fact_statement, concept_name, color="#8ecae6", width=1)
    
    # Add concept-to-con
def visualize_growth(k_graphs, filename="growth_chart.png"):
    """
    Plots Facts and Concepts as separate lines instead of a stacked area.
    This allows for independent analysis of each component's growth over time.
    """
    gens = [kg['generation'] for kg in k_graphs]
    facts_counts = [len(kg.get('facts', [])) for kg in k_graphs]
    concepts_counts = [len(kg.get('concepts', [])) for kg in k_graphs]
    total_nodes = [facts + concepts for facts, concepts in zip(facts_counts, concepts_counts)]

    plt.figure(figsize=(12, 8))
    

    plt.plot(gens, facts_counts, label='Facts', color='#ff6961', marker='o', linestyle='-', linewidth=2)
    plt.plot(gens, concepts_counts, label='Concepts', color='#ffb480', marker='o', linestyle='-', linewidth=2)
    plt.plot(gens, total_nodes, label='Total Nodes', color='#8ecae6', marker='x', linestyle='--', alpha=0.7)

    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Number of Nodes", fontsize=14)
    plt.legend(loc='upper left', fontsize=20, frameon=False)
    plt.xticks(gens)
    plt.ylim(bottom=0) # Ensure the y-axis starts at 0
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved independent growth chart to {filename}")
    
    
def visualize_fact_concept_ratio(kg_pairs, filename="fact_concept_ratio.png"):
    """Updated to work with the new KG structure"""
    labels, before_ratios, after_ratios = [], [], []
    for kg_before, kg_after in kg_pairs:
        facts_before = len(kg_before.get('facts', []))
        concepts_before = len(kg_before.get('concepts', []))
        before_ratios.append(facts_before / concepts_before if concepts_before > 0 else 0)
        
        facts_after = len(kg_after.get('facts', []))
        concepts_after = len(kg_after.get('concepts', []))
        after_ratios.append(facts_after / concepts_after if concepts_after > 0 else 0)
        
        labels.append(f"Gen {kg_before['generation']}→{kg_after['generation']}")
    
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, before_ratios, width, label='Before Sleep', color='#95a5a6')
    rects2 = ax.bar(x + width/2, after_ratios, width, label='After Sleep', color='#2ecc71')
    ax.set_ylabel("Fact-to-Concept Ratio",)
    ax.set_xticks(x, labels, fontsize=12, rotation=45, ha="right")
    ax.legend(fontsize=20, frameon=False)
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=10)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=10)
    ax.set_ylim(bottom=0, top=max(max(before_ratios, default=0), max(after_ratios, default=0)) * 1.5)
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved fact-to-concept ratio chart to {filename}")
def visualize_sleep_process(kg_before, kg_after, filename="sleep_process.png"):
    """Simple visualization of before/after states"""
    print(f"\n--- Visualizing Sleep Process: Gen {kg_before['generation']} -> Gen {kg_after['generation']} ---")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Before state
    facts_before = len(kg_before.get('facts', []))
    concepts_before = len(kg_before.get('concepts', []))
    ax1.pie([facts_before, concepts_before], labels=['Facts', 'Concepts'], 
            colors=['#ff6961', '#ffb480'], autopct='%1.1f%%')
    #ax1.set_title(f"Before Sleep (Gen {kg_before['generation']})")
    
    # After state
    facts_after = len(kg_after.get('facts', []))
    concepts_after = len(kg_after.get('concepts', []))
    ax2.pie([facts_after, concepts_after], labels=['Facts', 'Concepts'], 
            colors=['#ff6961', '#ffb480'], autopct='%1.1f%%')
    #ax2.set_title(f"After Sleep (Gen {kg_after['generation']})")
    
    #plt.suptitle(f"Sleep Process: Generation {kg_before['generation']} to {kg_after['generation']}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved sleep process visualization to {filename}")
    
def _create_networkx_graph_full(kg):
    """helper to build the complete graph including fact-to-fact links."""
    G = nx.Graph()
    concepts = [c['name'] for c in kg.get('concepts', [])]
    facts = [f['statement'] for f in kg.get('facts', [])]
    G.add_nodes_from(concepts, type='concept')
    G.add_nodes_from(facts, type='fact')
    for fact, linked_concepts in kg.get('fact_to_concept_links', {}).items():
        for concept in linked_concepts:
            if G.has_node(fact) and G.has_node(concept): G.add_edge(fact, concept)
    for c1, c2 in kg.get('concept_links', []):
        if G.has_node(c1) and G.has_node(c2): G.add_edge(c1, c2)

    for f1, f2 in kg.get('fact_to_fact_links', []):
        if G.has_node(f1) and G.has_node(f2): G.add_edge(f1, f2)
    return G


def visualize_key_experiences(kg, filename="key_experiences.png"):
    """
    Visualizes the full network, highlighting the most central "key experience" facts.
    """
    print(f"Generating Key Experience network graph for Gen {kg['generation']} -> {filename}")
    G = _create_networkx_graph_full(kg)
    if not G.nodes: return

    facts = {n for n, d in G.nodes(data=True) if d['type'] == 'fact'}
    concepts = {n for n, d in G.nodes(data=True) if d['type'] == 'concept'}
    

    centrality = nx.degree_centrality(G)
    

    top_facts = sorted(facts, key=lambda n: centrality[n], reverse=True)[:5]


    node_colors = []
    for node in G:
        if node in top_facts:
            node_colors.append('#ff0000')  # Bright Red for key facts
        elif G.nodes[node]['type'] == 'fact':
            node_colors.append('#ff6961')  # Standard Red for facts
        else:
            node_colors.append('#ffb480')  # Orange for concepts
            
    plt.figure(figsize=(24, 24))
    pos = nx.spring_layout(G, k=1.5/math.sqrt(G.number_of_nodes()), iterations=100, seed=42)
    
    nx.draw(G, pos, with_labels=False, node_color=node_colors, 
            node_size=[v * 10000 for v in centrality.values()], 
            width=0.5, edge_color='gray', alpha=0.7)


    labels = {n: fill(n, 15) for n in top_facts + list(concepts)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def _create_networkx_graph(kg):
    """Helper function to convert our KG dict into a NetworkX graph for analysis."""
    G = nx.Graph()
    concepts = [c['name'] for c in kg.get('concepts', [])]
    facts = [f['statement'] for f in kg.get('facts', [])]
    
    G.add_nodes_from(concepts, type='concept')
    G.add_nodes_from(facts, type='fact')
    
    for fact, linked_concepts in kg.get('fact_to_concept_links', {}).items():
        for concept in linked_concepts:
            if G.has_node(fact) and G.has_node(concept):
                G.add_edge(fact, concept)
    
    for c1, c2 in kg.get('concept_links', []):
        if G.has_node(c1) and G.has_node(c2):
            G.add_edge(c1, c2)
            
    return G
def visualize_concept_trajectories(kg_history, n_pillars=2, n_risers=3, filename="concept_trajectories.png"):
    """
    To ensure pillars and risers are distinct sets, telling a clearer story
    about the stable backbone vs. major new themes.
    """
    print(f"Generating Disjoint Concept Trajectories chart -> {filename}")
    centrality_df = pd.DataFrame()

    gens = [kg['generation'] for kg in kg_history]
    for i, kg in enumerate(kg_history):
        G = _create_networkx_graph(kg)
        if not G.nodes: continue
        degree_centrality = nx.degree_centrality(G)
        concept_centrality = {node: cent for node, cent in degree_centrality.items() if G.nodes[node].get('type') == 'concept'}
        s = pd.Series(concept_centrality, name=kg['generation'])
        centrality_df = pd.concat([centrality_df, s.to_frame()], axis=1)
    centrality_df = centrality_df.transpose().sort_index()

    # Pillars: High average centrality
    pillars = centrality_df.mean().nlargest(n_pillars).index
    
    # Rising Stars: Exclude pillars from the candidates to find distinct risers
    riser_candidates = centrality_df.drop(columns=pillars, errors='ignore')
    centrality_diff = riser_candidates.iloc[-1].fillna(0) - riser_candidates.iloc[0].fillna(0)
    risers = centrality_diff.nlargest(n_risers).index

    concepts_to_plot = pillars.union(risers)

    plt.figure(figsize=(12, 8))
    for concept_name in concepts_to_plot:
        trajectory = centrality_df[concept_name]
        style = '--' if concept_name in pillars else '-'
        linewidth = 1.5 if concept_name in pillars else 2.5
        alpha = 0.8 if concept_name in pillars else 1.0
        plt.plot(trajectory.index, trajectory.values, marker='o', linestyle=style, 
                 label=fill(concept_name, 20), linewidth=linewidth, alpha=alpha)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Degree Centrality", fontsize=14)
    #plt.title("Notable Concept Trajectories (Pillars vs. Risers)", fontsize=16)
    plt.xticks(gens)
    plt.legend(title="Concepts", bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
def visualize_associative_richness(kg_history, filename="associative_richness.png"):
    """Plots the Associative Richness Index (ARI): Avg. Concepts per Fact."""
    print(f"Generating Associative Richness chart -> {filename}")
    gens = [kg['generation'] for kg in kg_history]
    ari_scores = []
    for kg in kg_history:
        num_facts = len(kg.get('facts', []))
        total_links = sum(len(links) for links in kg.get('fact_to_concept_links', {}).values())
        ari_scores.append(total_links / num_facts if num_facts > 0 else 0)

    plt.figure(figsize=(12, 8))
    plt.plot(gens, ari_scores, marker='o', linestyle='-', color='#6a0dad', linewidth=2.5, label="System's ARI")
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='1-to-1 Mapping Baseline (ARI=1.0)')
    plt.xlabel("Generation")
    plt.ylabel("Avg. Concepts per Fact (ARI)")
    plt.xticks(gens)
    plt.legend(loc='lower right')
    plt.ylim(bottom=0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_conceptual_support(kg_history, filename="conceptual_support.png"):
    """Plots the Conceptual Support Index (CSI): Avg. Facts per Concept."""
    print(f"Generating Conceptual Support chart -> {filename}")
    gens = [kg['generation'] for kg in kg_history]
    csi_scores = []
    for kg in kg_history:
        num_concepts = len(kg.get('concepts', []))
        total_links = sum(len(links) for links in kg.get('fact_to_concept_links', {}).values())
        csi_scores.append(total_links / num_concepts if num_concepts > 0 else 0)

    plt.figure(figsize=(12, 8))
    plt.plot(gens, csi_scores, marker='o', linestyle='-', color='#17becf', linewidth=2.5, label="System's CSI")
    plt.xlabel("Generation")
    plt.ylabel("Avg. Facts per Concept (CSI)")
    plt.xticks(gens)
    plt.legend(loc='lower right')
    plt.ylim(bottom=0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_specialist_concepts(kg_history, num_to_show=8, filename="specialist_concepts.png"):
    """
    Plots trajectories of interesting 'middling' concepts by finding those with
    high variance and peak centrality, while excluding the absolute top global hubs.
    """
    print(f"Generating Specialist Concept Trajectories chart -> {filename}")
    centrality_df = pd.DataFrame()
    gens = [kg['generation'] for kg in kg_history] # Define gens here

    for kg in kg_history:
        G = _create_networkx_graph(kg)
        concept_centrality = {n: nx.degree_centrality(G)[n] for n, d in G.nodes(data=True) if d['type'] == 'concept'} if G.nodes else {}
        centrality_df = pd.concat([centrality_df, pd.Series(concept_centrality, name=kg['generation'])], axis=1)
    centrality_df = centrality_df.transpose().sort_index()


    top_hubs = centrality_df.mean().nlargest(5).index
    specialist_candidates = centrality_df.drop(columns=top_hubs, errors='ignore')


    notability_scores = specialist_candidates.max() + specialist_candidates.var().fillna(0)
    concepts_to_plot = notability_scores.nlargest(num_to_show).index

    plt.figure(figsize=(12, 8))
    for name in concepts_to_plot:
        trajectory = centrality_df[name]
        plt.plot(trajectory.index, trajectory.values, marker='o', linestyle='-', label=fill(name, 25))
    
    plt.xlabel("Generation")
    plt.ylabel("Degree Centrality")
    
    plt.xticks(gens) # Use the 'gens' list, not a single value
    
    plt.legend(title="Specialist Concepts",  loc=0, fontsize=17, frameon=False)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_static_network(kg, top_n_concepts=25, top_n_facts=50, filename="static_network.png"):
    """
    Creates a clean, ordered bipartite graph showing ONLY the most central concepts
    and facts, preventing visual clutter.
    """
    print(f"Generating ordered static network for Gen {kg['generation']} -> {filename}")
    G = _create_networkx_graph(kg)
    if not G.nodes: return


    concepts = {n for n, d in G.nodes(data=True) if d['type'] == 'concept'}
    facts = {n for n, d in G.nodes(data=True) if d['type'] == 'fact'}
    
    top_concepts = sorted(concepts, key=G.degree, reverse=True)[:top_n_concepts]
    top_facts = sorted(facts, key=G.degree, reverse=True)[:top_n_facts]


    SubG = G.subgraph(top_concepts + top_facts)


    pos = {}
    for i, node in enumerate(top_concepts): pos[node] = (-1, np.linspace(1, 0, len(top_concepts))[i])
    for i, node in enumerate(top_facts): pos[node] = (1, np.linspace(1, 0, len(top_facts))[i])

    plt.figure(figsize=(16, 24))
    

    nx.draw_networkx_nodes(SubG, pos, nodelist=top_facts, node_color='#ff6961', node_size=150)
    nx.draw_networkx_nodes(SubG, pos, nodelist=top_concepts, node_color='#ffb480', node_size=1200)


    nx.draw_networkx_edges(SubG, pos, alpha=0.25, width=0.6, edge_color='gray')


    concept_labels = {name: fill(name, 20) for name in top_concepts}
    nx.draw_networkx_labels(SubG, pos, labels=concept_labels, font_size=14, font_family='serif', horizontalalignment='right')

    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
def visualize_concept_ontology_graph(kg, filename="concept_ontology.png"):
    """
    Creates a 'bubble map' of the CONCEPT ontology.
    - Nodes are concepts only.
    - Edges are only concept-to-concept links.
    - Node size is proportional to its total degree (including fact links),
      representing its overall importance.
    """
    print(f"Generating Concept Ontology Bubble Map for Gen {kg['generation']} -> {filename}")


    Full_G = _create_networkx_graph(kg)
    if not Full_G.nodes:
        print(f"  - KG {kg['generation']} has no nodes. Skipping.")
        return


    Concept_G = nx.Graph()
    concept_names = [c['name'] for c in kg.get('concepts', [])]
    Concept_G.add_nodes_from(concept_names)
    for c1, c2 in kg.get('concept_links', []):
        if Concept_G.has_node(c1) and Concept_G.has_node(c2):
            Concept_G.add_edge(c1, c2)



    node_sizes = [500 + (Full_G.degree(n) * 50) for n in Concept_G.nodes()]


    plt.figure(figsize=(24, 24))

    pos = nx.spring_layout(Concept_G, k=1.5/math.sqrt(Concept_G.number_of_nodes()), iterations=100, seed=42)

    nx.draw_networkx_nodes(Concept_G, pos, node_color='#ffb480', node_size=node_sizes)
    nx.draw_networkx_edges(Concept_G, pos, alpha=0.6, width=1.0, edge_color='gray')
    nx.draw_networkx_labels(Concept_G, pos, font_size=14, font_family='serif')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()



def visualize_top_concept_centrality(kg_history, top_n=5, filename="concept_centrality.png"):
    """
    Tracks the degree centrality of the top N most important concepts over time.
    This shows how a thematic backbone emerges and solidifies within the KG.
    """
    centrality_data = defaultdict(lambda: [np.nan] * len(kg_history))
    
    for i, kg in enumerate(kg_history):
        G = _create_networkx_graph(kg)
        if not G.nodes: continue
        
        degree_centrality = nx.degree_centrality(G)
        
        # Filter for concepts only
        concept_centrality = {node: cent for node, cent in degree_centrality.items() if G.nodes[node]['type'] == 'concept'}
        
        for concept_name, centrality in concept_centrality.items():
            centrality_data[concept_name][i] = centrality
    
    # Find the top N concepts based on their peak centrality
    sorted_concepts = sorted(centrality_data.keys(), key=lambda c: np.nanmax(centrality_data[c]), reverse=True)
    top_concepts = sorted_concepts[:top_n]
    
    plt.figure(figsize=(12, 8))
    gens = [kg['generation'] for kg in kg_history]
    
    for concept_name in top_concepts:
        # Interpolate NaN values for smoother plotting if a concept disappears and reappears
        s = pd.Series(centrality_data[concept_name])
        s_interpolated = s.interpolate(method='linear', limit_direction='forward', axis=0)
        plt.plot(gens, s_interpolated, marker='o', linestyle='-', label=fill(concept_name, 20))

    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Degree Centrality", fontsize=14)
    plt.xticks(gens)
    plt.legend(title="Top Concepts",  loc=0, frameon=False, fontsize=20)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Top Concept Centrality chart to {filename}")

def visualize_lorenz_curve(kg_history, filename="lorenz_curve.png"):
    """
    Creates a standalone Lorenz curve plot to compare the degree distribution
    inequality between the first and final generations.
    """
    print(f"Generating Lorenz Curve comparison -> {filename}")

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- Lorenz curve for the first valid generation ---
    first_gen_kg = next((kg for kg in kg_history if kg.get('facts')), None)
    if first_gen_kg:
        G_first = _create_networkx_graph(first_gen_kg)
        degrees_first = np.array(sorted([d for n, d in G_first.degree()]))
        if degrees_first.size > 0:
            cum_degrees_first = np.cumsum(degrees_first)
            ax.plot(np.linspace(0, 1, len(degrees_first)), cum_degrees_first / cum_degrees_first[-1],
                     label=f"Gen {first_gen_kg['generation']} (Start)", color='#1f77b4', linewidth=2)

    # --- Lorenz curve for the last generation ---
    last_gen_kg = kg_history[-1]
    G_last = _create_networkx_graph(last_gen_kg)
    degrees_last = np.array(sorted([d for n, d in G_last.degree()]))
    if degrees_last.size > 0:
        cum_degrees_last = np.cumsum(degrees_last)
        ax.plot(np.linspace(0, 1, len(degrees_last)), cum_degrees_last / cum_degrees_last[-1],
                 label=f"Gen {last_gen_kg['generation']} (End)", color='#ff7f0e', linewidth=2)

    # --- Line of perfect equality for reference ---
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Equality')

    ax.set_xlabel("Cumulative Share of Nodes", fontsize=14)
    ax.set_ylabel("Cumulative Share of Connections", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_aspect('equal', adjustable='box') # Ensures the plot is square
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
def visualize_concept_bubble_chart(kg, filename="concept_bubble_chart.png"):
    """
    Creates a 'bubble chart' of the concept ontology, arranged like a word cloud.
    - The most important concept (highest degree) is fixed at the center.
    - All other concepts are arranged around it using a force-directed layout.
    - Node size is proportional to its total degree.
    - No edges are drawn, for maximum clarity.
    """
    print(f"Generating CENTRALIZED Concept Bubble Chart for Gen {kg['generation']} -> {filename}")


    Full_G = _create_networkx_graph(kg)
    if not Full_G.nodes:
        print(f"  - KG {kg['generation']} has no nodes. Skipping.")
        return


    concepts = {node: Full_G.degree(node) for node, data in Full_G.nodes(data=True) if data['type'] == 'concept'}
    
    if not concepts:
        print(f"  - KG {kg['generation']} has no concepts. Skipping.")
        return


    Concept_G = nx.Graph()
    Concept_G.add_nodes_from(concepts.keys())
    for c1, c2 in kg.get('concept_links', []):
        if Concept_G.has_node(c1) and Concept_G.has_node(c2):
            Concept_G.add_edge(c1, c2)


    central_node = max(concepts, key=concepts.get)
    fixed_nodes = [central_node]
    pos_initial = {central_node: (0, 0)} # Pin the most important node at (0,0)


    pos = nx.spring_layout(Concept_G, pos=pos_initial, fixed=fixed_nodes, 
                           k=1.8/math.sqrt(Concept_G.number_of_nodes()), # Controls spacing
                           iterations=200, seed=42)


    plt.figure(figsize=(20, 20))
    

    node_sizes = [concepts[node] * 200 for node in Concept_G.nodes()]
    
    nx.draw_networkx_nodes(Concept_G, pos, node_color='#ffb480', node_size=node_sizes, alpha=0.9)
    
    for node, (x, y) in pos.items():
        degree = concepts[node]
        font_size = 8 + 2 * math.log(1 + degree)
        plt.text(x, y, fill(node, 15), ha='center', va='center', fontsize=font_size, fontfamily='serif')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_centrality_bubble_chart(kg, node_type="concepts", filename="concept_bubble_chart.png"):
    """
    Creates a 'bubble chart' where nodes are arranged purely by importance
    (degree centrality), with the most important nodes in the center.
    
    Args:
        kg: Knowledge graph data
        node_type: "concepts", "facts", or "both" - which nodes to visualize
        filename: Output filename
    """
    print(f"Generating CENTRALITY-BASED Bubble Chart for {node_type} in Gen {kg['generation']} -> {filename}")

    Full_G = _create_networkx_graph(kg)
    if not Full_G.nodes:
        print(f"  - KG {kg['generation']} has no nodes. Skipping.")
        return

    all_nodes = {}
    for node, data in Full_G.nodes(data=True):
        if node_type == "concepts" and data['type'] == 'concept':
            all_nodes[node] = Full_G.degree(node)
        elif node_type == "facts" and data['type'] == 'fact':
            all_nodes[node] = Full_G.degree(node)
        elif node_type == "both":
            all_nodes[node] = {'degree': Full_G.degree(node), 'type': data['type']}
    
    if not all_nodes:
        print(f"  - KG {kg['generation']} has no {node_type}. Skipping.")
        return
    
    if node_type == "both":
        sorted_nodes = sorted(all_nodes.items(), key=lambda item: item[1]['degree'], reverse=True)
    else:
        sorted_nodes = sorted(all_nodes.items(), key=lambda item: item[1], reverse=True)

    pos = {}
    if sorted_nodes:
        central_node, _ = sorted_nodes[0]
        pos[central_node] = (0, 0)
        
        radius = 0.25
        nodes_in_ring = 6
        node_idx = 1
        
        while node_idx < len(sorted_nodes):
            angle_step = 2 * np.pi / nodes_in_ring
            for i in range(nodes_in_ring):
                if node_idx >= len(sorted_nodes): break
                angle = i * angle_step
                node_name, _ = sorted_nodes[node_idx]
                pos[node_name] = (radius * np.cos(angle), radius * np.sin(angle))
                node_idx += 1
            
            radius += 0.20
            nodes_in_ring = int(nodes_in_ring * 1.5)

    # Step 4: Draw the plot
    plt.figure(figsize=(20, 20))
    
    if node_type == "both":
        # Separate and draw by type
        concept_nodes = [(name, data) for name, data in sorted_nodes if data['type'] == 'concept']
        fact_nodes = [(name, data) for name, data in sorted_nodes if data['type'] == 'fact']
        
        # Draw concepts
        if concept_nodes:
            concept_names = [item[0] for item in concept_nodes]
            concept_sizes = [item[1]['degree'] * 200 for item in concept_nodes]
            concept_pos = {name: pos[name] for name in concept_names if name in pos}
            nx.draw_networkx_nodes(None, concept_pos, nodelist=concept_names, 
                                  node_color='#ffb480', node_size=concept_sizes, alpha=0.9)
        
        # Draw facts
        if fact_nodes:
            fact_names = [item[0] for item in fact_nodes]
            fact_sizes = [item[1]['degree'] * 100 for item in fact_nodes]
            fact_pos = {name: pos[name] for name in fact_names if name in pos}
            nx.draw_networkx_nodes(None, fact_pos, nodelist=fact_names, 
                                  node_color='#ff6961', node_size=fact_sizes, alpha=0.9)
        
        # Add legend for both types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ffb480', label='Concepts'),
            Patch(facecolor='#ff6961', label='Facts')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=14)
        
    else:
        # Single type visualization
        node_names = [item[0] for item in sorted_nodes]
        node_sizes = [item[1] * 200 for item in sorted_nodes]
        
        # Choose color based on type
        color = '#ffb480' if node_type == "concepts" else '#ff6961'
        
        nx.draw_networkx_nodes(None, pos, nodelist=node_names, 
                              node_color=color, node_size=node_sizes, alpha=0.9)
    

    top_nodes = sorted_nodes[:min(20, len(sorted_nodes))]  # Top 20 nodes only
    
    for item in top_nodes:
        node_name = item[0]
        if node_type == "both":
            degree = item[1]['degree']
            node_type_actual = item[1]['type']
        else:
            degree = item[1]
            node_type_actual = node_type.rstrip('s')  # "concepts" -> "concept"
        
        if node_name in pos:
            x, y = pos[node_name]
            font_size = max(8, 8 + 2 * np.log1p(degree))
            
            # Adjust label length based on type
            if node_type_actual == 'fact':
                label = fill(node_name, 10)
            else:
                label = fill(node_name, 15)
                
            plt.text(x, y, label, ha='center', va='center', 
                    fontsize=font_size, fontfamily='serif')

    plt.axis('off')
    
    # Set axis limits
    max_coord = radius * 1.1
    plt.xlim(-max_coord, max_coord)
    plt.ylim(-max_coord, max_coord)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
def visualize_dual_richness_metrics(kg_history, filename="dual_richness_metrics.png"):
    """
    Creates a two-panel plot showing ARI and CSI, stacked vertically.
    """
    print(f"Generating Dual Richness Metrics chart -> {filename}")
    
    gens = [kg['generation'] for kg in kg_history]
    ari_scores = [] # Concepts per Fact
    csi_scores = [] # Facts per Concept

    for kg in kg_history:
        num_facts = len(kg.get('facts', []))
        num_concepts = len(kg.get('concepts', []))
        
        total_links = sum(len(links) for links in kg.get('fact_to_concept_links', {}).values())
        
        ari_scores.append(total_links / num_facts if num_facts > 0 else 0)
        csi_scores.append(total_links / num_concepts if num_concepts > 0 else 0)


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), sharex=True)


    ax1.plot(gens, ari_scores, marker='o', linestyle='-', color='#6a0dad', linewidth=2.5, label="System's ARI")
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='1-to-1 Mapping Baseline (ARI=1.0)')
    ax1.set_ylabel("Avg. Concepts per Fact (ARI)", fontsize=14)
    ax1.legend(loc='lower right')
    ax1.set_ylim(bottom=0)


    ax2.plot(gens, csi_scores, marker='o', linestyle='-', color='#17becf', linewidth=2.5, label="System's CSI")
    ax2.set_xlabel("Generation", fontsize=14)
    ax2.set_ylabel("Avg. Facts per Concept (CSI)", fontsize=14)
    ax2.legend(loc='lower right')
    ax2.set_ylim(bottom=0)

    plt.xticks(gens)
    fig.tight_layout(pad=2.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()