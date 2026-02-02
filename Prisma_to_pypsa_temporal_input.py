import pypsa
import pandas as pd
import itertools
import os

def get_temporal_input(name):
    nodes = pd.read_csv("pop_layout_base_s_90.csv")   
    dac_input = pd.DataFrame(index=pd.to_datetime(n.snapshots))
    df_new = df.replace(to_replace="UK", value="GB")
    df_pivot = df_new.pivot(index='country', columns='temp', values=name)
    for cc in df_new["country"].unique():
        for node in nodes["name"][cc==nodes["ct"]]:
            #print(node, df_pivot.loc[cc,"winter"]*winter+df_pivot.loc[cc,"summer"]*summer+df_pivot.loc[cc,"year"]*spring_autumn)
            #print(df_pivot.loc[cc,"winter"])
            dac_input[node]=df_pivot.loc[cc,"winter"]*winter+df_pivot.loc[cc,"summer"]*summer+df_pivot.loc[cc,"year"]*spring_autumn
    filename = f"mofs_{par}/{name}.csv"
    dac_input.to_csv(filename)
    return dac_input
def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and.

    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0, r / (1.0 - 1.0 / (1.0 + r) ** n)
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n
    
def get_input():
    nodes = pd.read_csv("pop_layout_base_s_90.csv")   
    dac_input = pd.DataFrame(index=nodes["name"])
    df_new = df.replace(to_replace="UK", value="GB")
    for name in ["investment","FOM"]:
        df_pivot = df_new.pivot(index='country', columns='temp', values=name)

        dac_list = []
        for cc in nodes["ct"]:
            dac_list.append(df_pivot.loc[cc,"year"])
        dac_input[name] = dac_list
    def annuity_factor(v):
        return calculate_annuity(20, 0.07) + v["FOM"] / 100

    dac_input["fixed"] = [
        annuity_factor(v) * v["investment"]*1E6 for i, v in dac_input.iterrows()
    ]
    filename = f"mofs_{par}/fixed.csv"
    dac_input.to_csv(filename)

def get_clusters(df, tol_T=3, tol_RH=5):
    

    # Assume df has: 'country', 'temp', 'RH', 'T_feed'
    # Step 1: Sort values and collect vectors for each country
    df_sorted = df.sort_values(['country', 'temp'])  # sort to align conditions consistently

    vectors = (
        df_sorted
        .groupby('country')[['RH']]
        .apply(lambda g: g.to_numpy().flatten())
    )
    vectors2 = (
        df_sorted
        .groupby('country')[['T_feed']]
        .apply(lambda g: g.to_numpy().flatten())
    )
    # Step 2: Compare countries pairwise based on full vector
    tol_T = 3
    tol_RH = 5
    pairs = [
        (c1, c2)
        for c1, c2 in itertools.combinations(vectors.index, 2)
        if abs(vectors[c1] - vectors[c2]).max() < tol_RH and abs(vectors2[c1] - vectors2[c2]).max() < tol_T 
    ]




    # Initialize each country as its own parent
    parent = {country: country for country in vectors.index}

    # Union-Find helpers
    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]  # path compression
            c = parent[c]
        return c

    def union(c1, c2):
        root1, root2 = find(c1), find(c2)
        if root1 != root2:
            parent[root2] = root1

    # Build the union-find structure from matched pairs
    for c1, c2 in pairs:
        union(c1, c2)

    # Build clusters from union-find parents
    from collections import defaultdict
    cluster_map = defaultdict(set)
    for country in vectors.index:
        cluster_map[find(country)].add(country)

    # Output unique clusters
    clusters = list(cluster_map.values())
    for i, cl in enumerate(clusters, 1):
        print(f"Cluster {i}: {sorted(cl)}")
    return clusters


df = pd.read_csv("mofs_lewatit.csv").set_index("scenario")
df["country"] = df.index.str.split("_").str[2]
df["temp"] = df.index.str.split("_").str[5]   
clusters = get_clusters(df)
n = pypsa.Network()
n.import_from_netcdf("base_s_90_elec_.nc")
winter = n.snapshots.month.isin([12, 1, 2]).astype(int)
spring = n.snapshots.month.isin([3, 4, 5,]).astype(int)
summer = n.snapshots.month.isin([6, 7 ,8]).astype(int)
autumn = n.snapshots.month.isin([9, 10, 11]).astype(int)
spring_autumn = n.snapshots.month.isin([3, 4, 5, 9, 10, 11]).astype(int)


for par in ["lewatit"]: #["heat_max"]: #"capex_max", "capex", "heat", "heat_max", "el", "el_max"]:
    if not os.path.exists("mofs_"+par):
        os.makedirs("mofs_"+par)
    df = pd.read_csv(f"mofs_{par}.csv").set_index("scenario")
    countries = df.index.str.split("_").str[2]
    df["country"] = [cc[:2] for cc in df.index.str.split("_").str[2]]
    df["temp"] = df.index.str.split("_").str[5]    


    df = df.sort_values(["country","temp"])
    list_cc = []
    for cl in clusters:
        for cc in cl:
            if not df[df["country"]==cc].empty:

                cc_ind = df[df["country"]==cc]
                
                for cc2 in cl:
                    cc_ind_cp = cc_ind.copy()
                    cc_ind_cp.loc[:,"country"] = cc2

                    list_cc.append(cc_ind_cp)
                break

    df = pd.concat(list_cc)

    cp = get_temporal_input("compression-electricity-input")
    hi = get_temporal_input("heat-input")
    ei = get_temporal_input("electricity-input")
    get_input()