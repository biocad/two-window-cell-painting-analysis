#!/usr/bin/env python
# coding: utf-8

# In[3]:


"Code for two-window Cell Painting analysis (48 h vs 8 days)"
"This notebook reproduces the analysis pipeline used in the manuscript:A stability-weighted combined map for Cell Painting across regular and prolonged incubation"
"pre-processing → robust plate normalisation → PCA → kNN graph → Leiden clustering → UMAP,"
"and stability-weighted integration to select one profile per compound."
"**Input**: aggregated per-well feature table."
"**Outputs**: cluster assignments, UMAP coordinates, and summary figures."

import os
import pandas as pd
import numpy as np
import warnings
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import sys
import time
from matplotlib.lines import Line2D

# --- 0. Settings ---
warnings.filterwarnings('ignore')
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=100, frameon=False, facecolor='white')

def _separate_meta_features(df: pd.DataFrame):
    metadata_cols = [col for col in df.columns if col.lower().startswith('metadata_')]
    if not metadata_cols:
        metadata_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    if not metadata_cols:
        raise ValueError("No metadata columns found.")
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns
    feature_cols = [c for c in feature_cols if c not in non_numeric]
    return df[feature_cols], df[metadata_cols]


def load_dataset(filename: str):
    """
    Simple CSV upload without changing Metadata_Name.
    """
    try:
        df = pd.read_csv(filename, index_col=None)
        return df
    except Exception as e:
        print(f"Loading error: {e}")
        return None


def reduce_with_pca(features_df, meta_df, n_components, random_state=42):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    actual_n = min(n_components, features_scaled.shape[0] - 1)
    pca = PCA(n_components=actual_n, random_state=random_state)
    features_low = pca.fit_transform(features_scaled)
    pc_cols = [f'PC{i+1}' for i in range(actual_n)]
    features_low_df = pd.DataFrame(features_low, columns=pc_cols, index=features_df.index)
    return meta_df.join(features_low_df)


def calculate_distances(df_low, metric='euclidean'):
    pc_cols = [c for c in df_low.columns if c.startswith('PC')]
    results = []
    for name, group in df_low.groupby('Metadata_Name'):
        if len(group) < 2:
            continue
        try:
            dist = np.mean(pdist(group[pc_cols], metric=metric))
            results.append({'Metadata_Name': name, 'distance': dist})
        except Exception:
            continue
    return pd.DataFrame(results) if results else pd.DataFrame(columns=['Metadata_Name', 'distance'])


def compare_and_combine(df_short, df_long, metric='euclidean'):
    """
    For each compound, we calculate the intra-compound distances in the short and long windows,
select the window with the smallest spread, and take all the wells in that window.
    """
    dist_s = calculate_distances(df_short, metric)
    dist_l = calculate_distances(df_long, metric)

    if dist_s.empty or dist_l.empty:
        return pd.DataFrame()

    merged = pd.merge(dist_s, dist_l, on='Metadata_Name', suffixes=('_s', '_l'))
    merged['best'] = np.where(merged['distance_s'] <= merged['distance_l'], 'short', 'long')

    dfs = []
    for _, row in merged.iterrows():
        source = df_short if row['best'] == 'short' else df_long
        dfs.append(source[source['Metadata_Name'] == row['Metadata_Name']])

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def aggregate_data(df):
    """
   We aggregate by substance/source/MoA, taking the median of the features.
    """
    potential_keys = ['Metadata_Name', 'Metadata_Source', 'Metadata_MoA']
    group_keys = [col for col in potential_keys if col in df.columns]
    if not group_keys:
        group_keys = [c for c in df.columns if c.lower().startswith('metadata_')]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df.groupby(group_keys)[numeric_cols].median().reset_index()


def filter_by_long_iterations(df, min_long_iters=2):
    """
   We retain only those substances that have >= min_long_iters different iterations in the long incubation (according to Metadata_Source).
    """
    if 'Metadata_Name' not in df.columns:
        print("Warning: no 'Metadata_Name'. Long filter not applied.")
        return df

    source_col = next((c for c in df.columns if c.lower() == 'metadata_source'), None)
    if not source_col:
        print("Warning: no 'Metadata_Source'. Long filter not applied.")
        return df

    mask_long = df[source_col].astype(str).str.contains('l', case=False, na=False)
    df_long = df.loc[mask_long, ['Metadata_Name', source_col]].copy()

    if df_long.empty:
        print("Warning: long subset is empty. Filter not applied.")
        return df

    iters_per_compound = df_long.groupby('Metadata_Name')[source_col].nunique()
    keep_names = iters_per_compound[iters_per_compound >= min_long_iters].index

    print(f"Total substances in the original df: {df['Metadata_Name'].nunique()}")
    print(f"Compounds with >= {min_long_iters} long-iterations: {len(keep_names)}")

    df_filtered = df[df['Metadata_Name'].isin(keep_names)].copy()
    print(f"After filtering by long iterations, there are lines left: {len(df_filtered)}")

    return df_filtered


def process_clustering(df_raw, k, res, metric='euclidean', random_state=42):
    """
   The full cycle: aggregation -> neighbors -> Leiden -> silhouette.
Also returns AnnData and the aggregated df_agg (for UMAP).
    """
    try:
        df_agg = aggregate_data(df_raw)
        if df_agg.empty or len(df_agg) < 2:
            return 0, None, 0, None, None

        pc_cols = [c for c in df_agg.columns if c.startswith('PC')]
        adata = sc.AnnData(df_agg[pc_cols].values,
                           obs=df_agg.drop(columns=pc_cols))

        sc.pp.neighbors(adata,
                        n_neighbors=min(k, adata.n_obs - 1),
                        use_rep='X',
                        metric=metric,
                        random_state=random_state)
        sc.tl.leiden(adata,
                     resolution=res,
                     key_added='clusters',
                     random_state=random_state)

        df_agg['cluster'] = adata.obs['clusters'].values
        n_clust = df_agg['cluster'].nunique()

        if n_clust < 2:
            sil = -1.0
        else:
            sil = silhouette_score(df_agg[pc_cols],
                                   df_agg['cluster'],
                                   metric=metric)

        if 'Metadata_Name' in df_agg.columns:
            split_counts = df_agg.groupby('Metadata_Name')['cluster'].nunique()
            n_split = (split_counts > 1).sum()
        else:
            n_split = 0

        return n_clust, sil, n_split, adata, df_agg

    except Exception as e:
        print(f"Error in processing: {e}")
        return 0, None, 0, None, None


def cluster_and_score(df_agg, k, res, metric='euclidean', random_state=42):
    """
    Fast version for grid search (aggregation already done).
    """
    try:
        if df_agg.empty or len(df_agg) < 2:
            return 0, -1.0, 0

        pc_cols = [c for c in df_agg.columns if c.startswith('PC')]
        adata = sc.AnnData(df_agg[pc_cols].values,
                           obs=df_agg.drop(columns=pc_cols))

        sc.pp.neighbors(adata,
                        n_neighbors=min(k, adata.n_obs - 1),
                        use_rep='X',
                        metric=metric,
                        random_state=random_state)
        sc.tl.leiden(adata,
                     resolution=res,
                     key_added='clusters',
                     random_state=random_state)

        df_agg['cluster'] = adata.obs['clusters'].values
        n_clust = df_agg['cluster'].nunique()

        if n_clust < 2 or n_clust >= len(df_agg):
            sil = -1.0
        else:
            sil = silhouette_score(df_agg[pc_cols],
                                   df_agg['cluster'],
                                   metric=metric)

        if 'Metadata_Name' in df_agg.columns:
            split_counts = df_agg.groupby('Metadata_Name')['cluster'].nunique()
            n_split = (split_counts > 1).sum()
        else:
            n_split = 0

        return n_clust, sil, n_split

    except Exception:
        return 0, -1.0, 0

# =====================================================================
#   VISUALIZATION (UMAP) FOR A READY SET OF PARAMETERS
# =====================================================================

def run_experiment_visual(file_path, params):
    """
   Generates a detailed report on Short / Long / Combined data
and saves UMAP plots with labels and different point shapes:
- Short: circles
- Long: triangles
- Combined: circles (if the source is short) and triangles (if the source is long)
    """
    base_dir = os.path.dirname(file_path)
    results_dir = os.path.join(base_dir, "Result")

    full_df = load_dataset(file_path)
    if full_df is None:
        return

    # Filter by long-incubation quality
    full_df = filter_by_long_iterations(full_df, min_long_iters=2)

    features, meta = _separate_meta_features(full_df)
    seed = params.get('random_state', 42)

    df_pca = reduce_with_pca(features, meta,
                             params['n_comps'],
                             random_state=seed)

    source_col = next((c for c in df_pca.columns
                       if c.lower() == 'metadata_source'), None)
    if not source_col:
        print("Metadata_Source not found, unable to render.")
        return

    df_short = df_pca[df_pca[source_col].str.contains('sh', na=False)].copy()
    df_long  = df_pca[df_pca[source_col].str.contains('l',  na=False)].copy()

    if df_short.empty or df_long.empty:
        print("Short or Long are empty, visualization is not possible.")
        return

    df_combined = compare_and_combine(df_short, df_long,
                                      metric=params['metric'])
    datasets = {'Short': df_short, 'Long': df_long, 'Combined': df_combined}

    print(f"\n--- DETAILED REPORT FOR PARAMETERS {params} ---")
    print(f"{'Dataset':<10} | {'N_Clust':<8} | {'Silhouette':<10} | "
          f"{'Split_Subs':<10} | {'Samples(Agg)':<12}")
    print("-" * 60)

    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
            print(f"Folder for UMAP: {results_dir}")
        except Exception as e:
            print(f"Error creating folder: {e}")
            return

    # Captions on the final images
    pretty_names = {
        'Short':    'HCT116, short incubation',
        'Long':     'HCT116, long incubation',
        'Combined': 'HCT116, Combined dataset'
    }

    for name, df in datasets.items():
        if df.empty:
            continue

        n, s, split, adata, df_agg = process_clustering(
            df,
            params['k'],
            params['res'],
            params['metric'],
            random_state=seed
        )
        n_samples_agg = len(df_agg) if df_agg is not None else 0
        s_str = f"{s:.4f}" if s is not None else "None"
        print(f"{name:<10} | {n:<8} | {s_str:<10} | {split:<10} | {n_samples_agg:<12}")

        if adata is None or n <= 1:
            continue

        try:
            # UMAP
            sc.tl.umap(adata, random_state=seed)

            umap_coords = adata.obsm['X_umap']
            x = umap_coords[:, 0]
            y = umap_coords[:, 1]

            clusters = adata.obs['clusters'].astype(str).values
            unique_clusters = sorted(
                adata.obs['clusters'].astype(str).unique(),
                key=lambda z: int(z) if z.isdigit() else z
            )

            # Source (short/long)
            src_col_obs = next(
                (c for c in adata.obs.columns if c.lower() == 'metadata_source'),
                None
            )

            cmap = plt.get_cmap('tab10')
            color_map = {
                cl: cmap(i % 10)
                for i, cl in enumerate(unique_clusters)
            }

            fig, ax = plt.subplots(figsize=(16, 14))
            fig.subplots_adjust(right=0.8)  

            # ---- points ----
            for cl in unique_clusters:
                mask_cl = (clusters == cl)

                if name == 'Short':
                    ax.scatter(
                        x[mask_cl],
                        y[mask_cl],
                        s=80,
                        marker='o',
                        c=[color_map[cl]],
                        label=f'Cluster {cl}'
                    )

                elif name == 'Long':
                    ax.scatter(
                        x[mask_cl],
                        y[mask_cl],
                        s=80,
                        marker='^',
                        c=[color_map[cl]],
                        label=f'Cluster {cl}'
                    )

                else:  # Combined
                    if src_col_obs is not None:
                        src_vals = adata.obs[src_col_obs].astype(str).values
                        is_short = np.array([("sh" in s.lower()) for s in src_vals])
                        is_long  = np.array([("l" in s.lower())  for s in src_vals])

                        # short-origin – circles
                        mask_short = mask_cl & is_short
                        if mask_short.any():
                            ax.scatter(
                                x[mask_short],
                                y[mask_short],
                                s=80,
                                marker='o',
                                c=[color_map[cl]],
                                label=f'Cluster {cl}'  
                            )

                        # long-origin – triangles
                        mask_long = mask_cl & is_long
                        if mask_long.any():
                            ax.scatter(
                                x[mask_long],
                                y[mask_long],
                                s=80,
                                marker='^',
                                c=[color_map[cl]],
                                label=None  
                            )
                    else:
                        ax.scatter(
                            x[mask_cl],
                            y[mask_cl],
                            s=80,
                            marker='o',
                            c=[color_map[cl]],
                            label=f'Cluster {cl}'
                        )

            # compounds signatures
            if 'Metadata_Name' in adata.obs.columns:
                labels = adata.obs['Metadata_Name'].astype(str).values
                for i in range(adata.n_obs):
                    ax.text(
                        x[i],
                        y[i],
                        labels[i],
                        fontsize=8,
                        alpha=0.8,
                        ha='left',
                        va='bottom'
                    )

             # --- Legend ---
            fig.subplots_adjust(right=0.8)

            # the shape of the marker in the legend must match the picture
            marker_for_legend = '^' if name == 'Long' else 'o'

            # legend by clusters (colors)
            cluster_handles = [
                Line2D(
                    [0], [0],
                    marker=marker_for_legend,
                    linestyle='None',
                    markerfacecolor=color_map[cl],
                    markeredgecolor='none',
                    markersize=8,
                    label=f'Cluster {cl}'
                )
                for cl in unique_clusters
            ]

            if name == 'Combined':
                #additional legend for point shapes (short / long origin)
                shape_handles = [
                    Line2D([0], [0],
                           marker='o',
                           linestyle='None',
                           color='black',
                           markersize=8,
                           label='Short'),
                    Line2D([0], [0],
                           marker='^',
                           linestyle='None',
                           color='black',
                           markersize=8,
                           label='Long'),
                ]

                # First, the legend by cluster (top right)
                leg1 = ax.legend(
                    handles=cluster_handles,
        #            title='Clusters',
                    loc='upper left',
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.,
                    frameon=False
                )

                # then the legend on the shapes of the dots (just below)
                leg2 = ax.legend(
                    handles=shape_handles,
                    title='Incubation',
                    loc='upper left',
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.,
                    frameon=False
                )
                ax.add_artist(leg1)
            else:
                # only legend by clusters (right)
                ax.legend(
                    handles=cluster_handles,
           #         title='Clusters',
                    loc='upper left',
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.,
                    frameon=False
                )

            # Design of headings and axes
            main_title = pretty_names.get(name, name)
            ax.set_title(main_title, fontsize=14)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")

            fname = os.path.join(
                results_dir,
                f"UMAP3_{name}_{params['metric']}_pca{params['n_comps']}"
                f"_res{params['res']:.2f}_k{params['k']}.png"
            )
            fig.savefig(fname, dpi=300)
            plt.close(fig)
            print(f"   > UMAP saved: {fname}")

        except Exception as e:
            print(f"   >Error while building UMAP ({name}): {e}")

    print("\n")



# =====================================================================
#   GRID SEARCH + EXPORT OF THE BEST CONFIGURATION
# =====================================================================

def run_grid_search(file_path):

    base_dir = os.path.dirname(file_path)
    print(f"Base folder for saving results: {base_dir}")

    # Search ranges
    n_comps_range = range(50, 71, 5)
    metrics = ['euclidean', 'cosine', 'correlation']
    k_range = range(5, 31, 5)
    res_range = np.arange(0.2, 3.1, 0.2)

    total_combinations = len(n_comps_range) * len(metrics) * len(k_range) * len(res_range)
    print(f"--- Start Grid Search: {total_combinations} combinations ---")

    # Loading and filtering
    full_df = load_dataset(file_path)
    if full_df is None:
        return

    full_df = filter_by_long_iterations(full_df, min_long_iters=2)
    features_raw, meta_raw = _separate_meta_features(full_df)

    results = []
    counter = 0
    start_time = time.time()
    best_sil_combined = -1.0

    # Basic cycles
    for n_c in n_comps_range:

        df_pca = reduce_with_pca(features_raw, meta_raw, n_c)

        source_col = next((c for c in df_pca.columns if c.lower() == 'metadata_source'), None)
        if not source_col:
            print("Critical Error: Metadata_Source not found.")
            return

        df_short = df_pca[df_pca[source_col].str.contains('sh', na=False)].copy()
        df_long  = df_pca[df_pca[source_col].str.contains('l',  na=False)].copy()

        if df_short.empty or df_long.empty:
            print("Warning: empty short or long after PCA.")
            continue

        df_agg_short = aggregate_data(df_short)
        df_agg_long  = aggregate_data(df_long)

        for met in metrics:

            df_combined = compare_and_combine(df_short, df_long, metric=met)
            df_agg_combined = aggregate_data(df_combined)

            for k in k_range:
                for res in res_range:
                    counter += 1

                    n_cl_c, sil_c, split_c = cluster_and_score(df_agg_combined.copy(), k, res, metric=met)
                    n_cl_s, sil_s, split_s = cluster_and_score(df_agg_short.copy(),    k, res, metric=met)
                    n_cl_l, sil_l, split_l = cluster_and_score(df_agg_long.copy(),     k, res, metric=met)

                    results.append({
                        'n_comps': n_c,
                        'metric': met,
                        'k': k,
                        'res': res,
                        'n_clust_combined': n_cl_c,
                        'sil_combined': sil_c,
                        'split_combined': split_c,
                        'n_clust_short': n_cl_s,
                        'sil_short': sil_s,
                        'split_short': split_s,
                        'n_clust_long': n_cl_l,
                        'sil_long': sil_l,
                        'split_long': split_l
                    })

                    if sil_c is not None and sil_c > best_sil_combined:
                        best_sil_combined = sil_c

                    sys.stdout.write(
                        f"\rProgress: {counter}/{total_combinations} | "
                        f"PCA={n_c} {met[:3]} k={k} r={res:.1f} -> "
                        f"N_comb={n_cl_c} SilC={sil_c:.3f} | Best SilC={best_sil_combined:.3f}"
                    )
                    sys.stdout.flush()

    # ----- grid search final report -----
    print("\n\n" + "=" * 60)
    print(f"GRID SEARCH COMPLETED in {time.time() - start_time:.1f} sec.")
    print("=" * 60)

    if not results:
        print("Grid search did not return any results.")
        return

    results_df = pd.DataFrame(results)
    summary_path = os.path.join(base_dir, "grid_search_summary_test.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"The grid search summary is saved in: {summary_path}")

    # ----- choosing the best configuration -----
    df_valid = results_df.copy()

    # 1) First, we try exactly 7 Combined clusters
    df_valid = df_valid[(df_valid['n_clust_combined'] == 7) &
                        (df_valid['sil_combined'] > 0.0)]

    if df_valid.empty:
        print("There are no suitable configurations for exactly 7 clusters. We're trying a range of 6–15 clusters.")
              
        df_valid = results_df[
            (results_df['n_clust_combined'] >= 6) &
            (results_df['n_clust_combined'] <= 15) &
            (results_df['sil_combined'] > 0.0)
        ].copy()

    if df_valid.empty:
        print("After filters by number of clusters and silhouette, nothing remains. We're using the entire results_df.")
        df_valid = results_df.copy()

    df_valid = df_valid.copy()
    df_valid['score'] = df_valid['sil_combined'] - 0.02 * df_valid['split_combined']

    df_valid = df_valid.sort_values('score', ascending=False)
    print("\nTop 5 configurations by score:")
    print(df_valid[['n_comps','metric','k','res',
                    'n_clust_combined','sil_combined',
                    'split_combined','score']].head(5))

    best_row = df_valid.iloc[0]
    best_params = {
        'n_comps': int(best_row['n_comps']),
        'metric': best_row['metric'],
        'k': int(best_row['k']),
        'res': float(best_row['res']),
        'random_state': 42
    }

    print("\n=== BEST CONFIGURATION BY SCORE ===")
    print(best_params)

       # ----- export clusters for better configuration -----
    full_df = load_dataset(file_path)
    if full_df is None:
        return
    full_df = filter_by_long_iterations(full_df, min_long_iters=2)

    features_raw, meta_raw = _separate_meta_features(full_df)
    df_pca = reduce_with_pca(features_raw, meta_raw,
                             best_params['n_comps'],
                             random_state=best_params['random_state'])

    source_col = next((c for c in df_pca.columns if c.lower() == 'metadata_source'), None)
    if not source_col:
        print("Unable to export clusters: no Metadata_Source.")
        return

    df_short = df_pca[df_pca[source_col].str.contains('sh', na=False)].copy()
    df_long  = df_pca[df_pca[source_col].str.contains('l',  na=False)].copy()

    # short / long for visalization
    short_path = os.path.join(base_dir, "short_for_visual.csv")
    long_path  = os.path.join(base_dir, "long_for_visual.csv")
    df_short.to_csv(short_path, index=False)
    df_long.to_csv(long_path,  index=False)
    print(f"Data short saved in: {short_path}")
    print(f"Data long saved in: {long_path}")

    # We're building Combined using the best metrics.
    df_combined = compare_and_combine(df_short, df_long, metric=best_params['metric'])
    if df_combined.empty:
        print("Combined - subselect is empty, export is not possible.")
    else:
        # We aggregate three sets at the substance level
        df_agg_combined = aggregate_data(df_combined)
        df_agg_short    = aggregate_data(df_short)
        df_agg_long     = aggregate_data(df_long)

        # helper function for clustering and returning df with column 'cluster'
        def add_clusters(df_agg, params):
            pc_cols = [c for c in df_agg.columns if c.startswith('PC')]
            adata = sc.AnnData(df_agg[pc_cols].values,
                               obs=df_agg.drop(columns=pc_cols))
            sc.pp.neighbors(
                adata,
                n_neighbors=min(params['k'], adata.n_obs - 1),
                use_rep='X',
                metric=params['metric'],
                random_state=params['random_state']
            )
            sc.tl.leiden(
                adata,
                resolution=params['res'],
                key_added='clusters',
                random_state=params['random_state']
            )
            df_out = df_agg.copy()
            df_out['cluster'] = adata.obs['clusters'].values
            return df_out

        df_combined_clust = add_clusters(df_agg_combined, best_params)
        df_short_clust    = add_clusters(df_agg_short,    best_params)
        df_long_clust     = add_clusters(df_agg_long,     best_params)

        # general export function (Name, MoA, cluster)
        def export_clusters(df_in, filename):
            cols = [c for c in ['Metadata_Name', 'Metadata_MoA', 'cluster'] if c in df_in.columns]
            df_in[cols].to_excel(filename, index=False)
            print(f"File with clusters saved in: {filename}")

        # Combined
        out_comb = os.path.join(base_dir, "clusters_best_config.xlsx")
        export_clusters(df_combined_clust, out_comb)

        # Short and Long 
        out_short = os.path.join(base_dir, "clusters_short.xlsx")
        out_long  = os.path.join(base_dir, "clusters_long.xlsx")
        export_clusters(df_short_clust, out_short)
        export_clusters(df_long_clust,  out_long)

        # save Combined by wells for visualizations
        out_csv = os.path.join(base_dir, "combined_for_visual.csv")
        df_combined.to_csv(out_csv, index=False)
        print(f"Combined for visualization saved in: {out_csv}")

    # UMAP for Short / Long / Combined
    run_experiment_visual(file_path, best_params)

# ==============================================================
#   RUN
# ==============================================================

if __name__ == "__main__":
    FILE = "INPUT"
    run_grid_search(FILE)

