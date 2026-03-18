# AE-DPC
"""
    cluster = AEDPC(sample_ratio=1, k_min=m, k_max=n)
    labels = np.array(cluster.labels_)
"""
adaptive extended density peak clustering
import numpy as np
import time
import numpy as np
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

class AEDPC:
    def __init__(self, sample_ratio, k_min, k_max, n_jobs=1):
        self.sample_ratio = sample_ratio
        self.k_min = k_min
        self.k_max = k_max
        self.n_jobs = n_jobs
        self.tree_ = None
        self.labels_ = None
        self.clusters_ = None
    def _find_knee_manual(self, distances,i,plot=False):
        grad = np.gradient(distances)
        grad2 = np.gradient(grad)
        curvature = grad2 / (1 + grad**2)**1.5 
        peak_idx = np.argmax(curvature)
        result_idx = max(self.k_min, min(peak_idx, self.k_max))

        return result_idx
    def fit(self, X):
        X = np.array(X)

        if self.sample_ratio < 1:
            n_total = len(X)
            sample_size = int(n_total * self.sample_ratio)
            sampled_indices = np.random.choice(n_total, size=sample_size, replace=False)
            X_sampled = X[sampled_indices]
            self.sample_indices_ = sampled_indices  
        else:
            X_sampled = X
            self.sample_indices_ = np.arange(len(X)) 
        

        self.tree_ = BallTree(X_sampled)
        start_time1 = time.time()
        dists_all, neighbors_all = self.tree_.query(X_sampled, k=self.k_max+1)
        dists_all = dists_all[:, 1:]  
        neighbors_all = neighbors_all[:, 1:]  
        def precompute(i):
            dists = dists_all[i]
            k = self._find_knee_manual(dists,i)
            neighbors = neighbors_all[i, :k].tolist()   
            k_dist = dists[:k]

            mean_dist = np.mean(k_dist) + 1e-10
            sigma = np.std(k_dist) + 1e-10
            density = 1/(mean_dist+sigma)
            return {
                'index': i,
                'k': k,
                'density': density,
                'neighbors': neighbors,
                'neighbor_dists': dict(zip(neighbors, k_dist)),
            }

        precomputed = Parallel(n_jobs=self.n_jobs)(delayed(precompute)(i) for i in range(len(X_sampled)))
        point_info = {p['index']: p for p in precomputed}
        # print(f"阶段1耗时: {time.time()-start_time1:.4f}秒")

        start_time2 = time.time()
        all_densities = np.array([p['density'] for p in precomputed])

        peak_counts = np.zeros(len(X_sampled), dtype=int)
        for i in point_info:
            neighbors = point_info[i]['neighbors']
            peak_idx = neighbors[np.argmax(all_densities[neighbors])]
            peak_counts[peak_idx] += 1

        labels = -np.ones(len(X_sampled), dtype=int)
        clusters = {}
        for peak_idx in np.where(peak_counts > 0)[0]:
            info = point_info[peak_idx]
            clusters[peak_idx] = {
                'points': set(info['neighbors'] + [peak_idx]),
                'p': all_densities[peak_idx],
                'peak': peak_idx,
                'peak_score': peak_counts[peak_idx] * all_densities[peak_idx],
                'k': info['k']
            }
            labels[list(clusters[peak_idx]['points'])] = peak_idx
        # print(f"阶段2耗时: {time.time()-start_time2:.4f}秒")


        start_time3 = time.time()
        self.labels_, self.clusters_ =  labels, clusters
        self.labels_, self.clusters_ = self._peakcompete_clusters(X_sampled, labels, clusters, point_info)



        start_time4 = time.time()
        self.labels_, self.clusters_ = self._grow_clusters(X_sampled, self.labels_, clusters, point_info)
        # print(f"阶段4耗时: {time.time()-start_time4:.4f}秒")


        start_time5 = time.time()
        self.labels_, self.clusters_ = self._merge_clusters(X_sampled, self.labels_, clusters, point_info)
        # print(f"阶段5耗时: {time.time()-start_time5:.4f}秒")



        if self.sample_ratio < 1:
            full_labels = -np.ones(len(X), dtype=int)
            full_labels[self.sample_indices_] = self.labels_
            
            _, nearest_indices = self.tree_.query(X, k=1)
            for i in np.where(full_labels == -1)[0]:
                full_labels[i] = self.labels_[nearest_indices[i][0]]
            
            self.labels_ = full_labels
            self.clusters_ = self._expand_clusters(X, clusters)
        
        return self  

    def _peakcompete_clusters(self, X, labels, clusters, point_info):

        peak_to_cluster = {c['peak']: pid for pid, c in clusters.items()}
        peak_scores = {pid: c['peak_score'] for pid, c in clusters.items()}
        parent = {pid: pid for pid in clusters}
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  
                u = parent[u]
            return u

        changed = True
        while changed:
            changed = False
            
            min_diff = float('inf')
            best_pair = None
            
            active_pids = set(clusters.keys())
            for pid in active_pids:
                peak = clusters[pid]['peak']
                for neighbor in point_info[peak]['neighbors']:
                    if neighbor in peak_to_cluster:
                        other_pid = peak_to_cluster[neighbor]
                        if other_pid in active_pids and other_pid != pid:
                            current_diff = abs(peak_scores[pid] - peak_scores[other_pid])
                            if current_diff < min_diff:
                                min_diff = current_diff
                                best_pair = (pid, other_pid)
            

            if best_pair:
                a, b = best_pair
                root_a, root_b = find(a), find(b)
                
                if root_a != root_b:
                    winner = root_a if peak_scores[root_a] > peak_scores[root_b] else root_b
                    loser = root_b if winner == root_a else root_a
                    
                    clusters[winner]['points'].update(clusters[loser]['points'])
                    w_size = len(clusters[winner]['points'])
                    l_size = len(clusters[loser]['points'])
                    total = w_size + l_size
                    clusters[winner]['p'] = (clusters[winner]['p']*w_size + clusters[loser]['p']*l_size)/total
                    clusters[winner]['peak_score'] = (clusters[winner]['peak_score'] + clusters[loser]['peak_score'])/2
                    
                    parent[loser] = winner
                    peak_to_cluster.pop(clusters[loser]['peak'], None)
                    del clusters[loser]
                    changed = True
        
        labels = np.array([find(l) if l != -1 else -1 for l in labels])
        return labels, clusters

    def _grow_clusters(self, X, labels, clusters, point_info):
        import numpy as np
        from collections import defaultdict

        point_to_clusters = defaultdict(list)
        for pid, cluster in clusters.items():
            for point in cluster['points']:
                point_to_clusters[point].append(pid)

        for p, pids in list(point_to_clusters.items()):
            if len(pids) > 1:
                for pid in pids:
                    clusters[pid]['points'].discard(p)
                labels[p] = -1
                point_to_clusters[p] = []

        cluster_stats = {
            pid: {
                'avg_dist': 1 / (cluster['p'] + 1e-10),
                'members': set(cluster['points'])
            }
            for pid, cluster in clusters.items()
        }


        prev_unassigned = -1
        while True:
            unassigned = np.where(labels == -1)[0]
            if len(unassigned) == prev_unassigned:
                break
            prev_unassigned = len(unassigned)
            
            temp_assignments = {}
            
            for i in unassigned:
                neighbors = point_info[i]['neighbors']
                neighbor_labels = [labels[n] for n in neighbors if labels[n] != -1]
                
                if not neighbor_labels:
                    continue

                label_strength = defaultdict(float)
                for lbl in set(neighbor_labels):
                    label_strength[lbl] = (neighbor_labels.count(lbl) / len(neighbor_labels)) * cluster_stats[lbl]['avg_dist']

                for label, strength in sorted(label_strength.items(), key=lambda x: -x[1]):
                    cluster_members = [n for n in neighbors if labels[n] == label]
                    if not cluster_members:
                        continue
                    
                    min_dist = min(
                        point_info[i]['neighbor_dists'].get(n, float('inf'))
                        for n in cluster_members
                    )
                    
                    if min_dist < (1+strength)*cluster_stats[label]['avg_dist']:
                        temp_assignments[i] = label
                        break

            for i, lbl in temp_assignments.items():
                labels[i] = lbl
                clusters[lbl]['points'].add(i)
                cluster_stats[lbl]['members'].add(i)

        return labels, clusters

    def _merge_clusters(self, X, labels, clusters, point_info):

        labels_np = np.array(labels)
        
        def compute_cluster_props():
            props = {}
            for lbl, cluster in clusters.items():
                peak = cluster['peak']
                props[lbl] = {
                    'peak': peak,
                    'avg_dist': 1/(cluster['p'] + 1e-10), 
                    'score': cluster['peak_score'],
                    'members': list(cluster['points'])  
                }
            return props

        def compute_strength(lbl_a, lbl_b):

            avg_dist_a = cluster_props[lbl_a]['avg_dist']
            avg_dist_b = cluster_props[lbl_b]['avg_dist']
            threshold = min(avg_dist_a, avg_dist_b)
            score_ratio = cluster_props[lbl_a]['score'] / (cluster_props[lbl_b]['score'] + 1e-10)
            density_sim = np.exp(-abs(np.log(score_ratio)))
            
            count = 0
            for p in cluster_props[lbl_a]['members']:

                neighbors = point_info[p]['neighbors'] + [p]
                
                for neighbor in neighbors:
                    if labels[neighbor] == lbl_b:
                        if neighbor == p:
                            dist = 0.0
                        else:
                            dist = point_info[p]['neighbor_dists'].get(neighbor, float('inf'))
                        
                        if dist < threshold:
                            count += 1
            
            return count * density_sim

        changed = True
        while changed:
            changed = False
            cluster_props = compute_cluster_props()
            
            best_strength = -1
            best_pair = None
            
            cluster_list = np.array(list(clusters.keys()))
            n_clusters = len(cluster_list)
            
            for i in range(n_clusters):
                lbl_a = cluster_list[i]
                for j in range(i+1, n_clusters):
                    lbl_b = cluster_list[j]
                    
                    strength = compute_strength(lbl_a, lbl_b)
                    
                    if strength > best_strength:
                        best_strength = strength
                        best_pair = (lbl_a, lbl_b, strength)
            
            if best_pair and best_strength > 0:
                lbl_a, lbl_b, _ = best_pair
                winner = lbl_a if clusters[lbl_a]['peak_score'] >= clusters[lbl_b]['peak_score'] else lbl_b
                loser = lbl_b if winner == lbl_a else lbl_a
                

                loser_mask = (labels_np == loser)
                labels_np[loser_mask] = winner
                labels = labels_np.tolist()  
                
                clusters[winner]['points'].update(clusters[loser]['points'])
                del clusters[loser]
                changed = True
        
        return labels, clusters

    def _expand_clusters(self, X, sampled_clusters):
        full_clusters = {}
        

        for lbl, cluster in sampled_clusters.items():
            mask = (self.labels_ == lbl)
            full_clusters[lbl] = {
                'points': set(np.where(mask)[0]),
                'p': cluster['p'],
                'peak': self.sample_indices_[cluster['peak']],  
                'peak_score': cluster['peak_score'],
                'k': cluster['k']
            }
        
        return full_clusters






