import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def _knn_graph(points, k=8, symmetric=True):
    pts = np.asarray(points, float)
    N = len(pts)
    if N == 0:
        return [np.array([], dtype=int) for _ in range(0)], np.empty((0,0))
    d2 = np.sum((pts[:,None,:] - pts[None,:,:])**2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :min(k, max(N-1, 0))]  # (N,k') indices
    if symmetric:
        neigh_sets = [set(nn[i]) for i in range(N)]
        for i in range(N):
            for j in nn[i]:
                neigh_sets[j].add(i)
        neighbors = [np.fromiter(s, dtype=int) for s in neigh_sets]
    else:
        neighbors = [nn[i] for i in range(N)]
    return neighbors, d2

import numpy as np

def _knn_graph_kdtree(points, k=8, symmetric=True, leafsize=64):
    #words:build kNN graph without O(N^2) memory
    from scipy.spatial import cKDTree

    pts = np.asarray(points, float)
    N = pts.shape[0]
    if N == 0:
        return [np.array([], dtype=int) for _ in range(0)]

    kk = int(min(k + 1, N))  #words:include self, then drop it
    tree = cKDTree(pts, leafsize=leafsize)
    d, nn = tree.query(pts, k=kk)      #d shape (N,kk), nn shape (N,kk)

    #words:drop self neighbor in column 0 (usually)
    nn = nn[:, 1:]
    #words:directed neighbor lists
    if not symmetric:
        return [nn[i].astype(int) for i in range(N)]

    #words:symmetrize
    neigh_sets = [set(nn[i]) for i in range(N)]
    for i in range(N):
        for j in nn[i]:
            neigh_sets[int(j)].add(i)

    neighbors = [np.fromiter(s, dtype=int) for s in neigh_sets]
    return neighbors

def knn_groups_fast(points, min_size=3, max_size=6, k=8, seed=0, leafsize=64):
    #words:capacity-constrained region growing on a kNN graph without NxN distances
    assert 1 <= min_size <= max_size
    pts = np.asarray(points, float)
    N = len(pts)
    rng = np.random.default_rng(seed)

    neighbors = _knn_graph_kdtree(pts, k=k, symmetric=True, leafsize=leafsize)

    unassigned = set(range(N))
    labels = -np.ones(N, dtype=int)
    regions = []
    rid = 0

    seeds = np.array(list(unassigned), dtype=int)
    rng.shuffle(seeds)
    seed_idx = 0

    while unassigned:
        while seed_idx < len(seeds) and seeds[seed_idx] not in unassigned:
            seed_idx += 1
        s = int(seeds[seed_idx]) if seed_idx < len(seeds) else int(rng.choice(list(unassigned)))

        region = [s]
        labels[s] = rid
        unassigned.remove(s)

        target = int(rng.integers(min_size, max_size + 1))
        boundary = set(neighbors[s]) & unassigned

        while len(region) < target and boundary:
            b_list = np.array(list(boundary), dtype=int)
            last = region[-1]

            #words:compute squared distances only to boundary candidates (small)
            dv = pts[b_list] - pts[last]
            d2 = np.einsum("ij,ij->i", dv, dv) + 1e-12

            w = 1.0 / d2
            w /= w.sum()
            j = int(rng.choice(b_list, p=w))

            region.append(j)
            labels[j] = rid
            unassigned.remove(j)

            boundary |= (set(neighbors[j]) & unassigned)
            boundary -= set(region)

        regions.append(region)
        rid += 1

    #words:merge undersized regions (same idea as your original)
    for r_id, region in enumerate(regions):
        if len(region) >= min_size:
            continue

        neighbor_rids = set()
        for i in region:
            for j in neighbors[i]:
                lj = labels[j]
                if lj != r_id and lj != -1:
                    neighbor_rids.add(int(lj))

        if not neighbor_rids:
            others = np.where(labels != r_id)[0]
            #words:merge to nearest outside by point distance
            dv = pts[others] - pts[region[0]]
            j = int(others[np.argmin(np.einsum("ij,ij->i", dv, dv))])
            neighbor_rids = {int(labels[j])}

        cent = pts[region].mean(axis=0)
        best_r = min(neighbor_rids, key=lambda rr: np.sum((pts[regions[rr]].mean(axis=0) - cent) ** 2))

        for i in region:
            labels[i] = best_r
        regions[r_id] = []

    unique = np.unique(labels)
    remap = {old: i for i, old in enumerate(unique)}
    labels = np.array([remap[int(x)] for x in labels], dtype=int)
    K = labels.max() + 1
    new_regions = [np.where(labels == i)[0].tolist() for i in range(K)]

    return labels, new_regions


def circle_groups(points, radius=None, target_n=5, seed=0, return_centers=False):
    #purpose: assign points to uniformly sized circles on a hex grid; outside points → nan
    #inputs:
    # points: (N,2) array
    # radius: circle radius; if None, auto-choose using bbox density and target_n pts/circle
    # target_n: desired pts per circle when radius is None
    # return_centers: if True, also return (centers, radius)
    #outputs:
    # labels: (N,) float array; region ids 0..K-1, or nan if outside all circles
    # regions: list of index lists per circle (empties allowed if a circle captures no points)
    # optionally centers, radius

    pts = np.asarray(points, float)
    assert pts.ndim==2 and pts.shape[1]==2
    N = len(pts)
    if N == 0:
        return np.full(0, np.nan), [], (np.empty((0,2)), 0.0) if return_centers else (np.full(0, np.nan), [])

    #bbox
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    w = max(xmax - xmin, 1e-9)
    h = max(ymax - ymin, 1e-9)

    #auto radius from bbox density if not provided
    if radius is None:
        area = w * h
        rho = N / max(area, 1e-12)                 #pts per unit area
        radius = np.sqrt(max(target_n,1) / (np.pi * max(rho, 1e-12)))

    r = float(radius)
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r

    #generate hex-grid centers covering bbox with margin r
    xs = np.arange(xmin - r, xmax + r + 1e-12, dx)
    ys = np.arange(ymin - r, ymax + r + 1e-12, dy)
    centers = []
    for iy, y in enumerate(ys):
        xoffset = 0.0 if (iy % 2 == 0) else r
        for x in xs:
            centers.append((x + xoffset, y))
    centers = np.asarray(centers, float)
    M = len(centers)
    if M == 0:
        labels = np.full(N, np.nan)
        return (labels, [], (centers, r)) if return_centers else (labels, [])

    #assign each point to nearest center; outside radius → nan
    d2 = np.sum((pts[:, None, :] - centers[None, :, :])**2, axis=2)   #(N,M)
    jmin = np.argmin(d2, axis=1)                                      #(N,)
    d2min = d2[np.arange(N), jmin]
    inside = d2min <= r*r

    labels = np.full(N, np.nan, float)
    labels[inside] = jmin[inside].astype(float)

    #build regions as index lists per center id
    K = M
    regions = [np.where(labels == i)[0].tolist() for i in range(K)]

    return (labels, regions, (centers, r)) if return_centers else (labels, regions)


def knn_groups(points, min_size=3, max_size=6, k=8, seed=0):
    """
    purpose:
        partition N 2D/ND points into compact, contiguous groups by growing regions
        over a symmetric k-nearest-neighbor (kNN) graph until each group reaches a
        target size in [min_size, max_size].

    inputs:
        points  : array-like shape (N, D). N points in D dimensions.
        min_size: int >=1. minimum group size when growing a region.
        max_size: int >=min_size. maximum group size target when growing a region.
        k       : int >=1. number of nearest neighbors per node used to build the kNN graph.
        seed    : int. random seed for reproducible shuffling/choices.

    outputs:
        labels  : np.ndarray shape (N,), dtype=int. for each point i, labels[i] is the
                  region id (0..K-1) that point belongs to (K is number of final regions).
        regions : list[list[int]]. regions[r] is the list of point indices assigned to
                  region r. this is just an index view consistent with `labels`.

    notes&caveats:
        - kNN graph is built by brute-force O(N^2). for large N, replace with cKDTree.
        - neighbor sets are made symmetric and then converted to arrays; order is not guaranteed.
          if you need deterministic neighbor order, sort each neighbor array.
        - undersized regions (rare) are merged into adjacent regions by centroid proximity.
    """

    #validate
    assert 1 <= min_size <= max_size
    pts = np.asarray(points, float)
    N = len(pts)
    rng = np.random.default_rng(seed)

    #bruteforce pairwise squared distances; for big N, consider scipy.spatial.cKDTree
    d2 = np.sum((pts[:, None, :] - pts[None, :, :])**2, axis=2)     #shape(N,N)
    np.fill_diagonal(d2, np.inf)                                    #no self-neighbors

    #take k nearest neighbors per node (directed)
    nn = np.argsort(d2, axis=1)[:, :k]                              #shape(N,k) int

    #make graph symmetric: i->j or j->i implies undirected edge {i,j}
    neighbors = [set(nn[i]) for i in range(N)]                      #list of sets
    for i in range(N):
        for j in nn[i]:
            neighbors[j].add(i)

    #optional: convert to arrays (order arbitrary); keep as sets if you prefer
    neighbors = [np.fromiter(s, dtype=int) for s in neighbors]      #list of (deg_i,) arrays

    #bookkeeping for assignment
    unassigned = set(range(N))                                      #nodes not yet labeled
    labels = -np.ones(N, dtype=int)                                 #-1 means unassigned
    regions = []                                                    #will hold lists of indices
    rid = 0                                                         #running region id

    #seed selection: pre-shuffle all indices, then walk that list to pick seeds
    seeds = np.array(list(unassigned))
    rng.shuffle(seeds)
    seed_idx = 0

    #grow regions until all nodes are assigned
    while unassigned:
        #pick next unassigned seed deterministically from shuffled order
        while seed_idx < len(seeds) and seeds[seed_idx] not in unassigned:
            seed_idx += 1
        if seed_idx == len(seeds):
            #fallback if all shuffled seeds already taken (should be rare)
            s = rng.choice(list(unassigned))
        else:
            s = seeds[seed_idx]

        #initialize region with the seed
        region = [s]
        labels[s] = rid
        unassigned.remove(s)

        #pick a random target size in [min_size, max_size] for this region
        target = rng.integers(min_size, max_size + 1)

        #boundary: unassigned neighbors of the current region frontier
        boundary = set(neighbors[s]) & unassigned

        #region-growing: bias pick toward closer nodes to keep shapes compact
        while len(region) < target and boundary:
            b_list = np.array(list(boundary))
            last = region[-1]                                       #grow from the newest node
            w = 1.0 / (d2[last, b_list] + 1e-12)                    #inverse-distance weights
            w /= w.sum()
            j = rng.choice(b_list, p=w)                             #probabilistic nearest-first
            region.append(j)
            labels[j] = rid
            unassigned.remove(j)
            #update boundary: add neighbors of j that are still unassigned; remove anything already in region
            boundary |= (set(neighbors[j]) & unassigned)
            boundary -= set(region)

        #store this region (may be <min_size if boxed in; merged later)
        regions.append(region)
        rid += 1

    #postprocess: merge undersized regions into adjacent ones via graph contact
    for r_id, region in enumerate(regions):
        if len(region) >= min_size:
            continue

        #collect neighboring region ids via graph edges
        neighbor_rids = set()
        for i in region:
            for j in neighbors[i]:
                if labels[j] != r_id and labels[j] != -1:
                    neighbor_rids.add(labels[j])

        if not neighbor_rids:
            #isolated case: merge to geographically nearest other region by nearest point
            others = np.where(labels != r_id)[0]
            #find the single nearest outside point to any member of this region
            j = others[np.argmin(np.sum((pts[region][:, None, :] - pts[others][None, :, :])**2, axis=2))]
            neighbor_rids = {labels[j]}

        #choose the closest neighboring region by centroid distance
        cent = pts[region].mean(axis=0)
        best_r = min(neighbor_rids, key=lambda rr: np.sum((pts[regions[rr]].mean(axis=0) - cent)**2))

        #relabel all members of the undersized region to the chosen region
        for i in region:
            labels[i] = best_r
        regions[r_id] = []  #mark as emptied

    #compact labels to 0..K-1 and rebuild regions accordingly
    unique = np.unique(labels)
    remap = {old: i for i, old in enumerate(unique)}
    labels = np.array([remap[x] for x in labels], dtype=int)
    K = labels.max() + 1
    new_regions = [np.where(labels == i)[0].tolist() for i in range(K)]

    return labels, new_regions

def plot_afferent_regions(points, labels, neighbors=None, show_edges=False,
                          outline_boundaries=True, annotate=True, s=20,
                          seed=0, ax=None):
    #expects: points shape (N,2); labels shape (N,) with ints or NaNs for unassigned
    pts = np.asarray(points, float)
    lab = np.asarray(labels, float)   #allow NaNs
    assert pts.ndim==2 and pts.shape[1]==2 and lab.shape[0]==pts.shape[0]
    finite = np.isfinite(lab)         #assigned points

    #setup axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    else:
        fig = ax.figure

    #color mapping per region id (deterministic but shuffled)
    rids = np.unique(lab[finite]).astype(int)   #actual region ids present
    K = len(rids)
    rng = np.random.default_rng(seed)
    base_cmap = plt.get_cmap('tab20')
    palette = np.vstack([base_cmap(np.linspace(0, 1, 20)) for _ in range((K//20)+1)])[:K, :3]
    rng.shuffle(palette)

    #scatter points by region
    for idx, rid in enumerate(rids):
        mask = (lab == rid)
        if not np.any(mask):
            continue
        ax.scatter(pts[mask,0], pts[mask,1], s=s, c=[palette[idx]],
                   label=f"R{rid} (n={int(mask.sum())})", alpha=0.9)

    #optional: draw unassigned points (NaN labels)
    if np.any(~finite):
        ax.scatter(pts[~finite,0], pts[~finite,1], s=s, c='0.7', label='unassigned', alpha=0.6)

    #optional: show intra-region edges (kNN) as faint segments
    if show_edges:
        if neighbors is None:
            neighbors, _ = _knn_graph(pts, k=min(8, max(2, min(pts.shape[0]-1, 8))))
        segs = []
        for i, nbrs in enumerate(neighbors):
            if not finite[i]:
                continue
            for j in nbrs:
                if j > i and finite[j] and (lab[i] == lab[j]):
                    segs.append([pts[i], pts[j]])
        if segs:
            lc = LineCollection(segs, linewidths=0.5, alpha=0.25)
            ax.add_collection(lc)

    #optional: outline boundaries via edges crossing labels
    if outline_boundaries:
        if neighbors is None:
            neighbors, _ = _knn_graph(pts, k=min(8, max(2, min(pts.shape[0]-1, 8))))
        boundary_segs = []
        for i, nbrs in enumerate(neighbors):
            if not finite[i]:
                continue
            for j in nbrs:
                if j > i and finite[j] and (lab[i] != lab[j]):
                    boundary_segs.append([pts[i], pts[j]])
        if boundary_segs:
            lc_b = LineCollection(boundary_segs, linewidths=1.2, alpha=0.8, colors='k')
            ax.add_collection(lc_b)

    #optional: annotate region centroids
    if annotate:
        for rid in rids:
            mask = (lab == rid)
            if not np.any(mask):
                continue
            c = pts[mask].mean(axis=0)
            ax.text(c[0], c[1], f"{rid}\n(n={int(mask.sum())})",
                    ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    #formatting
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('afferent regions (contiguous, capacity-constrained)')
    if K <= 12:
        ax.legend(loc='best', fontsize=8, frameon=False)
    plt.tight_layout()
    return fig, ax


