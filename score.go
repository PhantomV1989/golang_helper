package main

import (
	"github.com/mpraski/clusters"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// MaxMin of normalized data
type MaxMin struct {
	max float64
	min float64
}

// Normalize to 0,1
func Normalize(embs *mat.Dense) (rembs *mat.Dense, mmArr []MaxMin) {
	// no. of pt x no. of dim
	//returns no. of dim x MaxMin
	_, dim := embs.Dims()
	embT := embs.T()
	for i := 0; i < dim; i++ {
		_r := mat.Row(nil, i, embT)
		mmArr = append(mmArr, MaxMin{max: floats.Max(_r), min: floats.Min(_r)})
	}
	for i := 0; i < dim; i++ {
		_r := mat.Col(nil, i, embs)
		floats.AddConst(-mmArr[i].min, _r)
		rng := mmArr[i].max - mmArr[i].min
		floats.Scale(1/rng, _r)
		embs.SetCol(i, _r)
	}
	return embs, mmArr
}

// GetHierarchicalPC returns 1st PC vector field
func GetHierarchicalPC(nembs [][]float64) {
	c, err := clusters.DBSCAN(5, 0.01, 5, clusters.EuclideanDistance)
	if err != nil {
		panic(err)
	}
	c.Learn(nembs)
	_ = c
	_ = err
	// dataset_cluster_labels = DBSCAN(eps=0.01, min_samples=5).fit_predict(dataset)
	// dataset_clusters = {}
	// for i, v in enumerate(dataset_cluster_labels):
	//     if not v in dataset_clusters:
	//         dataset_clusters[v] = [[i, dataset[i]]]
	//     else:
	//         dataset_clusters[v].append([i, dataset[i]])

	// pc = [[] for i in range(len(dataset))]
	// debug = []
	// for c in dataset_clusters:
	//     clu = dataset_clusters[c]
	//     clu_emb = [x[1] for x in clu]
	//     pcc = get_principle_components(clu_emb, rng, fig=fig)
	//     debug += pcc[1]
	//     pcc = pcc[0]
	//     for i, pos in enumerate([x[0] for x in clu]):
	//         try:
	//             pc[pos] = pcc[i]
	//         except:
	//             pass
	// return np.vstack(pc), debug
}

// GetAnomalyScore calculates anom score
func GetAnomalyScore(point []float64, embs [][]float64, vecs [][]float64) {

}
