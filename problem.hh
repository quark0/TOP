#ifndef _PROBLEM_HH_
#define _PROBLEM_HH_

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

typedef double val;
typedef Eigen::SparseMatrix<val, Eigen::RowMajor> sp_mat;
typedef Eigen::Matrix<val, Eigen::Dynamic, Eigen::Dynamic> mat;
typedef Eigen::Triplet<val> TripletF;

class Entity {
public:
    int n;                                // #training examples
    sp_mat G;                             // KNN graph induced from X
    std::unordered_map<int, std::string> id_of;     // given the instance index, return id
    std::unordered_map<std::string, int> index_of;  // given the instance id, return index
    /*
     * load the entity graph from sparse representation
     */
    Entity(const char* filename) {
        assert(loadSparseGraph(filename));
    }
    /*
     * @FORMAT
     * As in LibSVM, each line of the input file should consist of 
     * `instance_id<string> feature_1<int>:value_1<val> feature_2 value_2`
     *
     * A symmetric k-NN graph will be constructed afterwards
     *
     * Note the full kernel matrix is explicitly computed,
     * which might be inappropriate for large-scale problems
     */
    Entity(const char* filename, int k) {
        sp_mat X = load(filename);
        formCosKNNGraph(X, k);
    }
    bool saveSparseGraph(const char* filename);

private:
    void formCosKNNGraph(const sp_mat& X, int k);
    sp_mat load(const char* filename);
    bool loadSparseGraph(const char* filename);
};

class Relation {
public:
    Relation(const char* filename, const Entity& e1, const Entity& e2) {
        assert(load(filename, e1, e2));
    }
    /*
     * @DEFINITION
     * edge := (index_in_entity1, index_in_entity2, strength)
     */
    std::vector<TripletF> edges;
    /*
     * @FORMAT
     * instance_id_for_e1 instance_id_for_e2
     */
private:
    bool load(const char* filename, const Entity& e1, const Entity& e2);
};

#endif
