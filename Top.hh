#ifndef _TOP_HH_
#define _TOP_HH_

#include "problem.hh"
#include "RedSVD.hh"
#include <ctime>

class Top
{
    int d;
    val C;
    val tol;
    val alpha;
    val beta;
    int pcgIter;
    mat F;
public:
    Top(int d, val C, val tol, val alpha, val beta, int pcgIter);
    bool train(const Entity& e1, const Entity& e2, const Relation& r);
    bool predict(const Entity& e1, const Entity& e2, const Relation& r, const char* output);
private:
    /*
     * optimization objective:
     *      C*\|I.*(Y-F)\|_2^2 + 0.5*\vec(F)A\vec(F)
     * where A is the adjacency matrix of the product graph
     */
    val objective(
            const mat& F,
            const Relation& r,
            const mat& U,
            const mat& V,
            const mat& Sigma,
            val C);
    /*
     * gradient of the objective:
     */
    mat gradient(
            const mat& F,
            const Relation& r,
            const mat& U,
            const mat& V,
            const mat& Sigma,
            val C);
    /*
     * the Hessian defines a mapping from F to some output matrix
     */
    mat hessian_map(
            const mat& F,
            const Relation& r,
            const mat& U,
            const mat& V,
            const mat& Sigma,
            val C);
    /*
     * matrix-free conjugate gradient method for solving the linear system
     *      A\vec(X) = \vec(B)
     * where
     *      A = C*I + \sum \tau_{ij} (u_i \otimes v_j)(u_i \otimes v_j)^\top
     */
    mat matrix_pcg(
            const Relation& r,
            const mat& U,
            const mat& V,
            const mat& Sigma,
            const mat& F0,
            const mat& B,
            val C,
            int maxIter);
    /*
     * return the symmetrically normalized G 
     */
    sp_mat normalized_graph(const sp_mat& G);
};

#endif
