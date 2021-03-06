#include "Top.hh"

Top::Top(int _d, val _C, val _tol, val _alpha, val _beta, int _pcgIter)
    : d(_d), C(_C), tol(_tol), alpha(_alpha), beta(_beta), pcgIter(_pcgIter) {};

val Top::objective(
        const mat& F,
        const Relation& r,
        const mat& U,
        const mat& V,
        const mat& Sigma,
        val C)
{
    int i, j;
    val ell = 0, d;
    for (auto it = r.edges.cbegin(); it != r.edges.cend(); ++it) {
        i = it->row();
        j = it->col();
        d = F(i,j) - it->value();
        ell += d*d;
    }
    /*Presence of the last term is due to the trick for "Cartesian+Diffusion" that we first substract "1" from Sigma*/
    return C*ell +
        0.5*F.cwiseProduct(U*Sigma.cwiseProduct(U.transpose()*F*V)*V.transpose()).sum() + 0.5*F.cwiseProduct(F).sum(); 
}

mat Top::gradient(
        const mat& F,
        const Relation& r,
        const mat& U,
        const mat& V,
        const mat& Sigma,
        val C)
{
    int i, j;
    mat nabla_ell = mat::Zero(F.rows(),F.cols());
    for (auto it = r.edges.cbegin(); it != r.edges.cend(); ++it) {
        i = it->row();
        j = it->col();
        nabla_ell(i,j) += F(i,j) - it->value();
    }
    return 2*C*nabla_ell +
        U*Sigma.cwiseProduct(U.transpose()*F*V)*V.transpose() + F;
}

mat Top::hessian_map(
        const mat& F,
        const Relation& r,
        const mat& U,
        const mat& V,
        const mat& Sigma,
        val C)
{
    int i, j;
    mat gamma = mat::Zero(F.rows(),F.cols());
    for (auto it = r.edges.cbegin(); it != r.edges.cend(); ++it) {
        i = it->row();
        j = it->col();
        gamma(i,j) = F(i,j);
   }
    // 2C*I.*F + ...
    return 2*C*gamma +
        U*Sigma.cwiseProduct(U.transpose()*F*V)*V.transpose() + F;
}

mat Top::matrix_pcg(
        const Relation& r,
        const mat& U,
        const mat& V,
        const mat& Sigma,
        const mat& F0,
        const mat& B,
        val C,
        int maxIter)
{
    mat F = F0;
    mat R = B - hessian_map(F, r, U, V, Sigma, C);
    mat P = R;
    val alpha_num, alpha_den, alpha, beta;

    for (int i = 0; i < maxIter; i++) {
        /*
         *std::cout << R.norm() << std::endl;
         */
        alpha_num = R.squaredNorm();
        mat AP = hessian_map(P, r, U, V, Sigma, C);
        alpha_den = P.cwiseProduct(AP).sum(); 
        alpha = alpha_num/alpha_den;
        F = F + alpha*P;
        R = R - alpha*AP;
        beta = R.squaredNorm()/alpha_num;
        P = R + beta*P;
    }
    return F;
}

sp_mat Top::normalized_graph(const sp_mat& G) {
    mat inv_degree = (G*mat::Ones(G.rows(),1)).cwiseSqrt().cwiseInverse();
    return inv_degree.asDiagonal()*G*inv_degree.asDiagonal();
}

/*XXX: modify as the alternating newton's method*/
bool Top::train(const Entity& e1, const Entity& e2, const Relation& r) {

    RedSVD::RedSVD<sp_mat> svd1(normalized_graph(e1.G),this->d);
    RedSVD::RedSVD<sp_mat> svd2(normalized_graph(e2.G),this->d);

    mat U = svd1.matrixU(); 
    mat V = svd2.matrixU();

    /*Tensor Product Graph*/
    /*
     *mat Kappa = svd1.singularValues()*svd2.singularValues().transpose();
     */

    /*Diffusion Kernel over the Cartesian Product Graph*/
    mat Kappa = (svd1.singularValues().replicate(1,V.cols()) +
            svd2.singularValues().replicate(1,U.cols()).transpose()).array().exp();

    mat Sigma = Kappa.cwiseInverse();
    /*
     *XXX Since exp(0) = 1, we
     * 1. Set the exponentiated zero entries (the "ones") in Sigma to zero
     * 2. Append a regularization term to compensate in the objective function
     */
    Sigma.array() -= 1;

    F = mat::Zero(e1.n,e2.n);
    std::clock_t start;
    int i = 0;
    val t;
    val conv;
    val obj_new, obj_old = objective(F, r, U, V, Sigma, C);
    while (true) {
        start = std::clock();
        /*compute the gradient*/
        mat nabla_F = gradient(F, r, U, V, Sigma, C);
        /*compute newton direction*/
        mat delta_F = matrix_pcg(r, U, V, Sigma, F, nabla_F, C, pcgIter);
        /*backtracking for the dumped Newton step*/
        t = 1;
        while (objective(F - t*delta_F, r, U, V, Sigma, C) >
          obj_old - alpha*t*nabla_F.cwiseProduct(delta_F).sum())
            t *= beta;
        F -= t*delta_F;
        /*info disp and workflow control*/
        obj_new = objective(F, r, U, V, Sigma, C);
        conv = (obj_old - obj_new) / obj_old;
        val elapse = (std::clock() - start) / (double) CLOCKS_PER_SEC;
        printf("Newton %2d, ", ++i);
        printf("obj: %e, conv: %e, elapse: %f\n" , obj_new, conv, elapse);
        if (conv < tol)
            break;
        obj_old = obj_new;
    }
    return true;
}

bool Top::predict(const Entity& e1, const Entity& e2, const Relation& r, const char* output) {
    std::ofstream ofs(output);
    if( !ofs.fail() ) {
        int i, j;
        val mse = 0, delta;
        /*dump the predictions on test set for evaluation*/
        for (auto it = r.edges.cbegin(); it != r.edges.cend(); ++it) {
            i = it->row();
            j = it->col();
            delta = it->value() - F(i,j);
            ofs << e1.id_of.at(i) << ' '
                << e2.id_of.at(j) << ' '
                << F(i,j) << std::endl;
            mse += delta*delta;
        }
        ofs.close();
        mse /= r.edges.size();
        std::cout << "mse = " << mse << std::endl;
        return true;
    }
    return false;
}
