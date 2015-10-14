using namespace std;

#include <assert.h>
#include "Top.hh"
#include "SimpleIni.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " configuration.ini" << std::endl;
        exit(1);
    }
    /*load the configuration file *.ini*/
    CSimpleIniA ini(true, true, true);
    assert(ini.LoadFile(argv[1]) >= 0);

    /*load graph G on the left*/
    Entity g(ini.GetValue("IO", "G", NULL));
    printf("Nodes in Entity 1:\t%d\n", g.n);

    /*load graph H on the right*/
    Entity h(ini.GetValue("IO", "H", NULL));
    printf("Nodes in Entity 2:\t%d\n", h.n);

    /*load observed cross-graph links for training*/
    Relation trn(ini.GetValue("IO", "linksTrain", NULL), g, h);
    printf("Edges for Training:\t%zd\n", trn.edges.size());

    /*load hold-out cross-graph links for testing*/
    Relation tes(ini.GetValue("IO", "linksTest", NULL), g, h);
    printf("Edges for Testing:\t%zd\n", tes.edges.size());

    /*read algorithmic settings from file*/
    int d = ini.GetLongValue("Model", "d", 5);
    double C = ini.GetDoubleValue("Model", "C", 1e-3);
    int pcgIter = ini.GetLongValue("Optimization", "pcgIter", 15);
    double tol = ini.GetDoubleValue("Optimization", "tol", 1e-3);
    double alpha = ini.GetDoubleValue("Optimization", "alpha", 0.5);
    double beta = ini.GetDoubleValue("Optimization", "beta", 0.5);            

    /*initialize the algorithm*/
    Top top(d, C, tol, alpha, beta, pcgIter);

    /*training*/
    assert(top.train(g, h, trn));

    /*dump the predictions to file*/
    assert(top.predict(g, h, tes, ini.GetValue("IO", "prediction", "prediction.txt")));
}
