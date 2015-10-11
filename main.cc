using namespace std;

#include <assert.h>
#include "Top.hh"
#include "SimpleIni.h"

int main(int argc, char *argv[]) {
    /*load the configuration file*/
    CSimpleIniA ini(true, true, true);
    assert(ini.LoadFile("cfg.ini") >= 0);

    Entity e1(ini.GetValue("io", "entity1", NULL));
    Entity e2(ini.GetValue("io", "entity1", NULL));

    printf("Nodes in Entity 1:\t%d\n", e1.n);
    printf("Nodes in Entity 2:\t%d\n", e2.n);

    Relation r_trn(ini.GetValue("io", "trainLinks", NULL), e1, e2);
    Relation r_tes(ini.GetValue("io", "testLinks", NULL), e1, e2);

    printf("Edges for Training:\t%zd\n", r_trn.edges.size());
    printf("Edges for Testing:\t%zd\n", r_tes.edges.size());

    Top top(ini.GetLongValue("hyper", "d", 5),
            ini.GetDoubleValue("hyper", "C", 1e-3),
            ini.GetDoubleValue("opt", "tol", 1e-3),
            ini.GetDoubleValue("opt", "alpha", 0.5),
            ini.GetDoubleValue("opt", "beta", 0.5),
            ini.GetLongValue("opt", "pcgIter", 15));
    /*training*/
    assert(top.train(e1, e2, r_trn));
    /*prediction*/
    assert(top.predict(e1, e2, r_tes, ini.GetValue("io", "output", "output.log")));
}
