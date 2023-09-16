#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include "macros.h"

using namespace std;


std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 

/************************************************
* This program demonstrates how to use g2o solver for pose graph optimization.
* sphere.g2o is a Pose graph artificially generated, let's optimize it.
* Although the entire graph can be read directly through the load function, we still implement 
* the reading code ourselves in order to gain a deeper understanding.
* Here, SE3 in g2o/types/slam3d/is used to represent the pose, which is essentially a quaternion 
* rather than a Lie algebra.
***********************************************/

int main(int argc, char **argv) 
{
    std::string g2o_file = dataDir + "/g2o/sphere.g2o";
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " sphere.g2o" << endl;
    }
    if (argc == 2){
        g2o_file = argv[1];
    }
    ifstream fin(g2o_file);
    if (!fin) {
        cout << "file " << g2o_file << " does not exist." << endl;
        return 1;
    }

 
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType; // Jacs are 6x6 
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; 
    optimizer.setAlgorithm(solver); 
    optimizer.setVerbose(true); 

    int vertexCnt = 0, edgeCnt = 0; 
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
 
            g2o::VertexSE3 *v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
 
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1, idx2; 
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;
    optimizer.save("result.g2o");

    return 0;
}