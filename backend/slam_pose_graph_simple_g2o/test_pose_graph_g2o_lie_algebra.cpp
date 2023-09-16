#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <sophus/se3.hpp>

#include "macros.h"

using namespace std;
using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 

/************************************************
* This program demonstrates how to use g2o solver for pose graph optimization.
* sphere.g2o is a Pose graph artificially generated, let's optimize it.
* Although the entire graph can be read directly through the load function, we still implement 
* the reading code ourselves in order to gain a deeper understanding.
* This section uses Lie algebra to express the pose graph, the way of nodes and edges is custom.
***********************************************/

typedef Matrix<double, 6, 6> Matrix6d;

#define APPROXIMATE_JR_WITH_I 1

// Approximation of J_R^-1 with given error
Matrix6d JRInv(const SE3d &e) {
    Matrix6d J;
    J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log());
    J.block(0, 3, 3, 3) = SO3d::hat(e.translation());
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log());
#if APPROXIMATE_JR_WITH_I    
    J = Matrix6d::Identity(); // try Identity if you want
#else     
    J = J *0.5 + Matrix6d::Identity();    
#endif    
    return J;
}

//Vertex of Lie Algebra
typedef Matrix<double, 6, 1> Vector6d;

class VertexSE3LieAlgebra : public g2o::BaseVertex<6, SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; i++) is >> data[i];
        setEstimate(SE3d(
            Quaterniond(data[6], data[3], data[4], data[5]),
            Vector3d(data[0], data[1], data[2])
        ));
        return true; 
    }

    virtual bool write(ostream &os) const override {
        os << id() << " ";
        Quaterniond q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
        return true;
    }

    virtual void setToOriginImpl() override {
        _estimate = SE3d();
    }

    //left multiplication update
    virtual void oplusImpl(const double *update) override {
        Vector6d upd;
        upd << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = SE3d::exp(upd) * _estimate;
    }
};

//The edge of two Lie algebra nodes
class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; i++) is >> data[i];
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        setMeasurement(SE3d(q, Vector3d(data[0], data[1], data[2])));
        for (int i = 0; i < information().rows() && is.good(); i++)
            for (int j = i; j < information().cols() && is.good(); j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    virtual bool write(ostream &os) const override {
        VertexSE3LieAlgebra *v1 = static_cast<VertexSE3LieAlgebra *> (_vertices[0]);
        VertexSE3LieAlgebra *v2 = static_cast<VertexSE3LieAlgebra *> (_vertices[1]);
        os << v1->id() << " " << v2->id() << " ";
        SE3d m = _measurement;
        Eigen::Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        //information matrix
        for (int i = 0; i < information().rows(); i++)
            for (int j = i; j < information().cols(); j++) {
                os << information()(i, j) << " ";
            }
        os << endl;
        return true;
    }

    //The error calculation is consistent with the derivation in the book
    virtual void computeError() override {
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        _error = (_measurement.inverse() * v1.inverse() * v2).log();
    }

    //Jacobian calculation
    virtual void linearizeOplus() override {
        //SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        Matrix6d J = JRInv(SE3d::exp(_error));
        //try to approximate J to I?
        _jacobianOplusXi = -J * v2.inverse().Adj();
        _jacobianOplusXj = J * v2.inverse().Adj();
    }
};

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

    //set g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; // graph model
    optimizer.setAlgorithm(solver); // set up the solver
    optimizer.setVerbose(true); // turn on debug output

    int vertexCnt = 0, edgeCnt = 0; // number of vertices and edges

    vector<VertexSE3LieAlgebra *> vectices;
    vector<EdgeSE3LieAlgebra *> edges;
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            //vertices
            VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            vectices.push_back(v);
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            //SE3-SE3 edge
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int idx1, idx2;//the two associated vertices
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;

    // Because a custom vertex is used and it is not registered with g2o, save yourself here to implement it
    // Pretend to be SE3 vertices and edges so that g2o_viewer can recognize them
    ofstream fout("result_lie.g2o");
    for (VertexSE3LieAlgebra *v:vectices) {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for (EdgeSE3LieAlgebra *e:edges) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();
    return 0;
}
