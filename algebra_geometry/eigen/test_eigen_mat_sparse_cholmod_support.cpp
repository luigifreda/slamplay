#include <iostream>

#define EIGEN_MPL2_ONLY // this is for being sure of picking just the MPL2 licensed parts

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <Eigen/SparseCore>

using namespace std;

typedef Eigen::Triplet<double> T; // A triplet is a simple object representing a non-zero entry as the triplet: row index, column index, value.

void insertCoefficientLaplaceEquation(int id, int i, int j, double w, std::vector<T>& coeffs,
                       Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
{
    int n = boundary.size();
    int id1 = i + j*n;
    if (i == -1 || i == n) b(id) -= w * boundary(j); // constrained coefficient
    else if (j == -1 || j == n) b(id) -= w * boundary(i); // constrained coefficient
    else coeffs.push_back(T(id, id1, w)); // unknown coefficient
}

void buildProblemLaplaceEquation(std::vector<T>& coefficients, Eigen::VectorXd& b, int n)
{
    b.setZero();
    Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0, M_PI).sin().pow(2);
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            int id = i + j*n;
            insertCoefficientLaplaceEquation(id, i - 1, j, -1, coefficients, b, boundary);
            insertCoefficientLaplaceEquation(id, i + 1, j, -1, coefficients, b, boundary);
            insertCoefficientLaplaceEquation(id, i, j - 1, -1, coefficients, b, boundary);
            insertCoefficientLaplaceEquation(id, i, j + 1, -1, coefficients, b, boundary);
            insertCoefficientLaplaceEquation(id, i, j, 4, coefficients, b, boundary);
        }
    }
}

//In the main function, we declare a list coefficients of triplets (as a std vector) 
//and the right hand side vector $ b $ which are filled by the buildProblem function. 
//The raw and flat list of non-zero entries is then converted to a true SparseMatrix object A. 
//Note that the elements of the list do not have to be sorted, and possible duplicate entries will be summed up.
void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n)
{
    buildProblemLaplaceEquation(coefficients, b, n);
}

int main()
{
    int n = 300; // size of the image
    int m = n*n; // number of unknows (=number of pixels)

    // Assembly:
    std::vector<T> coefficients; // list of non-zeros coefficients
    Eigen::VectorXd b(m); // the right hand side-vector resulting from the constraints
    buildProblem(coefficients, b, n); // prepare the coefficients as a list of triplets 

    cout << "list of coefficients set" << endl; 
    
    // ...
    Eigen::SparseMatrix<double> A(m, m); // declares a column-major sparse matrix type of double
    // fill A
    A.setFromTriplets(coefficients.begin(), coefficients.end());
    
    cout << "sparse mat set" << endl; 

    Eigen::VectorXd x;

    // solve Ax = b
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double> > solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success)
    {
        cout << "decomposition failed " << endl; 
        // decomposition failed
        return -1;
    }
    x = solver.solve(b);
    if (solver.info() != Eigen::Success)
    {
        cout << "solving failed " << endl; 
        // solving failed
        return -1;
    }
    cout << "solving completed" << endl;
}