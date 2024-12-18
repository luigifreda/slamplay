// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include "sophus/se3.hpp"

#include <chrono>
#include <random>

#include <pangolin/pangolin.h>
#include <pangolin/display/default_font.h>
#include <unistd.h>

#include <macros.h>

/*
===============================================
> Short problem description: 
===============================================

Align a set of points {p1i} with a set of points {p2i} where i=1,2,...,N via an unknown transformation q \in SE(E). 
Note that p1i, p2i \in IR^3 for i=1,2,...,N
We assume data association has been already solved:
p1i <-> p2i  for i=1,2,...,N 

===============================================
> Preliminaries 
===============================================

a x b = [a]x b    cross product 

a x b = | i,  j,  k| = [ ay*bz - az*by] = [  0, -az,  ay]*[bx]
        |ax, ay, az|   [-ax*bz + az*bx]   [ az,   0, -ax] [by] 
        |bx, by, bz|   [ ax*by - ay*bx]   [-ay,  ax,   0] [bz]

[a]x = [  0, -az,  ay]
       [ az,   0, -ax]
       [-ay,  ax,   0]       

Given an angle-axis rotation vector v = u*theta  where u is unit vector \in IR^3 and theta an angle \in S^1.  
The corresponding rotation matrix R(v) \in SO(3):
R(v) = exp([v]x) = exp(theta*[u]x) ~= I + [v]x = I + theta*[u]x
exp(A) = \sum_k A^k/k!
One has: 
v = log(R(v))  where log(.) is the inverse operation of exp(.) over matrices.

From the approximate BCH formulas on SO(3) Lie algebra: 
exp([dv]x + [v]x) ~= exp([Jl(v)*dv]x)*exp([v]x)
where v, dv \in IR^3 are rotation vectors, and Jl(v) \in IR^{3x3} is the left Jacobian of SO(3): 
Jl(v) = sin(theta)/theta * I3 + (1-sin(theta)/theta)) * u*u^T + (1-cos(theta))/theta * [u]x
where, as above, the rotation vector v = u*theta with u unit vector \in IR^3, theta is an angle \in S^1,
and I3 is the 3x3 identity matrix. 
   
See a proper book for an introduction about exp(A), log(R(v)), BCH formulas and left Jacobian of SO(3). For instance: 
"State Estimation for Robotics" by Prof. Barfoot.

===============================================
> Alignment of 3D points via iterative approach 
===============================================

robot frame = {x front, y left, z up}  (ROS convention)
robot pose = q = [x,y,z, v]^T  \in SE(3)  where [x,y,z] \in IR^3 and v \in IR^3 is 
                                          an angle-axis representation of a rotation \in SO(3).

We are given two set of points p1 and p2, where p1i \in IR^3 and p2j \in IR^3.
We need to align p2 to p1. 
For the sake of simplicity, assume the data association has been already solved, that is: p1i <-> p2i

We consider a cartesian noise model. 

p1i = [x] + R * p2i + [epsix]   \in IR^3
      [y]             [epsiy]
      [z]             [epsiz]

epsi = [epsix] ~ N(0,sigmap)
       [epsiy]
       [epsiz]
and the covariance matrix is: 
sigmap = [sigmar^2,        0,           0]   
         [       0, sigmar^2,           0]  \in IR^{3x3} 
         [       0,        0,    sigmar^2]
where sigmar \in IR

Note that: 
    ei = (R*p2i+t-p1i) ~ N(0,sigmap;q)
where: 
R = R(v) \in SO(3)
t = [x]  \in IR^3
    [y]
    [z]

In long vector form:
 e = [e1,e2,...,eN] \in IR^{2*Nx1}
 
Measurement model. Assuming: 
 (i) points p1 belongs to the map m, and points p2 are new observation of p1  
(ii) independent noise on each observed point p2i = zi 
we have that: 
p(e|q, map) = \prod_i p(ei|q, map) = \prod_i N(R*p2i+t-p1i|0,sigmap;q)

We have to register the new scan to the map. We use a MLE estimation approach. 

q* = arg min_{q} NLL = arg min_{q} \prod_i log(N(R*p2i+t-p1i|0,sigmap;q)) 
                     = arg min_{q} \sum_i (R*p2i+t-p1i)^T * sigmap^-1 * (R*p2i+t-p1i) 

ei = (R*p2i+t-p1i)
L = chi2 = 1/2 * \sum_i (R*p2i+t-p1i)^T * sigmap^-1 * (R*p2i+t-p1i) = 1/2 * \sum_i ei^T * sigmap^-1 * ei

We can (left) perturb ei as follows: 
ei = (R*p2i+t-p1i) = [R(deltav) * R(v)] * p2i + [x + deltax] - p1i 
                                                [y + deltay]
                                                [z + deltaz]
where R(deltav) = exp([deltav]x) \in SO(3) as explained above (in the "Preliminary" section), and                                             
deltaq = [deltax]   \in SE(3) where [deltax] \in IR^3 and deltav \in IR^3 is a angle-axis representation of a rotatio in SO(3)
         [deltay]                   [deltay]
         [deltaz]                   [deltaz]
         [deltav]

In order to optimize L, we use an iterative approach and enforce dL/d(deltaq) = 0 at each step. 
Namely, we can approx:
    ei(q ■ deltaq) = e0i + Ji * deltaq
where e0i = ei(q) \in IR^3 and Ji = dei/d(deltaq) \in IR^{3x6}

Note that:
1) Ji = dei/d(deltaq) must be elaluated at the same q where e0i = ei(q) is evalutated
2) q ■ deltaq implies a proper composition where:
x <- x+ deltax
y <- y+ deltay
z <- z+ deltaz
v <- log(R(deltav)*R(v))

=> L = 1/2 * \sum_i (e0i + Ji * deltaq)^T * sigmap^-1 * (e0i + Ji * deltaq)

dL/d(deltaq) = d/d(deltaq) [1/2 * \sum_i (e0i + Ji * deltaq)^T * sigmap^-1 * (e0i + Ji * deltaq)] = 
= \sum_i (e0i + Ji * deltaq)^T * sigmap^-1 *Ji  = 0

Then, we obtain the following normal equations: 
\sum_i (Ji * sigmap^-1 *Ji) * deltaq = -\sum_i Ji^T * sigmap^-1 *e0i
In compact form: 
    H * deltaq = b
where 
    H =  \sum_i (Ji * sigmap^-1 *Ji)   \in IR^{6x6}
    b = -\sum_i Ji^T * sigmap^-1 *e0i  \in IR^{6x1}

ei(q) = (R*p2i+t-p1i) = R(v) * p2i + [x] - p1i 
                                     [y]
                                     [z]

Let's now compute and analyse the perturbation model with deltaq = (deltax, deltay, deltaz, deltav):
ei(q ■ deltaq) = (R*p2i+t-p1i) = [R(deltav) * R(v)] * p2i + [x + deltax] - p1i 
                                                            [y + deltay] 
                                                            [z + deltaz] 

ei ~= ei(deltaq=0) + Ji * deltaq ~= (I + [deltav]x)* R(v) * p2i + [x] + [deltax] - p1i
                                                                  [y]   [deltay]
                                                                  [z]   [deltaz]
    = (R(v) * p2i + [x] - p1i) +  [deltav]x * R(v) * p2i + [deltax] = 
                    [y]                                    [deltay]
                    [z]                                    [deltaz]
    = (R(v) * p2i + [x] - p1i)  - [R(v) * p2i]x * deltav  + [deltax] = 
                    [y]                                     [deltay]
                    [z]                                     [deltaz]

Ji \in IR^{3x6} can be easily found from the last ei equation: 
Ji = dei/d(deltaq) = [1, 0, 0 |              ] 
                     [0, 1, 0 | -[R(v)*p2i]x ]
                     [0, 0, 1 |              ]

Once the solution to the following system is found 
\sum_i Ji * sigmap^-1 *Ji * deltaq = -\sum_i Ji^T * sigmap^-1 *e0i
then we update: 
    [x    ] += deltax
    [y    ] += deltay
    [z    ] += deltaz    
    [v    ] = log(R(deltav)*R(v))

In matrix form:
    e = [e1,e2,...,eN] \in IR^{3Nx1}
    J = d(e)/d(deltaq) \in IR^{3Nx6} 
    J^T * sigmap^-1 * J * deltaq = -J^T *sigmap^-1 * e
    H * deltaq = b
    H =  J^T * sigmap^-1 * J   \in IR^{6x6}
    b = -J^T * sigmap^-1 * e   \in IR^{6x1}


===============================================
> Closed-form alignment of points via SVD  
===============================================

ei(q)  = [x] + R(v) * [p2ix] - [p1ix]   
         [y]          [p2iy]   [p1iy]    
         [z]          [p2iz]   [p1iz]    

ei(q+dq) = [x+dx] + exp([v+dv]x) * [p2ix] - [p1ix] =  
           [y+dy]                  [p2iy]   [p1iy]           
           [z+dz]                  [p2iz]   [p1iz]         
(by using the above approximate BCH formulas)
~= [x+dx] + exp([Jl(v)*dv]x) * exp([v]x) * [p2ix] - [p1ix] =  
   [y+dy]                                  [p2iy]   [p1iy]           
   [z+dz]                                  [p2iz]   [p1iz]  
~= [x+dx] + (I + [Jl(v)*dv]x) * R(v) * p2i - p1i 
   [y+dy]                                  
   [z+dz]                                
~= [x] + R(v) * p2i - p1i + [dx] -[R(v)*p2i]x*Jl(v)* v 
   [y]                      [dy]
   [z]                      [dz]  

Therefore, one has: 
dei/dq   = [1,0,0 |                    ] 
           [0,1,0 | -[R(v)*p2i]x*Jl(v) ]
           [0,1,0 |                    ]

lamdap = 1/(sigmap*sigmap)
lamda = sigmap^-1 = [lamdap,     0,       ] = I3*lamdap where I3 is the 3x3 identity matrix and lamdap \in IR.
                    [     0,lamdap,       ]                          
                    [     0,     0, lamdap]
L =  1/2 * \sum_i ei^T * lamda * ei
dL/dq = 0  => \sum_i ei^T * lamda * dei/dq = 0      [ (1x3)*(3x3)*(3x6) ]


If we just considider the derivative w.r.t. (x, y, z): 
\sum_i ei^T * lamda * dei/d(x,y,z) = 0   =>  \sum_i ei^T * lamda = 0 =>  \sum_i ei = 0
          [x] + R(v) * [p2ix] - [p1ix]   
=>\sum_i  [y]          [p2iy]   [p1iy]  = 0  
          [z]          [p2iz]   [p1iz]    
=> N*[x] = \sum_i p1i - \sum_i  R(v)* p2i     
     [y]                             
     [z]                       
=> [x] = 1./N * \sum_i p1i - R(v) * 1./N * \sum_i p2i     
   [y]                                           
   [z]   
We can replace this last equation in the single ei and in the cost function L and get: 
ei(q)  = R(v) * (p2i - 1./N * \sum_j p2j) - (p1i - 1./N * \sum_j p1j)   
                                    

Therefore, if we compute 
p1m = 1./N * \sum_i p1i
p2m = 1./N * \sum_i p2i

dp1i = p1i - p1m 
dp2i = p2i - p2m 

=> ei(theta) =  R(v)*dp2i - dp1i                                      
                     

L = \sum_i   (R(v)*dp2i - dp1i)^T * lamda * (R(v)*dp2i - dp1i)
  = \sum_i dp2i^T*R(v)^T*lamda*R(v)*dp2i - 2*dp2i^T*R(v)^T*lamda*dp1i + dp1i^T*dp1i       
  ( by using the trace trick: trace(A*B*C) = trace(C*A*B) = trace(B*C*A) and lambda = lamdap * I3 with lamdap \in IR) 
  = \sum_i lamdap*dp2i^T*dp2i - 2*lamdap*trace(dp1i*dp2i^T*R(v)^T) + dp1i^T*dp1i 
  = c +  2*lamdap*trace(\sum_i (dp1i*dp2i^T)*R(v)^T)    
 where c is a constant that does not depend on the variables under optimization

Let Q = \sum_i (dp1i*dp2i^T)  \in IR^{3x3}
L can be minimized by maximizing (we neglect the constanct factor 2*lamdap and the term c):
J = trace(Q*R(v)^T)
If we decompose via SVD:
Q = U * S * V^T   where S = diag(s1,s2,s3) with s1 >= s2 >= s3 >= 0
then one has: 
J = trace(Q*R(v)^T) = J = trace(U*S*V^T*R^T) = trace(S*V^T*R^T*U)  (by using the trace trick)
Given V and U are orthogonal matrices, it can be easily seen that a solution is:
R(v) = U * D * V^T  
where D = diag(1,1,1) or D = diag(1,1,-1) and the +-1 is modulated in order to adjust det(R)>0.
Indeed, we cannot include S in the solution since we need an orthogonal matrix.
In fact, with such a solution: 
J = trace(Q*R(v)^T) = J = trace(S*V^T*R^T*U) = trace(S*D) = s1 + s2 +-s2   (+- depending on D selection)
which is constant and corresponds to the sum of the singular values.
In particular, we possibly need to adjust R in order to get det(R)>0 and we do that on the last term of D 
since we want the maximize J and the last term s3 is the smaller one since s3 <= s2 <= s1. 

*/

using Vector6d = Eigen::Matrix<double,6,1>;
using Matrix6d = Eigen::Matrix<double,6,6>;
using VecVector2d = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;
using VecVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

struct Range
{
    double r;    // radius 
    double th;   // azimuth 
    double ph;   // elevation 
};

void initRange(std::vector<Range>& z, size_t hN, size_t vN, double rangeMin, double rangeMax)
{
    const double deltath = 2*M_PI/hN;     // [0, 2*pi]
 
    const double startph = -M_PI/6; 
    const double endph   = +M_PI/6;
    const double deltaph = (endph-startph)/vN;   // [-pi/4, -pi/4]     
    
    const double deltaRange = rangeMax - rangeMin;

    z.resize(vN*hN); // rowMajor order 
    for(size_t jj=0; jj<vN; jj++) 
    {
        const double deltaphj = startph + deltaph * jj;   
        const size_t jjxvN = jj*hN;                 
        for(size_t ii=0; ii<hN; ii++)   
        {   
            const size_t idx = jjxvN + ii;            
            const double deltathi = deltath * ii;             
            z[idx].th = deltathi;               
            z[idx].ph = deltaphj;                   
    #if 0        
            z[idx].r  = rangeMin + deltaRange*(rand()/RAND_MAX);
    #else 
            z[idx].r  = rangeMin + deltaRange*fabs(cos(2*deltathi) + sin(3*deltathi) * sin(4*deltathi));        
    #endif       
        }
    }
}

// q = [x, y, z, v] where v \in IR^3 is an angle-axis rotation vector  
void rotoTranslateRange(const std::vector<Range>& z, const Vector6d& q, VecVector3d& p)
{
    Eigen::Vector3d tq = q.head(3);
    Eigen::Vector3d vq = q.tail(3); 
    Eigen::Matrix3d Rq = Eigen::AngleAxisd(vq.norm(), vq.normalized()).toRotationMatrix();
    //Sophus::SO3d Rq = Sophus::SO3d::exp(q.tail(3));
    
    p.resize(z.size()); 
    for(size_t ii=0; ii<z.size(); ii++)
    {
        Eigen::Vector3d pbi; 
        pbi[0] = z[ii].r*cos(z[ii].th)*cos(z[ii].ph);
        pbi[1] = z[ii].r*sin(z[ii].th)*cos(z[ii].ph);
        pbi[2] = z[ii].r*sin(z[ii].ph); 
        p[ii] = tq + Rq*pbi;
    }
}

// q = [x, y, z, v] where v \in IR^3 is an angle-axis rotation vector  
void rotoTranslatePoints(const VecVector3d& pin, const Vector6d& q, VecVector3d& pout)
{
    Eigen::Vector3d tq = q.head(3);
    Eigen::Vector3d vq = q.tail(3); 
    Eigen::Matrix3d Rq = Eigen::AngleAxisd(vq.norm(), vq.normalized()).toRotationMatrix();
    //Sophus::SO3d Rq = Sophus::SO3d::exp(q.tail(3));
    
    pout.resize(pin.size());   
    for(size_t ii=0; ii<pin.size(); ii++)
    {
        pout[ii] = tq + Rq*pin[ii];
    }
}

Eigen::Matrix3d skew(const Eigen::Vector3d& a)
{
    /*
    [a]x = [  0, -az,  ay]
           [ az,   0, -ax]
           [-ay,  ax,   0]  
    */
   Eigen::Matrix3d res; 
   res <<   0.0, -a[2],  a[1], 
           a[2],   0.0, -a[0],
          -a[1],  a[0],    0;
   return res;
}

void alignIterative(const VecVector3d& p2, 
                    const VecVector3d& p1, 
                    const double sigmap, 
                    Vector6d& q)
{
    assert(p1.size() == p2.size()); 

    const int maxNumIterations=100;
    const double stopDeltaqNorm = 1e-6; 

    Vector6d deltaq;
    deltaq << 0,0,0, 0,0,0; 
    
    Matrix6d H; 
    Vector6d b;
    Eigen::Matrix<double,3,6> Ji; 
    Eigen::Matrix3d lambdap = (1.0/(sigmap*sigmap)) * Eigen::Matrix3d::Identity(); 

    Ji.block<3,3>(0,0) = Eigen::Matrix3d::Identity(); 

    int numIteration = 1; 
    do
    {
        H = Matrix6d::Zero(); 
        b = Vector6d::Zero(); 

        const Eigen::Vector3d tq = q.head(3);
        //const Eigen::Vector3d vq = q.tail(3); 
        //const Eigen::Matrix3d Rq = Eigen::AngleAxisd(vq.norm(), vq.normalized()).toRotationMatrix();
        Sophus::SO3d Rq = Sophus::SO3d::exp(q.tail(3));
        
        for(size_t ii=0; ii<p2.size(); ii++)
        {
            // compute new prediction 
            const Eigen::Vector3d rp2i = Rq * p2[ii];            
            const Eigen::Vector3d p1i_pred = tq + rp2i;
            const Eigen::Vector3d ei = p1i_pred - p1[ii];  
            
            /*
                Ji \in IR^{3x6}
                Ji = dei/d(deltaq) = [1, 0, 0 |              ] 
                                     [0, 1, 0 | -[R(v)*p2i]x ]
                                     [0, 0, 1 |              ]                        
            */
            //Ji.block<3,3>(0,3) = -Sophus::SO3d::hat(rp2i);
            Ji.block<3,3>(0,3) = -skew(rp2i);
            
            H +=  Ji.transpose() * lambdap * Ji; 
            b += -Ji.transpose() * lambdap * ei;
        }
        
        // solve H * deltaq = b 
        deltaq = H.ldlt().solve(b); 

        q.head(3) += deltaq.head(3); 
        Rq = Sophus::SO3d::exp(deltaq.tail(3)) * Rq; 
        q.tail(3) = Rq.log();

        std::cout << "iteration : " << numIteration << ", deltaq: " << deltaq.transpose() << std::endl; 

        if(numIteration>=maxNumIterations)
        {
            std::cout << "reached max num iteration: " << maxNumIterations << ", current deltaq: " << deltaq.transpose() << std::endl;
            break; 
        }
        numIteration++;
    } while(deltaq.norm() > stopDeltaqNorm);
}


void alignSVD(const VecVector3d& p2, 
              const VecVector3d& p1,  
              Vector6d& q)
{
    assert(p1.size() == p2.size()); 

    /*
    Q = \sum_i (dp1i*dp2i^T)

    If we decompose via SVD:
    Q = U * S * V^T
    then a solution is 
    R(v) = U * V^T  (here, we cannot include S since we need an orthogonal matrix for R)    
    */

    const size_t N = p2.size();
    Eigen::Vector3d p1m = Eigen::Vector3d::Zero();
    Eigen::Vector3d p2m = Eigen::Vector3d::Zero();
    for(size_t ii=0; ii<p2.size();ii++)
    {
        p1m += p1[ii];
        p2m += p2[ii];
    }
    p1m /= N;
    p2m /= N;

    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero(); 
    for(size_t ii=0; ii<p2.size();ii++)
    {
        Q += (p1[ii]-p1m)*(p2[ii]-p2m).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = U*V.transpose();
    if(R.determinant()<0) 
    {
        R = U*Eigen::DiagonalMatrix<double,3>(1,1,-1)* V.transpose(); // we must guarantee det(R) = 1 > 0
    }

    /*
    => t = 1/N * \sum_i p1i - R(v) * 1/N * \sum_i p2i   = p1m - R(v) * p2m  
    */

    Eigen::Vector3d t = p1m - R * p2m;

#if 0
    std::cout << "R: \n" << R << std::endl; 
    std::cout << "det(R): " << R.determinant() << std::endl;  
    std::cout << "t: " << t.transpose() << std::endl; 
#endif 

    q.head(3) = t;
    Sophus::SO3d Rq(R); 
    q.tail(3) = Rq.log();
}


void drawScans(VecVector3d& p0, VecVector3d& pz, VecVector3d& pzalign1, VecVector3d& pzalign2) 
{
    //create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Scan Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const double eyeZ = 100.; 
    pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(0, 0, eyeZ, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));


    auto drawScan = [&](VecVector3d& pv, GLint pointSize, GLfloat red, GLfloat green, GLfloat blue)
    {
        glPointSize(pointSize);
        glBegin(GL_POINTS);
        glColor3f(red, green, blue);        
        for (size_t i = 0; i < pv.size(); i++) 
        {
            glVertex3d(pv[i][0], pv[i][1], pv[i][2]);
        }
        glEnd();        
    };

    while (pangolin::ShouldQuit() == false) 
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        drawScan(p0,       2, 1,0,0);
        drawScan(pz,       2, 0,1,0);
        drawScan(pzalign1, 4, 0,0,1);  
        drawScan(pzalign2, 4, 0,1,1);               
        pangolin::FinishFrame();
        usleep(5000);//sleep 5 ms
    }
}

void generateRandPose(Vector6d& q, const double maxDeltaTranslation, const double maxAngle)
{
    q[0] = -maxDeltaTranslation + 2*maxDeltaTranslation * rand()/RAND_MAX;
    q[1] = -maxDeltaTranslation + 2*maxDeltaTranslation * rand()/RAND_MAX;
    q[2] = -maxDeltaTranslation + 2*maxDeltaTranslation * rand()/RAND_MAX;

    Eigen::Vector3d u(-1.0 + 2.0*rand()/RAND_MAX,-1.0 + 2.0*rand()/RAND_MAX,-1.0 + 2.0*rand()/RAND_MAX); 
    u = u.normalized()*maxAngle; 
    q[3] = u[0];
    q[4] = u[1];
    q[5] = u[2];    
}

VecVector3d addCartesianNoiseToPoints(VecVector3d& p, double sigmap)
{
    VecVector3d pn; 
    pn.resize(p.size()); 

    std::default_random_engine generator;
    generator.seed(time(NULL));

    std::normal_distribution<double> distributionx(0.0,sigmap);
    std::normal_distribution<double> distributiony(0.0,sigmap);
    std::normal_distribution<double> distributionz(0.0,sigmap);    
    for(size_t ii=0; ii<p.size(); ii++)
    {
        pn[ii][0] = p[ii][0] + distributionx(generator);
        pn[ii][1] = p[ii][1] + distributiony(generator);       
        pn[ii][2] = p[ii][2] + distributionz(generator);              
    }
    return pn;
}

int main(int argc, char **argv) 
{
	// init random seed value
	srand((unsigned) time(NULL));

    const size_t hN = 360; // horizontal resolution 
    const size_t vN = 100; // vertical resolution 
    const double rangeMin = 10; 
    const double rangeMax = 50; 
    const double sigmap = 0.01; // [m]

    std::vector<Range> z;
    initRange(z, hN, vN, rangeMin, rangeMax);

    // q = [x, y, z, v] where v \in IR^3 is an angle-axis rotation vector  

    Vector6d q0;
    const double maxDeltaTranslation = 2.0;
    const double maxAngle = M_PI/4; 
    generateRandPose(q0,maxDeltaTranslation,maxAngle);
    //q0 << 1.4,2.3,0.0, 0.0,0.0,M_PI/6; // delta pose from origin that we need to recover via alignment
    VecVector3d p0; 
    rotoTranslateRange(z, q0, p0);  // generate map points without noise

    Vector6d qz;
    qz << 0.0,0.0,0.0,  0.0,0.0,0.0; 
    VecVector3d pz;    
    rotoTranslateRange(z, qz, pz); // same range but starting from different pose 
    //VecVector3d pzn = pz;     
    VecVector3d pzn = addCartesianNoiseToPoints(pz, sigmap); // add noise to unaligned points 

    Vector6d qalign1 = qz; 
    Vector6d qalign2 = qz;     
    // align noisy points pzn to map points p0
    alignIterative(pz, p0, sigmap, qalign1);
    alignSVD(pz, p0, qalign2);    

    std::cout << "q0: " << q0.transpose() << std::endl; 
    std::cout << "q1 (iterative): " << qalign1.transpose() << std::endl; 
    std::cout << "q2 (SVD): " << qalign2.transpose() << std::endl;     

    Sophus::SO3d R0 = Sophus::SO3d::exp(q0.tail(3));
    Sophus::SO3d Ralign1 = Sophus::SO3d::exp(qalign1.tail(3));    
    Sophus::SO3d Ralign2 = Sophus::SO3d::exp(qalign2.tail(3));

    Vector6d err1;
    err1.head(3) = q0.head(3) - qalign1.head(3);
    err1.tail(3) = (R0.inverse()*Ralign1).log();

    Vector6d err2;
    err2.head(3) = q0.head(3) - qalign2.head(3);
    err2.tail(3) = (R0.inverse()*Ralign2).log();

    std::cout << "error1 (iterative): " << err1.norm() << std::endl; 
    std::cout << "error2 (SVD): " << err2.norm() << std::endl;     

    VecVector3d pzalign1, pzalign2;    
    rotoTranslatePoints(pzn, qalign1, pzalign1); 
    rotoTranslatePoints(pzn, qalign2, pzalign2);     

    drawScans(p0, pzn, pzalign1, pzalign2);

    return 0; 
}