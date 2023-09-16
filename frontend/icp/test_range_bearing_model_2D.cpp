#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include <chrono>
#include <random>

#include <pangolin/pangolin.h>
#include <pangolin/display/default_font.h>
#include <unistd.h>

#include "macros.h"
#include "S1Utils.h"

/*

===============================================
> Short problem description: 
===============================================

Given a set of range-bearing readings {[ri,thi]}, where ri, thi \in IR i=1,2,...,N, 
align the set of points 
pzi = [x] + [ri*cos(theta + thi)]   \in IR^2
      [y]   [ri*sin(theta + thi)]
corresponding to {[ri,thi]} with a set of known landmark points {pi} \in IR^2
via an unknown transformation q=[x,y,theta]^T \in SE(2). 
Note that pi, pzi \in IR^2 for i=1,2,...,N.
We assume data association has been already solved:
pi <-> pzi  for i=1,2,...,N 

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


Let's define [theta]z =  [0    ,  -theta]   \in IR^{2x2}
                         [theta,       0]

A rotation matrix R \in SO(2) can be represented as follows: 

R(theta) = [cos(theta), -sin(theta)] = exp([0    ,  -theta]) ~= I + [theta]z = I + theta * [0, -1]
           [sin(theta),  cos(theta)]       [theta,       0]                                [1,  0]

One has: 
R(theta) = exp([theta]z) = [1]^T * exp( [0    ] ) = [1]^T * Rz(theta)    where Rz(theta) \in SO(3) is a rotation of theta about the z axis
                           [1]          [0    ]     [1] 
                           [0]          [theta]x    [0]

Note that: 
exp([theta]z + [dtheta]z) = exp([dtheta]z) * exp([theta]z) = exp([theta]z) * exp([dtheta]z)   
since rotations above the same axis can commute.   

===============================================
> Range bearing problem and model 
===============================================

Range scan: 
z = [ [r1,   r2,...,rN],     <- i-th beam lenght: ri \in IR^+
      [th1, th2,....thN] ]   <- i-th orientation: thi \in IR relative to robot orientation (0 is frontal heading)

zi = [ri,thi]

robot frame = {x front, y left, z up}  (ROS convention)
robot pose = q = [x,y,theta]^T  \in SE(2)

We assume the range readings z have been already associated with landmark points that are present in the map \in IR^2. 
That is: {ri,thi} <-> pi  where pi is the i-th landmark point \in IR^2 that is stored in the map.    

Note that:
[ri*cos(theta+thi)] = R(theta) * [ri*cos(thi)] 
[ri*sin(theta+thi)]              [ri*sin(thi)]   

Let be:
pzi = [x] + [ri*cos(theta + thi)]  \in IR^2
      [y]   [ri*sin(theta + thi)]

We consider 3 noise models for the polar range readings z. 
I) No noise: 
pi = pzi = [x] + [ri*cos(theta + thi)]  \in IR^2
           [y]   [ri*sin(theta + thi)]
Here each reading is charaterized by a Dirac PDF. 

II) Polar noise: 
pi = pzi + polar noise     (we could also see it as the other way around, noise on the other side of the equation)
That is: 
pi = [x] + [(ri+epsir)*cos(theta + thi + epsith)] = [x] + [cos(theta), -sin(theta)] * [(ri+epsir)*cos(thi+ epsith)] \in IR^2
     [y]   [(ri+epsir)*sin(theta + thi + epsith)]   [y]   [sin(theta),  cos(theta)]   [(ri+epsir)*sin(thi+ epsith)]

where 
epsir \in IR ~ N(0,sigmar^2)      with sigmar,sigmath \in IR 
epsith \in IR ~ N(0,sigmath^2)       
and we assume all epsir and epsith are independent noises (for i=1,2,...,N).

III) Cartesian noise:
pi = pzi + cartesian noise     (we could also see it as the other way around, noise on the other side of the equation)  
pi =  [x] + [ri*cos(theta + thi)] + [epsix] = [cos(theta), -sin(theta)] * [ri*cos(thi)] + [epsix] \in IR^2
      [y]   [ri*sin(theta + thi)]   [epsiy]   [sin(theta),  cos(theta)]   [ri*sin(thi)]   [epsiy]

where 
epsi = [epsix] \in IR^2 ~ N(0,sigmap)  
       [epsiy]
and the covariance matrix is : 
sigmap = [sigmax^2,       0 ]   where sigmax,sigmay \in IR and for simplicity sigmax=sigmay
         [       0, sigmay^2]
We assume all epsix and epsiy are independent noises (for i=1,2,...,N).

NOTE: 
The characteristics of the cartesian noise can be approximately derived (up to first oder) 
from an assumed polar noise model. In fact:

pi = [x] + R(theta)*[(ri+epsir)*cos(thi + epsith)]  = [x] + R(theta)*R(epsith)*[(ri+epsir)*cos(thi)]
     [y]            [(ri+epsir)*sin(thi + epsith)]    [y]                      [(ri+epsir)*sin(thi)]

By using
R(epsith) ~= I + epsith*[0, -1]
                        [1,  0]
one has:
pi = [x] + R(theta)*R(epsith)*[(ri+epsir)*cos(thi)] =
     [y]                      [(ri+epsir)*sin(thi)]
~= [x] + R(theta)*( [ri*cos(thi)] + epsir*[cos(thi)] + epsith*[0, -1]*[ri*cos(thi)] ) 
   [y]              [ri*sin(thi)]         [sin(thi)]          [1,  0] [ri*sin(thi)]
Here, we neglected second order noise terms (factors with epsir*epsith). 
=> pi = [x] + R(theta)*[ri*cos(thi)] + R(theta)* ( epsir*[cos(thi)] + epsith*[-ri*sin(thi)] ) =
        [y]            [ri*sin(thi)]                     [sin(thi)]          [ ri*cos(thi)]
       = [x] + R(theta)*[ri*cos(thi)] + R(theta)*R(thi)*[1, 0]*[epsir] 
         [y]            [ri*sin(thi)]                   [0,ri] [epsith] 
By comparing this last equation with the cartesian noise model equation, one has up to first order: 
 [epsix] ~= R(theta)*R(thi)*[1, 0]*[ epsir] = A(theta,ri,thi) * [ epsir]
 [epsiy]                    [0,ri] [epsith]                     [epsith]
that implies that the Cartesian noise characteristics at the i-th range reading depends on (ri,thi).
In general, one has: 
 [sigmax^2] =  A(theta,ri,thi) * [sigmar,       0] * A(theta,ri,thi)^T
 [sigmay^2]                      [     0, sigmath]
In the particular case epsir and epsith are small enough and on the average epsir~=ri*epsith (isotropic Gaussian noise), 
one can approximate up to first order: 
 [sigmax] ~= [sigmar] 
 [sigmay]    [sigmar]
since rotations do not change the characteristics of an isotropic noise model. 

===============================================
> Problem with Cartesian noise model 
===============================================

Let's start by considering the cartesian noise model. 
pi = pzi + cartesian noise  
That is: 
pi = [x] + [ri*cos(theta + thi)] + [epsix]  = [x] + R(theta) * [ri*cos(thi)] + [epsix]   \in IR^2
     [y]   [ri*sin(theta + thi)]   [epsiy]    [y]              [ri*sin(thi)]   [epsiy]

robot pose = q = [x,y,theta]^T  \in SE(2)

Let:
 pzi = [x] + R(theta) * [ri*cos(thi)]   \in IR^2 
       [y]              [ri*sin(thi)] 

One has: 
 pi ~ N(pzi,sigmap;q)
 ei = (pzi-pi)  ~ N(0,sigmap;q)
In long vector form:
 e = [e1,e2,...,eN] \in IR^{2*Nx1}

Measurement model. Assuming: 
 (i) points p1 belongs to the map m, points p1 are newly observed from a different pose as pzi (computed from {[ri,thi]})  
(ii) we have independent noise on each observed points p2i
we have that: 
p(z|q, map) = \prod_i p(zi|q, map) = \prod_i p(pzi|q, map) 

=> p(e|q, map) = \prod_i p(ei|q, map) = \prod_i N(pzi-pi|0,sigmap;q)

We have to register the new scan to the map. We use a MLE estimation approach. 

q* = arg min_{q} NLL = arg min_{q} \prod_i log(N(pzi-pi|0,sigmap;q)) = arg min_{q} \sum_i (pzi-pi)^T * sigmap^-1 * (pzi-pi) 

L = chi = 1/2 * \sum_i (pzi-pi)^T * sigmap^-1 * (pzi-pi) = 1/2 * \sum_i ei^T * sigmap^-1 * ei

We consider a perturabation model around (x,y,theta):

deltaq = [deltax]   \in SE(2)
         [deltax]
         [deltath]

ei = (pzi-pi) = [cos(deltath), -sin(deltath)] * R(theta) * [ri*cos(thi)] + [x + deltax - xi] 
                [sin(deltath),  cos(deltath)]              [ri*sin(thi)]   [y + deltay - yi]


In order to optimize L, we use an iterative approach and enforce dL/d(deltaq) = 0 at each step. 
Namely, we can approx:
    ei(q+deltaq) = e0i + Ji * deltaq, where e0i = e(q) \in IR^2 and Ji = dei/d(deltaq)|_{deltaq=0} \in IR^{2x3} 

=> L = 1/2 * \sum_i (e0i + Ji * deltaq)^T * sigmap^-1 * (e0i + Ji * deltaq)

dL/d(deltaq) = d/d(deltaq) [1/2 * \sum_i (e0i + Ji * deltaq)^T * sigmap^-1 * (e0i + Ji * deltaq)] = 
= \sum_i (e0i + Ji * deltaq)^T * sigmap^-1 *Ji  = 0

Then we obtain the following normal equations: 
\sum_i (Ji * sigmap^-1 *Ji) * deltaq = -\sum_i Ji^T * sigmap^-1 *e0i
In compact form: 
    H * deltaq = b
where 
    H =  \sum_i (Ji * sigmap^-1 *Ji)   \in IR^{3x3}
    b = -\sum_i Ji^T * sigmap^-1 *e0i  \in IR^{3x1}

ei(q) = (pzi-pi) = R(theta)] * [ri*cos(thi)] + [x] - pi 
                               [ri*sin(thi)]   [y]

Let's now put the perturbation model into action. 
We have deltaq = (deltax, deltay, deltath):
ei(q+deltaq) = (pzi-pi) = [cos(deltath), -sin(deltath)] * R(theta) * [ri*cos(thi)] + [x + deltax] - pi 
                          [sin(deltath),  cos(deltath)]              [ri*sin(thi)]   [y + deltay] 

ei ~=  ei(deltaq=0) + Ji * deltaq ~= (I + [      0, -deltath])* R(theta) * [ri*cos(thi)] + [x] + [deltax] - pi
                                          [deltath,        0]              [ri*sin(thi)]   [y]   [deltay]

    = (R(theta) * [ri*cos(thi)] + [x] - pi) +  deltath * [ 0, -1] * R(theta) * [ri*cos(thi)] + [deltax]
                  [ri*sin(thi)]   [y]                    [ 1,  0]              [ri*sin(thi)]   [deltay]

Recall that:
R(theta) * [ri*cos(thi)] = [ri*cos(theta+thi)] 
           [ri*sin(thi)]   [ri*sin(theta+thi)] 

Ji \in IR^{2x3} can be easily found from the last ei equation: 
Ji = dei/d(deltaq) = [1, 0,  -ri*sin(theta+thi)]
                     [0, 1,   ri*cos(theta+thi)]

Once the solution to the following system is found 
\sum_i Ji * sigmap^-1 *Ji * deltaq = - \sum_i Ji^T * sigmap^-1 *e0i
then we update: 
    [x    ] += deltax
    [y    ] += deltay
    [theta] += deltath    <-- here, we apply proper angle wrapping in order to keep the representation in [-pi,pi]

In matrix form:
    e = [e1,e2,...,eN] \in IR^{2*Nx1}
    J = d(e)/d(deltaq) \in IR^{2*Nx3} 
    J^T * sigmap^-1 * J * deltaq = -J^T *sigmap^-1 * e
    H * deltaq = b
    H =  J^T * sigmap^-1 * J   \in IR^{3x3}
    b = -J^T * sigmap^-1 * e   \in IR^{3x1}

===============================================
> Problem with polar noise model 
===============================================
We assume:
pi = pzi + cartesian noise  
That is: 
pi = [x] + [(ri+epsir)*cos(theta + thi + epsith)] = [x] + [cos(theta), -sin(theta)] * [(ri+epsir)*cos(thi+ epsith)] \in IR^2
     [y]   [(ri+epsir)*sin(theta + thi + epsith)]   [y]   [sin(theta),  cos(theta)]   [(ri+epsir)*sin(thi+ epsith)]

robot pose = q = [x,y,theta]^T  \in SE(2)

In fact, we can characterize the sensor model (measurement model) with the following equations:
* p(ri|x,y,theta, pi) = N(mr,sigmar^2) = c * exp(-1/2 * (r-mr)^2/sigmar^2)
    where mr = |pi - [x,y]| \in IR  
* p(thi|x,y,theta, pi) = N(mt,sigmath^2) = c * exp(-1/2 * (thi-mt)^2/sigmath^2)
    where mt = atan2(pi.y - y, pi.x - x) - theta =  atan2(pi_b_y, pi_b_x) 
    where in the sensor frame (b) one has 
    [pi_b_x] = R(theta)^T * [pi.x - x]             (note that R(theta) = Rwb)
    [pi_b_y]                [pi.y - y]

We can factorize: 
p(zi|x,y,theta, pi) = p(ri|x,y,theta, pi)*p(thi|x,y,theta, pi) 
Let 
eri  = sqrt((pi.x - x)^2 + (pi.y - y)^2) - ri   => eri ~ N(0,sigmar^2)
ethi = atan2(pi.y - y, pi.x - x) - theta - thi  => ethi ~ N(0,sigmath^2)
ei = [ eri] \in IR^2
     [ethi]

Measurement model. Assuming: 
 (i) points pi belongs to the map m, points pi are newly observed from a different pose as pzi (computed from {[ri,thi]})  
(ii) we have independent noise on each observed points pzi
we have that: 
p(e|q,map) = \prod_i p(ei|q, map) = \prod_i p(eri|x,y,theta, pi)*p(ethi|x,y,theta, pi) 

q* = arg min_{q} NLL = arg min_{q} \prod_i NLL(p(e|q,map)) 
   = arg min_{q} \sum_i [ eri^T * sigmar^-2 * eri + ethi^T * sigmath^-2 * ethi]

L = chi = \sum_i [ eri^T * sigmar^-2 * eri + ethi^T * sigmath^-2 * ethi]

We consider a perturabation model around (x,y,theta):
deltaq = [deltax]   \in SE(2)
         [deltax]
         [deltath]

eri  = sqrt((pi.x - (x+deltax))^2 + (pi.y - (y+deltay))^2) - ri 
ethi = atan2(pi.y - (y+deltay), pi.x - (x+deltax)) - (theta+deltath) - thi =
     = atan([pi.y - (y+deltay)]/[pi.x - (x+deltax)]) - (theta+deltath) - thi

eri = eri(deltaq=0) + Jri * [deltax, deltay, deltath]^T
Jri  = d(eri)/d(deltaq) = [-2*(pi.x - x), -2*(pi.y - y), 0] * 1.0/[2*sqrt((pi.x-x)^2 + (pi.y-y)^2)]  \in IR^{1x3}

ethi = ethi(deltaq=0) + Jthi * [deltax, deltay, deltath]^T
Jthi = d(ethi)/d(deltaq) = [w * (pi.y-y)/((pi.x-x)^2), -w*(1/pi.x-x)), -1]   \in IR^{1x3}
where w = 1.0/(1+a^2) and a = (pi.y-y)/(pi.x-x)
      => w = (pi.x-x)^2/[(pi.x-x)^2 + (pi.y-y)^2]

Let Ji = [ Jri] \in IR^{2x3}
         [Jthi]
As above, once the solution to the following system is found, 
\sum_i Ji * sigmap^-1 *Ji * deltaq = - \sum_i Ji^T * sigmap^-1 *e0i
then we update: 
    [x    ] += deltax
    [y    ] += deltay
    [theta] += deltath    <-- here, we apply proper angle wrapping in order to keep the representation in [-pi,pi]

*/


using VecVector2d = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;

struct Range
{
    double r; 
    double th;
};

void initRange(std::vector<Range>& z, size_t N, double rangeMin, double rangeMax)
{
    const double deltath = 2*M_PI/N; 
    const double deltaRange = rangeMax - rangeMin;

    z.resize(N);
    for(size_t ii=0; ii<N; ii++)
    {
        const double deltathii = deltath * ii;        
#if 0        
        z[ii].r  = rangeMin + deltaRange*(rand()/RAND_MAX);
#else 
        z[ii].r  = rangeMin + deltaRange*fabs(cos(2*deltathii) + sin(3*deltathii) * sin(4*deltathii));        
#endif         
        z[ii].th = deltathii; 
    }
}

std::vector<Range> addNoiseToRange(const std::vector<Range>& z, double sigmar, double sigmath)
{
    std::vector<Range> zn; 
    zn.resize(z.size()); 

    std::default_random_engine generator;
    generator.seed(time(NULL));

    std::normal_distribution<double> distributionr(0.0,sigmar);
    std::normal_distribution<double> distributionth(0.0,sigmath);
    for(size_t ii=0; ii<z.size(); ii++)
    {
        zn[ii].r  = z[ii].r + distributionr(generator);
        zn[ii].th = z[ii].th + distributionth(generator);        
    }
    return zn;
}

// q = [x, y, theta]
void rotoTranslateRange(const std::vector<Range>& z, const Eigen::Vector3d& q, VecVector2d& p)
{
    p.resize(z.size()); 
    for(size_t ii=0; ii<z.size(); ii++)
    {
        const double theta = q[2]+z[ii].th;
        const double ctheta = cos(theta); 
        const double stheta = sin(theta); 
        p[ii][0] = q[0] + z[ii].r*ctheta;
        p[ii][1] = q[1] + z[ii].r*stheta;
    }
}

// keep the angle in [-pi, pi]
double wrapAngle(const double a)
{
	return fmod(a + M_PI, 2*M_PI ) - M_PI;    
}

// return the shortest angle diff 
double angleDiff(const double a, const double b)
{
    double res = fmod(a-b, 2*M_PI); 
    if(fabs(res)>M_PI) res-=2*M_PI;
    return res;  
}

// Align by using Cartesian noise model 
void alignCM(const VecVector2d& p, 
            const std::vector<Range>& z, 
            const double sigmap, 
            Eigen::Vector3d& q)
{
    std::cout << "starting alignCM" << std::endl; 
    assert(p.size() == z.size()); 

    const int maxNumIterations=100;
    const double stopDeltaqNorm = 1e-6; 

    Eigen::Vector3d deltaq(0,0,0); 
    
    Eigen::Matrix3d H; 
    Eigen::Vector3d b;
    Eigen::Matrix<double,2,3> Ji; 
    Eigen::Matrix2d lambdap = (1.0/(sigmap*sigmap)) * Eigen::Matrix2d::Identity(); 

    int numIteration = 1; 
    do
    {
        H = Eigen::Matrix3d::Zero(); 
        b = Eigen::Vector3d::Zero(); 

        for(size_t ii=0; ii<p.size(); ii++)
        {
            const double thetai = q[2]+z[ii].th;
            const double deltazix = z[ii].r*cos(thetai); // ri*cos(theta+thi) 
            const double deltaziy = z[ii].r*sin(thetai); // ri*sin(theta+thi)
            const Eigen::Vector2d pzi(q[0] + deltazix, 
                                      q[1] + deltaziy);
            const Eigen::Vector2d ei = pzi - p[ii];  
            
            /*
                Ji \in IR^{2x3}
                Ji = dei/d(deltaq) = [1, 0,  -ri*sin(theta+thi)]
                                     [0, 1,   ri*cos(theta+thi)]
            */

            Ji << 1, 0, -deltaziy, 
                  0, 1,  deltazix; 

            H +=  Ji.transpose() * lambdap * Ji; 
            b += -Ji.transpose() * lambdap * ei;
        }
        
        // solve H * deltaq = b 
        deltaq = H.ldlt().solve(b); 

        q += deltaq; 
        q[2] = wrapAngle(q[2]);

        std::cout << "iteration : " << numIteration << ", deltaq: " << deltaq.transpose() << std::endl; 

        if(numIteration>=maxNumIterations)
        {
            std::cout << "reached max num iteration: " << maxNumIterations << ", current deltaq: " << deltaq.transpose() << std::endl;
            break; 
        }
        numIteration++;
    } while(deltaq.norm() > stopDeltaqNorm);
}



// Align by using polar noise model 
void alignPM(const VecVector2d& p, 
            const std::vector<Range>& z, 
            const double sigmar,
            const double sigmath, 
            Eigen::Vector3d& q)
{
    std::cout << "starting alignPM" << std::endl; 

    assert(p.size() == z.size()); 

    const int maxNumIterations=100;
    const double stopDeltaqNorm = 1e-6; 

    Eigen::Vector3d deltaq(0,0,0); 
    
    Eigen::Matrix3d H; 
    Eigen::Vector3d b;
    Eigen::Matrix<double,2,3> Ji; 
    const double lambdar = 1.0/pow2(sigmar);
    const double lambdath = 1.0/pow2(sigmath);
    Eigen::Matrix2d lambda = Eigen::DiagonalMatrix<double,2>(lambdar,lambdath);

    int numIteration = 1; 
    do
    {
        H = Eigen::Matrix3d::Zero(); 
        b = Eigen::Vector3d::Zero(); 

        for(size_t ii=0; ii<p.size(); ii++)
        {
            /*
            eri  = sqrt((pi.x - x))^2 + (pi.y - (y+deltay))^2) - ri 
            ethi = atan2(pi.y - y, pi.x - x) - theta - thi 
            */ 

            const Eigen::Vector2d& pi = p[ii];   
            const double deltaxi = pi[0] - q[0];
            const double deltayi = pi[1] - q[1]; 
            const double deltaxi2 = pow2(deltaxi); 
            const double deltayi2 = pow2(deltayi);   
            const double disti2 = deltaxi2 + deltayi2;                
            const double disti = sqrt(disti2);         
            const double thi = S1Utils::diffS1(atan2(deltayi, deltaxi),q[2]);  
            const Eigen::Vector2d ei( disti - z[ii].r,  
                                      S1Utils::diffS1(thi,z[ii].th) );
            
            /*
            Jri  = d(eri)/d(deltaq) = [-2*(pi.x - x), -2*(pi.y - y), 0] * 1.0/[2*sqrt((pi.x-x)^2 + (pi.y-y)^2)]  \in IR^{1x3}
            Jthi = d(ethi)/d(deltaq) = [w * (pi.y-y)/((pi.x-x)^2), -w*(1/pi.x-x)), -1]   \in IR^{1x3}
            where w = 1.0/(1+a^2) and a = (pi.y-y)/(pi.x-x)
                => w = (pi.x-x)^2/[(pi.x-x)^2 + (pi.y-y)^2]

            Ji = [ Jri] \in IR^{2x3}
                 [Jthi]
            */

            const double w = disti2>1e-6? deltaxi2/disti2 : deltaxi2/(1e-6 + disti2);
            Ji <<     -deltaxi/disti,  -deltayi/disti,   0.0, 
                  w*deltayi/deltaxi2,      -w/deltaxi,  -1.0; 

            H +=  Ji.transpose() * lambda * Ji; 
            b += -Ji.transpose() * lambda * ei;
        }
        
        // solve H * deltaq = b 
        deltaq = H.ldlt().solve(b); 

        q += deltaq; 
        q[2] = wrapAngle(q[2]);

        std::cout << "iteration : " << numIteration << ", deltaq: " << deltaq.transpose() << std::endl; 

        if(numIteration>=maxNumIterations)
        {
            std::cout << "reached max num iteration: " << maxNumIterations << ", current deltaq: " << deltaq.transpose() << std::endl;
            break; 
        }
        numIteration++;
    } while(deltaq.norm() > stopDeltaqNorm);
}

void drawScans(VecVector2d& p, VecVector2d& pz, VecVector2d& pzalign, VecVector2d& pzalign2) 
{
    //create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Scan Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(0, 0, 100, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));


    auto drawScan = [&](VecVector2d& pv, GLint lineWidth, GLfloat red, GLfloat green, GLfloat blue)
    {
        glLineWidth(lineWidth);
        for (size_t i = 0; i < pv.size()-1; i++) 
        {
            //draw three coordinate axes for each pose
            Eigen::Vector3d p1(  pv[i][0],  pv[i][1],0);
            Eigen::Vector3d p2(pv[i+1][0],pv[i+1][1],0);      
            glBegin(GL_LINES);
            glColor3f(red, green, blue);
            glVertex3d(p1[0], p1[1], p1[2]);
            glVertex3d(p2[0], p2[1], p2[2]);
            glEnd();
        }
    };

    while (pangolin::ShouldQuit() == false) 
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        drawScan(p,        1, 1,0,0);
        drawScan(pz,       1, 0,1,0);
        drawScan(pzalign,  2, 0,0,1);        
        drawScan(pzalign2, 2, 0,1,1);             
        pangolin::FinishFrame();
        usleep(5000);//sleep 5 ms
    }
}

void generateRandPose(Eigen::Vector3d& q, const double maxDeltaTranslation, const double maxAngle)
{
    q[0] = -maxDeltaTranslation + 2*maxDeltaTranslation * rand()/RAND_MAX;
    q[1] = -maxDeltaTranslation + 2*maxDeltaTranslation * rand()/RAND_MAX;
    q[2] = -maxAngle + 2*maxAngle*rand()/RAND_MAX;  
}

int main(int argc, char **argv) 
{
	// init random seed value
	srand((unsigned) time(NULL));
        
    const size_t N = 360; 
    const double rangeMin = 10; 
    const double rangeMax = 50; 
    const double sigmap = 0.01; // [m]
    const double sigmar = 0.01; // [m]    
    const double sigmath = 2.0*M_PI/N * 1.0/3.0; // [m]        

    std::vector<Range> z;
    initRange(z, N, rangeMin, rangeMax);

    // get noisy reading in order to generate noisy points 
    std::vector<Range> zn = addNoiseToRange(z,sigmar,sigmath);

    Eigen::Vector3d q0;
    const double maxDeltaTranslation = 2.0;
    const double maxAngle = M_PI/4; 
    generateRandPose(q0,maxDeltaTranslation,maxAngle);    
    //q0 << 1.4,2.3,M_PI/6; // delta pose from origin that we need to recover via alignment

    // generate clean landmark points 
    VecVector2d p0; 
    rotoTranslateRange(z, q0, p0); 

    Eigen::Vector3d qz(0,0,0); // first guess 
    VecVector2d pz;    
    // generate the clean unaligned landmark points as perceived from the first guess (for drawing them)    
    rotoTranslateRange(z, qz, pz); 

    Eigen::Vector3d qalign1 = qz; 
    // align the noisy scan (zn,qalign1) to p0 via Cartesian noise model 
    alignCM(p0, zn, sigmap, qalign1);

    Eigen::Vector3d qalign2 = qz; 
    // align the noisy scan (zn,qalign2) to p0
    alignPM(p0, zn, sigmar, sigmath, qalign2);    

    std::cout << "q0: " << q0.transpose() << std::endl; 
    std::cout << "qalign1 (Cartesian noise): " << qalign1.transpose() << std::endl; 
    std::cout << "qalign2 (polar noise): " << qalign2.transpose() << std::endl;     

    Eigen::Vector3d err1 = q0-qalign1;
    err1[2] = angleDiff(q0[2],qalign1[2]);
    std::cout << "error1 (Cartesian noise): " << err1.norm() << std::endl; 

    Eigen::Vector3d err2 = q0-qalign2;
    err2[2] = angleDiff(q0[2],qalign2[2]);
    std::cout << "error2 (polar noise): " << err2.norm() << std::endl;     

    VecVector2d pzalign1;    
    rotoTranslateRange(z, qalign1, pzalign1); 
    VecVector2d pzalign2;    
    rotoTranslateRange(z, qalign2, pzalign2); 

    drawScans(p0, pz, pzalign1, pzalign2);

    return 0; 
}
