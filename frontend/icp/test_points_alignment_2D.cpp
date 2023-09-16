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

#include <macros.h>

/*

===============================================
> Short problem description: 
=============================================== 

Align a set of points {p1i} with a set of points {p2i} where i=1,2,...,N via an unknown  transformation q \in SE(2). 
Note that p1i, p2i \in IR^2 for i=1,2,...,N
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


Let define [theta]z =  [0    ,  -theta]   \in IR^{2x2}
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
> Alignment of 2D points via iterative approach 
===============================================

robot frame = {x front, y left, z up}  (ROS convention)
robot pose = q = [x,y,theta]^T  \in SE(2)

We are given two set of points p1 and p2, where p1i \in IR^2 and p2j \in IR^2.
We need to align p2 to p1. 
For the sake of simplicity, assume the data association has been already solved, that is: p1i <-> p2i

We consider a cartesian noise model. 

p1i = [x] + [cos(theta), -sin(theta)] * p2i + [epsix]   \in IR^2
      [y]   [sin(theta),  cos(theta)]         [epsiy]
   
epsi = [epsix] ~ N(0,sigmap)  
       [epsiy]
and the covariance matrix is : 
sigmap = [sigmar^2,       0 ]   where sigmar \in IR 
         [       0, sigmar^2]

Note that: 
    ei = (R*p2i+t-p1i) ~ N(0,sigmap;q)
where: 
R = [cos(theta), -sin(theta)] \in IR^{2x2}
    [sin(theta),  cos(theta)]
t = [x] \in IR^2
    [y]

In long vector form:
 e = [e1,e2,...,eN] \in IR^{2*Nx1}

Measurement model. Assuming: 
 (i) points p1 belongs to the map m, points p1 are newly observed from a different pose as p2  
(ii) we have independent noise on each observed points p2i
we have that: 
p(e|q, map) = \prod_i p(ei|q, map) = \prod_i N(R*p2i+t-p1i|0,sigmap;q)

We have to register the new scan to the map. We use a MLE estimation approach. 

q* = arg min_{q} NLL = arg min_{q} \prod_i log(N(R*p2i+t-p1i|0,sigmap;q)) 
                     = arg min_{q} \sum_i (R*p2i+t-p1i)^T * sigmap^-1 * (R*p2i+t-p1i) 

L = chi2 = 1/2 * \sum_i (R*p2i+t-p1i)^T * sigmap^-1 * (R*p2i+t-p1i) = 1/2 * \sum_i ei^T * sigmap^-1 * ei

We can (left) perturb ei as follows:  
ei = (R*p2i+t-p1i) = [ cos(deltath), -sin(deltath)] * R(theta)] * p2i + [x + deltax] - p1i 
                     [ sin(deltath),  cos(deltath)]                     [y + deltay]

deltaq = [deltax]   \in SE(2)  where [deltax] \in IR^2 and  deltath \in S^1
         [deltay]                    [deltay]
         [deltath]

In order to optimize L, we use an iterative approach and enforce dL/d(deltaq) = 0 at each step. 
Namely, we can approx:
    ei(q+deltaq) = e0i + Ji * deltaq, where e0i = ei(q) and Ji = dei/d(deltaq) \in IR^{2x3}
Note that Ji = dei/d(deltaq) must be elaluated at the same q where e0i = ei(q) is evalutated.

=> L = 1/2 * \sum_i (e0i + Ji * deltaq)^T * sigmap^-1 * (e0i + Ji * deltaq)

dL/d(deltaq) = d/d(deltaq) 1/2 * \sum_i (e0i + Ji * deltaq)^T * sigmap^-1 * (e0i + Ji * deltaq) = 
= \sum_i (e0i + Ji * deltaq)^T * sigmap^-1 *Ji  = 0

Then we obtain the following normal equations: 
\sum_i (Ji * sigmap^-1 *Ji) * deltaq = -\sum_i Ji^T * sigmap^-1 *e0i
In compact form: 
    H * deltaq = b
where 
    H =  \sum_i (Ji * sigmap^-1 *Ji)   \in IR^{3x3}
    b = -\sum_i Ji^T * sigmap^-1 *e0i  \in IR^{3x1}

ei(q) = (R*p2i+t-p1i) = R(theta) * p2i + [x] - p1i 
                                         [y]

Let's perturbe the model with deltaq = (deltax, deltay, deltath):
ei(q+deltaq) = (R*p2i+t-p1i) = [ cos(deltath), -sin(deltath)] * R(theta)] * p2i + [x + deltax] - p1i 
                               [ sin(deltath),  cos(deltath)]                     [y + deltay] 

ei ~=  ei(deltaq=0) + Ji * deltaq ~= (I + [      0, -deltath])* R(theta) * p2i + [x] + [deltax] - p1i
                                          [deltath,        0]                    [y]   [deltay]

    = (R(theta) * p2i + [x] - p1i) +  deltath * [ 0, -1] * R(theta) * p2i + [deltax]
                        [y]                     [ 1,  0]                    [deltay]

R(theta) * p2i = [p2i[0]*cos(theta) - p2i[1]*sin(theta)]
                 [p2i[0]*sin(theta) + p2i[1]*cos(theta)

Ji \in IR^{2x3} can be easily found from the last ei equation: 
Ji = dei/d(deltaq) = [1, 0,  -p2i[0]*sin(theta)-p2i[1]*cos(theta)]
                     [0, 1,   p2i[0]*cos(theta)-p2i[1]*sin(theta)]

Once the solution to the following system is found 
\sum_i Ji * sigmap^-1 *Ji * deltaq = -\sum_i Ji^T * sigmap^-1 *e0i
then we update: 
    [x    ] += deltax
    [y    ] += deltay
    [theta] += deltath    <-- here, we apply proper angle wrapping in order to keep the reprentation in [-pi,pi]

In matrix form:
    e = [e1,e2,...,eN] \in IR^{2*Nx1}
    J = d(e)/d(deltaq) \in IR^{2*Nx3} 
    J^T * sigmap^-1 * J * deltaq = -J^T *sigmap^-1 * e
    H * deltaq = b
    H =  J^T * sigmap^-1 * J   \in IR^{3x3}
    b = -J^T * sigmap^-1 * e   \in IR^{3x1}

===============================================
> Closed-form alignment of 2D points via SVD  
===============================================

ei(q)  = [x] + [cos(theta), -sin(theta)] * [p2ix] - [p1ix]   
         [y]   [sin(theta),  cos(theta)]   [p2iy]   [p1iy]    

ei(q+dq) = [x+dx] + exp([theta]z + [dtheta]z) * [p2ix] - [p1ix] =   
           [y+dy]                               [p2iy]   [p1iy]     
(rotations about the z-axis are commutative)
= [x+dx] + exp([dtheta]z) * exp([theta]z) * [p2ix] - [p1ix] 
  [y+dy]                                    [p2iy]   [p1iy]
~= [x+dx] + (I+[dtheta]z) * exp([theta]z) * [p2ix] - [p1ix] 
   [y+dy]                                   [p2iy]   [p1iy]
~= [x] + exp([theta]z) * [p2ix] - [p1ix] + [dx] + dtheta*[1]z * exp([theta]z) * [p2ix] 
   [y]                   [p2iy]   [p1iy]   [dy]                                 [p2iy]

Therefore, one has: 
dei/d(x,y,theta) = [1,0 | [1]z * exp([theta]z) * [p2ix] ] 
                   [0,1 |                        [p2iy] ]

lamdap = 1/(sigmap*sigmap)
lamda = sigmap^-1 = [lamdap,     0] = I2*lamdap where I2 = [1,0]
                    [     0,lamdap]                        [0,1]

L =  1/2 * \sum_i ei^T * lamda * ei
dL/dq = 0  => \sum_i ei^T * lamda * dei/dq = 0      [ (1x2)*(2x2)*(2x3) ]

If we just considider the derivative w.r.t. x and y: 
\sum_i ei^T * lamda * dei/d(x,y) = 0   =>  \sum_i ei^T * lamda = 0 =>  \sum_i ei = 0
=> N*[x] = \sum_i [p1ix] - \sum_i [cos(theta), -sin(theta)] * [p2ix]     
     [y]          [p2ix]          [sin(theta),  cos(theta)]   [p2iy]   
=> [x] = 1./N * \sum_i [p1ix] - R(theta) * 1./N * \sum_i  [p2ix]     
   [y]                 [p2ix]                              [p2iy]

We can replace this last equation in the single ei and in the cost function L and get: 
ei(q)  = R(theta) * ([p2ix] - 1./N * \sum_j  [p2jx]) - ([p1ix] - 1./N * \sum_j [p1jx])   
                    ([p2iy]                  [p2jy])   ([p1ix]                 [p1jy])  

Therefore, if we compute 
p1m = 1./N * \sum_i [p1ix]
                    [p1iy]

p2m = 1./N * \sum_i [p2ix]
                    [p2iy]

dp1i = p1i - p1m 
dp2i = p2i - p2m 

=> ei(theta) =  R(theta)*dp2i - dp1i                                      
                     

L = \sum_i   (R(theta)*dp2i - dp1i)^T * lamda * (R(theta)*dp2i - dp1i)
  = \sum_i dp2i^T*R(theta)^T*lamda*R(theta)*dp2i - 2*dp2i^T*R(theta)^T*lamda*dp1i + dp1i^T*dp1i       
  ( by using the trace trick: trace(A*B*C) = trace(C*A*B) = trace(B*C*A) and lambda = lamdap * I22 ) 
  = \sum_i lamdap*dp2i^T*dp2i - 2*lamdap*trace(dp1i*dp2i^T*R(theta)^T) + dp1i^T*dp1i 
  = c +  2*lamdap*trace(\sum_i (dp1i*dp2i^T)*R(theta)^T)    
    where c is a constant that does not depend on the variables under optimization

Let Q = \sum_i (dp1i*dp2i^T)  \in IR^{2x2}
L can be minimized by maximizing (we neglect the constanct factor 2*lamdap):
J = trace(Q*R(theta)^T)
If we decompose via SVD:
Q = U * S * V^T   where S = diag(s1,s2) with s1 >= s2 >= 0
then one has: 
J = trace(Q*R(theta)^T) = J = trace(U*S*V^T*R^T) = trace(S*V^T*R^T*U)  (by using the trace trick)
Given V and U are orthogonal matrices, it can be easily seen that a solution is:
R(theta) = U * D * V^T  
where D = diag(1,1) or D = diag(1,-1) and the +-1 is modulated in order to adjust det(R)>0.
Indeed, we cannot include S in the solution since we need an orthogonal matrix.
In fact, with such a solution 
J = trace(Q*R(theta)^T) = J = trace(S*V^T*R^T*U) = trace(S*D) = s1 +-s2  (+- depending on D selection)
which is constant and corresponds to the sum of the singular values.
In particular, we possibly need to adjust R in order to get det(R)>0 and we do that on the last term of D 
since we want the maximize J and the last term is the smaller one since: s2 <= s1. 

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

// q = [x, y, theta]
void rotoTranslatePoints(const VecVector2d& pin, const Eigen::Vector3d& q, VecVector2d& pout)
{
    pout.resize(pin.size()); 
    const double ctheta = cos(q[2]); 
    const double stheta = sin(q[2]);     
    for(size_t ii=0; ii<pin.size(); ii++)
    {
        pout[ii][0] = q[0] + pin[ii][0]*ctheta - pin[ii][1]*stheta;
        pout[ii][1] = q[1] + pin[ii][0]*stheta + pin[ii][1]*ctheta;
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

void alignIterative(const VecVector2d& p2, 
                    const VecVector2d& p1, 
                    const double sigmap, 
                    Eigen::Vector3d& q)
{
    assert(p1.size() == p2.size()); 

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
        
        const double ctheta = cos(q[2]);
        const double stheta = sin(q[2]);

        for(size_t ii=0; ii<p2.size(); ii++)
        {
            const double rp2i_x = p2[ii][0]*ctheta-p2[ii][1]*stheta;            
            const double rp2i_y = p2[ii][0]*stheta+p2[ii][1]*ctheta;

            // compute the new prediction 
            const Eigen::Vector2d p1i_pred(q[0] + rp2i_x, 
                                           q[1] + rp2i_y);
            const Eigen::Vector2d ei = p1i_pred-p1[ii];  
            
            /*
                Ji \in IR^{2x3}
                Ji = dei/d(deltaq) = [1, 0,  -p2i[0]*sin(theta)-p2i[1]*cos(theta)] = [1, 0,  -rp2_y]
                                     [0, 1,   p2i[0]*cos(theta)-p2i[1]*sin(theta)]   [0, 1,   rp2_x]                           
            */

            Ji << 1, 0, -rp2i_y, 
                  0, 1,  rp2i_x; 

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


void alignSVD(const VecVector2d& p2, 
              const VecVector2d& p1,  
              Eigen::Vector3d& q)
{
    assert(p1.size() == p2.size()); 

    /*
    Q = \sum_i (dp1i*dp2i^T)

    If we decompose via SVD:
    Q = U * S * V^T
    then a solution is 
    R(theta) = U * V^T  (here, we cannot include S since we need an orthogonal matrix for R)    
    */

    const size_t N = p2.size();
    Eigen::Vector2d p1m = Eigen::Vector2d::Zero();
    Eigen::Vector2d p2m = Eigen::Vector2d::Zero();
    for(size_t ii=0; ii<p2.size();ii++)
    {
        p1m += p1[ii];
        p2m += p2[ii];
    }
    p1m /= N;
    p2m /= N;

    Eigen::Matrix2d Q = Eigen::Matrix2d::Zero(); 
    for(size_t ii=0; ii<p2.size();ii++)
    {
        Q += (p1[ii]-p1m)*(p2[ii]-p2m).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d U = svd.matrixU();
    Eigen::Matrix2d V = svd.matrixV();
    Eigen::Matrix2d R = U*V.transpose();
    if(R.determinant()<0) 
    {
        R = U*Eigen::DiagonalMatrix<double,2>(1,-1)* V.transpose(); // we must guarantee det(R) = 1 > 0
    }

    /*
    => [x] = 1/N * \sum_i [p1ix] - R(theta) * 1/N * \sum_i  [p2ix]   = p1m - R(theta) * p2m  
       [y]                [p2ix]                            [p2iy]    
    */

    Eigen::Vector2d t = p1m - R * p2m;

#if 0
    std::cout << "R: \n" << R << std::endl; 
    std::cout << "det(R): " << R.determinant() << std::endl;  
    std::cout << "t: " << t.transpose() << std::endl; 
#endif 

    q[0]=t[0];
    q[1]=t[1];
    q[2] = atan2(R(1,0),R(0,0));
}


void drawScans(VecVector2d& p0, VecVector2d& pz, VecVector2d& pzalign1, VecVector2d& pzalign2) 
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


    auto drawScan = [&](VecVector2d& pv, GLint lineWidth, GLfloat red, GLfloat green, GLfloat blue)
    {
        glLineWidth(lineWidth);
        glBegin(GL_LINES);
        glColor3f(red, green, blue);        
        for (size_t i = 0; i < pv.size()-1; i++) 
        {  
            glVertex3d( pv[i][0],   pv[i][1], 0);
            glVertex3d(pv[i+1][0],pv[i+1][1], 0);
        }
        glEnd();        
    };

    while (pangolin::ShouldQuit() == false) 
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        drawScan(p0,       1, 1,0,0);
        drawScan(pz,       1, 0,1,0);
        drawScan(pzalign1, 2, 0,0,1);  
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

VecVector2d addCartesianNoiseToPoints(VecVector2d& p, double sigmap)
{
    VecVector2d pn; 
    pn.resize(p.size()); 

    std::default_random_engine generator;
    generator.seed(time(NULL));

    std::normal_distribution<double> distributionx(0.0,sigmap);
    std::normal_distribution<double> distributiony(0.0,sigmap);
    for(size_t ii=0; ii<p.size(); ii++)
    {
        pn[ii][0] = p[ii][0] + distributionx(generator);
        pn[ii][1] = p[ii][1] + distributiony(generator);        
    }
    return pn;
}

int main(int argc, char **argv) 
{
	// init random seed value
	srand((unsigned) time(NULL));
        
    const size_t N = 360; 
    const double rangeMin = 10; 
    const double rangeMax = 50; 
    const double sigmap = 0.01; // [m]

    std::vector<Range> z;
    initRange(z, N, rangeMin, rangeMax);

    Eigen::Vector3d q0;
    const double maxDeltaTranslation = 2.0;
    const double maxAngle = M_PI/4; 
    generateRandPose(q0,maxDeltaTranslation,maxAngle);    
    //q0 << 1.4,2.3,M_PI/6; // delta pose from origin that we need to recover via alignment
    VecVector2d p0; 
    rotoTranslateRange(z, q0, p0); // generate map points without noise 

    Eigen::Vector3d qz(0,0,0);
    VecVector2d pz;    
    rotoTranslateRange(z, qz, pz); // same range but starting from different pose 
    //VecVector2d pzn = pz;     
    VecVector2d pzn = addCartesianNoiseToPoints(pz, sigmap); // add noise to unaligned points 

    // set first guess
    Eigen::Vector3d qalign1 = qz; 
    Eigen::Vector3d qalign2 = qz;     
    // align noisy points pzn to map points p0
    alignIterative(pzn, p0, sigmap, qalign1);
    alignSVD(pzn, p0, qalign2);    

    std::cout << "q0: " << q0.transpose() << std::endl; 
    std::cout << "q1 (iterative): " << qalign1.transpose() << std::endl; 
    std::cout << "q2 (SVD): " << qalign2.transpose() << std::endl;     

    Eigen::Vector3d err1 = q0-qalign1;
    Eigen::Vector3d err2 = q0-qalign2;    
    err1[2] = angleDiff(q0[2],qalign1[2]);
    err2[2] = angleDiff(q0[2],qalign2[2]);    
    std::cout << "error1 (iterative): " << err1.norm() << std::endl; 
    std::cout << "error2 (SVD): " << err2.norm() << std::endl;     

    VecVector2d pzalign1, pzalign2;    
    rotoTranslatePoints(pzn, qalign1, pzalign1); 
    rotoTranslatePoints(pzn, qalign2, pzalign2);     

    drawScans(p0, pzn, pzalign1, pzalign2);

    return 0; 
}