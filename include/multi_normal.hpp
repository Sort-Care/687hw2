#ifndef __MVN
#define __MVN
class MVN
{
public:
    MVN(const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covar):
        mean(mean), covar(covar){};
    ~MVN(){};
    double prob_density(const Eigen::VectorXd& x) const;
    Eigen::VectorXd sample(unsigned int iter_num = 20) const;

    Eigen::VectorXd mean;
    Eigen::MatrixXd covar;
};
#endif
