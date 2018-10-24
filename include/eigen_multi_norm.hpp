#ifndef __EIGENMULTIVARIATENORMAL_HPP
#define __EIGENMULTIVARIATENORMAL_HPP

#include <Eigen/Dense>
#include <random>

namespace Eigen {
    namespace internal {
        template<typename Scalar>
        struct scalar_normal_dist_op
        {
            static std::mt19937 rng;                        // The uniform pseudo-random algorithm
            mutable std::normal_distribution<Scalar> norm; // gaussian combinator
	
            EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

            template<typename Index>
            inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
            inline void seed(const uint64_t &s) { rng.seed(s); }
        };

        template<typename Scalar>
        std::mt19937 scalar_normal_dist_op<Scalar>::rng;
      
        template<typename Scalar>
        struct functor_traits<scalar_normal_dist_op<Scalar> >
        { enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };

    } // end namespace internal
    
    template<typename Scalar>
    class EigenMultivariateNormal
    {
        Matrix<Scalar,Dynamic,Dynamic> _covar;
        Matrix<Scalar,Dynamic,Dynamic> _transform;
        Matrix< Scalar, Dynamic, 1> _mean;
        internal::scalar_normal_dist_op<Scalar> randN; // Gaussian functor
        bool _use_cholesky;
        SelfAdjointEigenSolver<Matrix<Scalar,Dynamic,Dynamic> > _eigenSolver;
    
    public:
        EigenMultivariateNormal(const Matrix<Scalar,Dynamic,1>& mean,const Matrix<Scalar,Dynamic,Dynamic>& covar,
                                const bool use_cholesky=false,const uint64_t &seed=std::mt19937::default_seed)
            :_use_cholesky(use_cholesky){
                randN.seed(seed);
                setMean(mean);
                setCovar(covar);
            }

        void setMean(const Matrix<Scalar,Dynamic,1>& mean) { _mean = mean; }
        Eigen::VectorXd getMean(){return _mean;}
        void setCovar(const Matrix<Scalar,Dynamic,Dynamic>& covar){
                _covar = covar;
                
                _eigenSolver = SelfAdjointEigenSolver<Matrix<Scalar,Dynamic,Dynamic> >(_covar);
                _transform = _eigenSolver.eigenvectors()*_eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
            }


        Matrix<Scalar,Dynamic,-1> samples(int nn){
                return (_transform * Matrix<Scalar,Dynamic,-1>::NullaryExpr(_covar.rows(),nn,randN)).colwise() + _mean;
            }
    }; 
}
#endif
