ctypedef unsigned int uint

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng:
        pass
    ctypedef struct gsl_rng_type:
        pass
    gsl_rng* gsl_rng_alloc(gsl_rng_type*)
    gsl_rng_type* gsl_rng_taus
    void gsl_rng_set(gsl_rng*, uint)
    double gsl_rng_uniform(gsl_rng*)
    
cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gamma(gsl_rng*, double, double)
    double gsl_ran_gaussian(gsl_rng*, double)
    double gsl_ran_beta(gsl_rng*, double, double)
    double gsl_ran_gamma_pdf(double, double)
    uint gsl_ran_poisson(gsl_rng*, double)	
    ctypedef struct gsl_ran_discrete_t:
        pass
    gsl_ran_discrete_t* gsl_ran_discrete_preproc(uint K, double*)
    uint gsl_ran_discrete(gsl_rng*, gsl_ran_discrete_t*)
    

cdef extern from "gsl/gsl_sf.h":
     double gsl_sf_lngamma(double)
     double gsl_sf_gamma(double)
     double gsl_sf_beta(double, double)
     double gsl_sf_lnbeta(double, double)

cdef extern from "gsl/gsl_statistics_double.h":
    double gsl_stats_max(double *, uint stride, uint n)
    double gsl_stats_min(double *,uint stride, uint n)
    uint gsl_stats_max_index(double *,uint stride, uint n)
    uint gsl_stats_min_index(double *,uint stride, uint n)