// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// lcpa_cpp
Rcpp::List lcpa_cpp(arma::mat x, arma::vec y, int R_max, int time_limit);
RcppExport SEXP _riskslimr_glpk_lcpa_cpp(SEXP xSEXP, SEXP ySEXP, SEXP R_maxSEXP, SEXP time_limitSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type R_max(R_maxSEXP);
    Rcpp::traits::input_parameter< int >::type time_limit(time_limitSEXP);
    rcpp_result_gen = Rcpp::wrap(lcpa_cpp(x, y, R_max, time_limit));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_riskslimr_glpk_lcpa_cpp", (DL_FUNC) &_riskslimr_glpk_lcpa_cpp, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_riskslimr_glpk(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}