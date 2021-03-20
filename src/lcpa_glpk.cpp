#include <RcppArmadillo.h>
#include "LossComputer.h"
#include "glpk.h"

void lazycut_cb(glp_tree *T, void *info) {
  switch(glp_ios_reason(T)) {
  case GLP_IROWGEN: {
    //Rcpp::Rcout << "=======================" << std::endl;

    // cast info pointer to appropriate type
    LossComputer *computer = static_cast<LossComputer*>(info);

    // get subproblem
    glp_prob *p = glp_ios_get_prob(T);

    double obj_val = glp_get_obj_val(p);
    double gap = glp_ios_mip_gap(T);
    //Rcpp::Rcout << "obj val: " << obj_val << std::endl;
    //Rcpp::Rcout << "gap: " << gap << std::endl;

    // number of variables
    int m = glp_get_num_cols(p);
    int d = (m - 3)/2;

    // lambda vec
    arma::vec lambda(d);
    //Rcpp::Rcout << "lambda: [";
    for (int i = 1; i <= d; i++) {
      lambda[i - 1] = glp_get_col_prim(p, d + i);
      //Rcpp::Rcout << lambda[i - 1] << ", ";
    }
    //Rcpp::Rcout << "]" << std::endl;

    double loss_actual = computer->loss(lambda);
    arma::vec loss_slope = computer->loss_grad(lambda);

    // current loss from cutting plane approximation
    double loss_approx = glp_get_col_prim(p, 2*d + 2);
    //Rcpp::Rcout << "loss (cutting plane): " << loss_approx << std::endl;
    //Rcpp::Rcout << "loss (actual): " << loss_actual << std::endl;


    if (std::fabs(loss_actual - loss_approx) <= 1E-6) {
      //Rcpp::Rcout << "close enough, not adding cutting plane" << std::endl;
      return;
    }

    // add constraint
    // L >= loss_actual  + loss_slope*(lambda - lambda_k);
    // L - loss_slope*lambda >= loss_actual - loss_slope*lambda_k
    //Rcpp::Rcout << "adding constraint" << std::endl;
    int n = glp_get_num_rows(p);
    //Rcpp::Rcout << "num rows: " << n << std::endl;

    //glp_add_rows(p, 1);
    double lb = loss_actual - arma::sum(loss_slope % lambda);
    //Rcpp::Rcout << "new lb: " << lb << std::endl;

    glp_add_rows(p, 1);
    glp_set_row_bnds(p, n + 1, GLP_LO, lb, 0.0);
    int aj[1 + (d + 1)];
    double av[1 + (d + 1)];

    for (int i = 1; i <= d; i++) {
      aj[i] = d + i, av[i] = -loss_slope[i - 1];
    }
    aj[d + 1] = 2*d + 2, av[d + 1] = 1.0;
    glp_set_mat_row(p, n + 1, d + 1, aj, av);
    //glp_set_mat_row(p, n + 1, m, aj, av);

    //Rcpp::Rcout << "=======================" << std::endl;
    break;
  }
    //default:
    //Rcpp::Rcout << "other callback" << std::endl;
  }

  return;
}

void lcpa_add_vars(glp_prob* mip, int d, int R_max, int intercept_min, int intercept_max, int coef_min, int coef_max) {
  int num_vars = 2*d + 3;

  glp_add_cols(mip, num_vars);
  std::string varname;

  // alpha
  for (int i = 1; i <= d; i++) {
    varname = "alpha" + std::to_string(i);
    glp_set_col_name(mip, i, varname.c_str());
    glp_set_col_kind(mip, i, GLP_BV);
  }

  // lambda
  for (int i = 1; i <= d; i++) {
    varname = "lambda" + std::to_string(i);
    glp_set_col_name(mip, d + i, varname.c_str());
    glp_set_col_kind(mip, d + i, GLP_IV);

    if (i == 1) {
      glp_set_col_bnds(mip, d + i, GLP_DB, (double) intercept_min, (double) intercept_max);
    } else {
      glp_set_col_bnds(mip, d + i, GLP_DB, (double) coef_min, (double) coef_max);
    }
  }

  // R
  glp_set_col_name(mip, 2*d + 1, "R");
  glp_set_col_bnds(mip, 2*d + 1, GLP_DB, 1.0, R_max);
  glp_set_col_kind(mip, 2*d + 1, GLP_IV);

  // L
  glp_set_col_name(mip, 2*d + 2, "L");
  glp_set_col_bnds(mip, 2*d + 2, GLP_LO, 0.0, 0.0);

  // V
  glp_set_col_name(mip, 2*d + 3, "V");
  glp_set_col_bnds(mip, 2*d + 3, GLP_LO, 0.0, 0.0);
  glp_set_obj_coef(mip, 2*d + 3, 1.0);
}

void lcpa_add_constraints(glp_prob* mip, int d) {
  int num_constraints = 2*d + 2;
  glp_add_rows(mip, num_constraints);

  // lambda[j] - alpha[j]*coef_min >= 0
  // lambda[j] - alpha[j]*coef_max <= 0
  for (int i = 1; i <= d; i++) {
    glp_set_row_bnds(mip, i, GLP_LO, 0.0, 0.0);
    glp_set_row_bnds(mip, d + i, GLP_UP, 0.0, 0.0);
  }

  // R = sum(alpha)
  glp_set_row_bnds(mip, 2*d + 1, GLP_FX, 0.0, 0.0);

  // V = L + c0*R
  glp_set_row_bnds(mip, 2*d + 2, GLP_FX, 0.0, 0.0);
}

void lcpa_add_matrix(glp_prob* mip, int d, int intercept_min, int intercept_max, int coef_min, int coef_max) {
  int mi[1 + 1000], mj[1 + 1000];
  double mv[1 + 1000];

  double c0 = 1E-8;

  // constraint matrix is size (2d + 2) x (2d + 3)
  // we don't have to fill in the zero values though
  int k = 1;

  // constraints 1:d are the lower bound of lambda
  int val = 0;
  for (int i = 1; i <= d; i++) {
    if (i == 1) {
      val = intercept_min;
    } else {
      val = coef_min;
    }

    mi[k] = i, mj[k] = i, mv[k] = -(double) val; // alpha[i]
    mi[k + 1] = i, mj[k + 1] = i + d, mv[k + 1] = 1.0; // lambda[i]
    k += 2;
  }

  // constraints (d+1):2*d are the upper bound of lambda
  for (int i = 1; i <= d; i++) {
    if (i == 1) {
      val = intercept_max;
    } else {
      val = coef_max;
    }

    mi[k] = i + d, mj[k] = i, mv[k] = -(double) val; // alpha[i]
    mi[k + 1] = i + d, mj[k + 1] = i + d, mv[k + 1] = 1.0; // lambda[i]
    k += 2;
  }

  // constraint 2d + 1 is R = sum(alpha)
  mi[k] = 2*d + 1, mj[k] = 2*d + 1, mv[k] = -1.0; // R
  k++;
  for (int i = 1; i <= d; i++) {
    mi[k] = 2*d + 1, mj[k] = i, mv[k] = 1.0; //alpha[i]
    k++;
  }

  // constraint 2d + 2 is V = L + c0*R
  mi[k] = 2*d + 2, mj[k] = 2*d + 3, mv[k] = 1.0; // V
  k++;
  mi[k] = 2*d + 2, mj[k] = 2*d + 2, mv[k] = -1.0; // L
  k++;
  mi[k] = 2*d + 2, mj[k] = 2*d + 1, mv[k] = -c0; // R

  // load constraint matrix
  glp_load_matrix(mip, k, mi, mj, mv);
}

// [[Rcpp::export]]
Rcpp::List lcpa_cpp(arma::mat x, arma::vec y, int R_max = 3, int time_limit = 60) {
  // create problem
  glp_prob *mip;
  mip = glp_create_prob();
  glp_set_prob_name(mip, "lcpa");
  glp_set_obj_dir(mip, GLP_MIN);

  // initialize the loss computer
  // add intercept to x beforehand
  LossComputer computer(x, y);
  int d = x.n_cols;

  // create callback
  glp_iocp parm;
  glp_init_iocp(&parm);
  parm.cb_func = lazycut_cb;
  parm.cb_info = &computer;
  parm.tm_lim = time_limit * 1000; // milliseconds

  // num of variables in x matrix
  int coef_min = -5;
  int coef_max = 5;
  int intercept_min = -10;
  int intercept_max = 10;

  // setup
  lcpa_add_vars(mip, d, R_max, intercept_min, intercept_max, coef_min, coef_max);
  lcpa_add_constraints(mip, d);
  lcpa_add_matrix(mip, d, intercept_min, intercept_max, coef_min, coef_max);

  // write problem
  //glp_write_lp(mip, NULL, "problem.lp");
  //glp_write_mps(mip, GLP_MPS_FILE, NULL, "problem.mps");
  //glp_write_mip(mip, "mip.lp");

  // trying to fix numerical instability
  glp_smcp lp_parm;
  glp_init_smcp(&lp_parm);
  lp_parm.meth = GLP_DUALP;

  // solve problem
  int ret;
  ret = glp_simplex(mip, &lp_parm);
  //ret = glp_simplex(mip, &s);
  if (ret == 0) {
    glp_intopt(mip, &parm);
  } else {
    return Rcpp::List::create(
      Rcpp::Named("error_code") = ret
    );
  }

  // extract solution
  int status = glp_mip_status(mip);
  double obj_val = glp_mip_obj_val(mip);

  std::vector<int> alpha(d);
  std::vector<int> lambda(d);
  for (int i = 1; i <= d; i++) {
    alpha[i - 1] = glp_mip_col_val(mip, i);
    lambda[i - 1] = glp_mip_col_val(mip, i + d);
  }

  // cleanup problem
  glp_delete_prob(mip);

  // return value
  return Rcpp::List::create(
    Rcpp::Named("is_optimal") = status == GLP_OPT,
    //Rcpp::name("gap") = glp_ios_mip_gap(mip);
    Rcpp::Named("objective_value") = obj_val,
    Rcpp::Named("optimality_gap") = -1,
    Rcpp::Named("alpha") = alpha,
    Rcpp::Named("lambda") = lambda
  );
}
