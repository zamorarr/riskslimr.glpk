#include "LossComputer.h"

LossComputer::LossComputer(const arma::mat &_x, const arma::vec &_y):
  x(_x), y(_y) {
  z = x.each_col() % y;
}

double LossComputer::loss(arma::vec lambda) {
  return arma::mean(arma::log(1 + arma::exp(-z * lambda)));
}

arma::vec LossComputer::loss_grad(arma::vec lambda) {
  arma::vec b = 1 + arma::exp(z * lambda); // {n x 1} vector
  arma::mat a = z.each_col() / b;
  arma::mat lg = arma::mean(-a, 0); // col means
  return arma::vectorise(lg);
}
