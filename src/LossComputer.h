#include <RcppArmadillo.h>

class LossComputer {
private:
  arma::mat x; // {n x d} matrix
  arma::vec y; // {n x 1} vector
  arma::mat z; // {n x d} matrix

public:
  //LossComputer() {cout << "empty loss computer?\n";};
  //LossComputer(const LossComputer &tocopy) {cout << "copy constructor loss computer?\n";};

  LossComputer(const arma::mat &_x, const arma::vec &_y);
  double loss(arma::vec lambda);
  arma::vec loss_grad(arma::vec lambda);
};
