#' Lattice Cutting Plane Algorithm using GLPK
#'
#' @param x feature matrix
#' @param y response vector
#' @param R_max max number of features (including intercept)
#' @param time_limit max running time in seconds
#' @export
lcpa_glpk <- function(x, y, R_max, time_limit) {
  lcpa_cpp(x, y, R_max, time_limit)
}
