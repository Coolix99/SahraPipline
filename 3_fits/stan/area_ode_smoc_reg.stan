functions {
  #include area_ode_functions.stanfunctions
}

data {
  real<lower=0> alpha_fixed;
  real<lower=0> beta_fixed;
  real<lower=0> A_end_fixed;
  int<lower=1> N_smoc_reg;
  array[N_smoc_reg] real t_smoc_reg;
  vector[N_smoc_reg] A_smoc_reg;
  int<lower=1> N_ppc_48;
  array[N_ppc_48] real t_ppc_48;
}

parameters {
  real A_0_smoc_reg_tilde;
  real<lower=0> sigma_smoc_reg;
  real<lower=0> sigma_rel_smoc_reg;
}

transformed parameters {
  real alpha = alpha_fixed;
  real beta_ = beta_fixed;
  real A_end = A_end_fixed;
  real A_0_smoc_reg = pow(10.0, A_0_smoc_reg_tilde);
}

model {
  vector[N_smoc_reg] Ahat_smoc_reg = to_vector(area_trajectory(
    t_smoc_reg, 47.999, A_0_smoc_reg, 0, alpha, beta_, A_end));
  A_0_smoc_reg_tilde ~ normal(0.3, 0.15);
  sigma_smoc_reg ~ normal(0, 2);
  sigma_rel_smoc_reg ~ normal(0, 0.25);
  A_smoc_reg ~ normal(
    Ahat_smoc_reg, sigma_smoc_reg + sigma_rel_smoc_reg .* Ahat_smoc_reg);
}

generated quantities {
  vector[N_ppc_48] Ahat_smoc_reg_48 = to_vector(area_trajectory(
    t_ppc_48, 47.999, A_0_smoc_reg, 0, alpha, beta_, A_end));
  vector[N_ppc_48] A_smoc_reg_ppc;
  for (i in 1:N_ppc_48) {
    A_smoc_reg_ppc[i] = normal_rng(
      Ahat_smoc_reg_48[i], sigma_smoc_reg + sigma_rel_smoc_reg * Ahat_smoc_reg_48[i]);
  }
}
