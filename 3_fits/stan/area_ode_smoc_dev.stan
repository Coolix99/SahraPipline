functions {
  #include area_ode_functions.stanfunctions
}

data {
  real<lower=0> alpha_fixed;
  real<lower=0> beta_fixed;
  int<lower=1> N_smoc_dev;
  array[N_smoc_dev] real t_smoc_dev;
  vector[N_smoc_dev] A_smoc_dev;
  int<lower=1> N_ppc_48;
  array[N_ppc_48] real t_ppc_48;
}

parameters {
  real A_end_smoc_dev_tilde;
  real A_0_smoc_dev_tilde;
  real<lower=0> sigma_smoc_dev;
  real<lower=0> sigma_rel_smoc_dev;
}

transformed parameters {
  real alpha = alpha_fixed;
  real beta_ = beta_fixed;
  real A_end_smoc_dev = pow(10.0, A_end_smoc_dev_tilde);
  real A_0_smoc_dev = pow(10.0, A_0_smoc_dev_tilde);
  real g_0_smoc_dev = beta_ * (A_end_smoc_dev - A_0_smoc_dev) / A_end_smoc_dev;
}

model {
  vector[N_smoc_dev] Ahat_smoc_dev = to_vector(area_trajectory(
    t_smoc_dev, 47.999, A_0_smoc_dev, g_0_smoc_dev,
    alpha, beta_, A_end_smoc_dev));
  A_end_smoc_dev_tilde ~ normal(0.87, 0.1);
  A_0_smoc_dev_tilde ~ normal(0.3, 0.15);
  sigma_smoc_dev ~ normal(0, 2);
  sigma_rel_smoc_dev ~ normal(0, 0.25);
  A_smoc_dev ~ normal(
    Ahat_smoc_dev, sigma_smoc_dev + sigma_rel_smoc_dev .* Ahat_smoc_dev);
}

generated quantities {
  vector[N_ppc_48] Ahat_smoc_dev_48 = to_vector(area_trajectory(
    t_ppc_48, 47.999, A_0_smoc_dev, g_0_smoc_dev,
    alpha, beta_, A_end_smoc_dev));
  vector[N_ppc_48] A_smoc_dev_ppc;
  for (i in 1:N_ppc_48) {
    A_smoc_dev_ppc[i] = normal_rng(
      Ahat_smoc_dev_48[i], sigma_smoc_dev + sigma_rel_smoc_dev * Ahat_smoc_dev_48[i]);
  }
}
