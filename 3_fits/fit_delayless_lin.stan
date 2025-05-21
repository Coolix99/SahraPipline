functions {
  // Piecewise-linear g(A)
  real g_A(real A, real beta_, real A_end) {
      return beta_ * (A_end - A) / A_end;
  }

  // Simplified ODE system: dA/dt = A * g(A)
  vector ode_system(real t, vector y, array[] real params) {
    real A = y[1];
    real beta_ = params[1];
    real A_end = params[2];

    vector[1] dydt;
    dydt[1] = A * g_A(A, beta_, A_end);
    return dydt;
  }

  array[] real A_theor(array[] real t, real A_0, real beta_, real A_end) {
    vector[1] y0 = [A_0]';
    array[2] real params = {beta_, A_end};

    array[num_elements(t)] vector[1] sol = ode_rk45(ode_system, y0, 47.999, t, params);
    array[num_elements(t)] real A;
    for (i in 1:num_elements(t))
      A[i] = sol[i, 1];
    return A;
  }

  real A_theor(real t, real A_0, real beta_, real A_end) {
    array[1] real t_arr = {t};
    return A_theor(t_arr, A_0, beta_, A_end)[1];
  }
}

data {
  int N_Dev;
  int N_Reg;

  array[N_Dev] real t_Dev;
  vector[N_Dev] A_Dev;

  array[N_Reg] real t_Reg;
  vector[N_Reg] A_Reg;

  int<lower=1> N_ppc;
  array[N_ppc] real t_ppc;
}

parameters {
  real<lower=0> sigma;

  real beta_tilde;
  real A_end_tilde;


  real A_0_Dev_tilde;
  real A_0_Reg_tilde;
}

transformed parameters {
  real beta_ = 10 ^ beta_tilde;
  real A_end = 10 ^ A_end_tilde;

  real A_0_Dev = 10 ^ A_0_Dev_tilde;
  real A_0_Reg = 10 ^ A_0_Reg_tilde;
}

model {
  // Priors
  sigma ~ normal(0, 2.0);
  beta_tilde ~ normal(-1.0, 0.5);
  A_end_tilde ~ normal(1.0, 0.1);
  A_0_Dev_tilde ~ normal(0.3, 0.15);
  A_0_Reg_tilde ~ normal(0.3, 0.15);

  // Likelihood
  A_Dev ~ normal(A_theor(t_Dev, A_0_Dev, beta_, A_end), sigma);
  A_Reg ~ normal(A_theor(t_Reg, A_0_Reg, beta_, A_end), sigma);
}

generated quantities {
  array[N_ppc] real A_Dev_ppc = normal_rng(A_theor(t_ppc, A_0_Dev, beta_, A_end), sigma);
  array[N_ppc] real A_Reg_ppc = normal_rng(A_theor(t_ppc, A_0_Reg, beta_, A_end), sigma);
}
