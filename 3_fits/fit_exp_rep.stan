functions {
  // --- g_target(E): piecewise-linear activation ---
  real g_target(real E, real E_cut, real E_end) {
    if (E < E_cut) {
      return 1.0;
    } else if (E < E_end) {
      return (E_end - E) / (E_end - E_cut);
    } else {
      return 0.0;
    }
  }

  // --- s(C): sigmoid repression function ---
  real s_C(real C, real s_max, real C_swap) {
    return s_max / (1 + exp(C - C_swap));
  }

  // --- C(A,E): sigmoidal morphogen-dependent signal ---
  real C_fun(real A, real E, real C0) {
    return C0 / sinh(A / E);
  }

  // --- ODE system: [A, g, E] ---
  vector ode_system(real t, vector y, array[] real params) {
    real A = y[1];
    real g = y[2];
    real E = y[3];

    real tau_A   = params[1];
    real tau_g   = params[2];
    real g_0     = params[3];
    real tau_E   = params[4];
    real E_cut   = params[5];
    real E_end   = params[6];
    real k       = params[7];
    real s_max   = params[8];
    real C_swap  = params[9];
    real C0      = params[10];

    real C     = C_fun(A, E, C0);
    real g_t   = g_0 * g_target(E, E_cut, E_end);
    real s_val = s_C(C, s_max, C_swap);

    vector[3] dydt;
    dydt[1] = A * g / tau_A;
    dydt[2] = (g_t - g) / tau_g;
    dydt[3] = (-k * E + s_val) / tau_E;
    return dydt;
  }

  array[] real A_theor(array[] real t, real A0, real g0, real E0, array[] real params) {
    vector[3] y0 = [A0, g0, E0]';
    array[num_elements(t)] vector[3] sol = ode_rk45(ode_system, y0, 47.999, t, params);
    array[num_elements(t)] real A_out;
    for (i in 1:num_elements(t)) {
      A_out[i] = sol[i, 1];
    }
    return A_out;
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

  real<lower=0> tau_A;
  real<lower=0> tau_g;
  real g_0;
  real<lower=0> tau_E;

  real<lower=0> E_cut;
  real<lower=0> E_end;

  real<lower=0> k;
  real<lower=0> s_max;
  real<lower=0> C_swap;
  real<lower=0> C0;

  real<lower=0> A0_Dev;
  real<lower=0> g0_Dev;
  real<lower=0> E0_Dev;

  real<lower=0> A0_Reg;
  real<lower=0> g0_Reg;
  real<lower=0> E0_Reg;
}

model {
  // Priors
  sigma ~ normal(0, 2.0);
  g_0 ~ normal(0, 0.1);

  tau_A ~ lognormal(0.5, 0.3);
  tau_g ~ lognormal(0.5, 0.3);
  tau_E ~ lognormal(0.5, 0.3);

  E_cut ~ lognormal(log(0.35), 0.3);
  E_end ~ lognormal(log(1.0), 0.3);

  k ~ lognormal(log(0.3), 0.3);
  s_max ~ lognormal(log(0.75), 0.3);
  C_swap ~ lognormal(log(1.0), 0.3);
  C0 ~ lognormal(log(3.0), 0.3);

  A0_Dev ~ uniform(0.5, 1.5);
  g0_Dev ~ uniform(0.2, 1.0);
  E0_Dev ~ uniform(0.5, 1.5);

  A0_Reg ~ uniform(0.5, 1.5);
  g0_Reg ~ uniform(0.2, 1.0);
  E0_Reg ~ uniform(0.5, 1.5);

  // Data fit
  A_Dev ~ normal(A_theor(t_Dev, A0_Dev, g0_Dev, E0_Dev,
               {tau_A, tau_g, g_0, tau_E, E_cut, E_end, k, s_max, C_swap, C0}), sigma);
  A_Reg ~ normal(A_theor(t_Reg, A0_Reg, g0_Reg, E0_Reg,
               {tau_A, tau_g, g_0, tau_E, E_cut, E_end, k, s_max, C_swap, C0}), sigma);
}

generated quantities {
  array[N_ppc] real A_Dev_ppc = normal_rng(
    A_theor(t_ppc, A0_Dev, g0_Dev, E0_Dev,
      {tau_A, tau_g, g_0, tau_E, E_cut, E_end, k, s_max, C_swap, C0}), sigma);

  array[N_ppc] real A_Reg_ppc = normal_rng(
    A_theor(t_ppc, A0_Reg, g0_Reg, E0_Reg,
      {tau_A, tau_g, g_0, tau_E, E_cut, E_end, k, s_max, C_swap, C0}), sigma);
}
