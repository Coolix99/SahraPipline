functions {
    vector ode_system(real t, vector y, array[] real params) {
        real A = y[1];
        real g = y[2];
        real alpha = params[1];
        real beta_ = params[2];
        real A_end = params[3];

        vector[2] dydt;
        dydt[1] = g * A;
        dydt[2] = -alpha * (g - beta_ * (A_end - A) / A_end);
        return dydt;
    }

    // Generic solver with custom start time
    array[] real A_theor_t0(
        array[] real t,
        real t0,
        real A_0,
        real g_0,
        real alpha,
        real beta_,
        real A_end
    ) {
        vector[2] y0 = [A_0, g_0]';
        array[3] real params = {alpha, beta_, A_end};

        array[num_elements(t)] vector[2] sol =
            ode_rk45(ode_system, y0, t0, t, params);

        array[num_elements(t)] real A;
        for (i in 1:num_elements(t))
            A[i] = sol[i, 1];

        return A;
    }
}


data {
    int N_Dev;
    int N_Reg;
    int N_4850cut;
    int N_7230cut;

    array[N_Dev] real t_Dev;
    vector[N_Dev] A_Dev;

    array[N_Reg] real t_Reg;
    vector[N_Reg] A_Reg;

    array[N_4850cut] real t_4850cut;
    vector[N_4850cut] A_4850cut;

    array[N_7230cut] real t_7230cut;
    vector[N_7230cut] A_7230cut;

    int<lower=1> N_ppc_48;
    array[N_ppc_48] real t_ppc_48;

    int<lower=1> N_ppc_72;
    array[N_ppc_72] real t_ppc_72;
}


parameters {
    real<lower=0> sigma_Dev;
    real<lower=0> sigma_rel_Dev;
    real<lower=0> sigma_Reg;
    real<lower=0> sigma_rel_Reg;
    real<lower=0> sigma_4850cut;
    real<lower=0> sigma_rel_4850cut;
    real<lower=0> sigma_7230cut;
    real<lower=0> sigma_rel_7230cut;

    real alpha_tilde;
    real beta_tilde;
    real A_end_tilde;

    real A_0_Dev_tilde;
    real g_0_Dev;
    real A_0_Reg_tilde;
    real g_0_Reg;
    real A_0_4850cut_tilde;
    real g_0_4850cut;
    real A_0_7230cut_tilde;
    real g_0_7230cut;
}


transformed parameters {
    real alpha = 10^alpha_tilde;
    real beta_ = (alpha / 4) * 10^beta_tilde;
    real A_end = 10^A_end_tilde;

    real A_0_Dev = 10^A_0_Dev_tilde;
    real A_0_Reg = 10^A_0_Reg_tilde;
    real A_0_4850cut = 10^A_0_4850cut_tilde;
    real A_0_7230cut = 10^A_0_7230cut_tilde;
}


model {
    vector[N_Dev] Ahat_Dev;
    vector[N_Reg] Ahat_Reg;
    vector[N_4850cut] Ahat_4850cut;
    vector[N_7230cut] Ahat_7230cut;

    // Priors
    alpha_tilde ~ normal(-0.5, 0.5);
    beta_tilde ~ normal(0, 0.1);
    A_end_tilde ~ normal(1.0, 0.1);

    g_0_Dev ~ normal(0, 0.1);
    g_0_Reg ~ normal(0, 0.1);
    g_0_4850cut ~ normal(0, 0.1);
    g_0_7230cut ~ normal(0, 0.1);

    A_0_Dev_tilde ~ normal(0.3, 0.15);
    A_0_Reg_tilde ~ normal(0.3, 0.15);
    A_0_4850cut_tilde ~ normal(0.3, 0.15);
    A_0_7230cut_tilde ~ normal(0.75, 0.15);

    sigma_Dev ~ normal(0, 2);
    sigma_Reg ~ normal(0, 2);
    sigma_4850cut ~ normal(0, 2);
    sigma_7230cut ~ normal(0, 2);

    sigma_rel_Dev ~ normal(0, 0.1);
    sigma_rel_Reg ~ normal(0, 0.25);
    sigma_rel_4850cut ~ normal(0, 0.25);
    sigma_rel_7230cut ~ normal(0, 0.25);

    // ODE solves
    Ahat_Dev =
        to_vector(A_theor_t0(t_Dev, 47.999, A_0_Dev, g_0_Dev, alpha, beta_, A_end));

    Ahat_Reg =
        to_vector(A_theor_t0(t_Reg, 47.999, A_0_Reg, g_0_Reg, alpha, beta_, A_end));

    Ahat_4850cut =
        to_vector(A_theor_t0(t_4850cut, 47.999, A_0_4850cut, g_0_4850cut, alpha, beta_, A_end));

    Ahat_7230cut =
        to_vector(A_theor_t0(t_7230cut, 71.999, A_0_7230cut, g_0_7230cut, alpha, beta_, A_end));

    // Likelihood
    A_Dev ~ normal(Ahat_Dev, sigma_Dev + sigma_rel_Dev .* Ahat_Dev);
    A_Reg ~ normal(Ahat_Reg, sigma_Reg + sigma_rel_Reg .* Ahat_Reg);
    A_4850cut ~ normal(Ahat_4850cut, sigma_4850cut + sigma_rel_4850cut .* Ahat_4850cut);
    A_7230cut ~ normal(Ahat_7230cut, sigma_7230cut + sigma_rel_7230cut .* Ahat_7230cut);
}


generated quantities {
    vector[N_ppc_48] A_Dev_ppc;
    vector[N_ppc_48] A_Reg_ppc;
    vector[N_ppc_48] A_4850cut_ppc;
    vector[N_ppc_72] A_7230cut_ppc;

    vector[N_ppc_48] Ahat_48;
    vector[N_ppc_72] Ahat_72;

    Ahat_48 =
        to_vector(A_theor_t0(t_ppc_48, 47.999, A_0_Dev, g_0_Dev, alpha, beta_, A_end));

    Ahat_72 =
        to_vector(A_theor_t0(t_ppc_72, 71.999, A_0_7230cut, g_0_7230cut, alpha, beta_, A_end));

    for (i in 1:N_ppc_48) {
        A_Dev_ppc[i] =
            normal_rng(Ahat_48[i], sigma_Dev + sigma_rel_Dev * Ahat_48[i]);
        A_Reg_ppc[i] =
            normal_rng(Ahat_48[i], sigma_Reg + sigma_rel_Reg * Ahat_48[i]);
        A_4850cut_ppc[i] =
            normal_rng(Ahat_48[i], sigma_4850cut + sigma_rel_4850cut * Ahat_48[i]);
    }

    for (i in 1:N_ppc_72) {
        A_7230cut_ppc[i] =
            normal_rng(Ahat_72[i], sigma_7230cut + sigma_rel_7230cut * Ahat_72[i]);
    }
}


