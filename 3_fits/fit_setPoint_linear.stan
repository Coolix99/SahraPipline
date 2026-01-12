functions {
    // Define the system of first-order ODEs 
    vector ode_system(real t, vector y, array[] real params) {
        real A = y[1];
        real g = y[2];
        real alpha = params[1];
        real beta_ = params[2];
        real A_end = params[3];

        
        vector[2] dydt;
        dydt[1] = g * A;  // dA/dt
        

        dydt[2] = -alpha * (g - beta_ * (A_end - A) / A_end);  // dg/dt 
      
        
        return dydt;
    }

    // Solve the ODE system
    array[] real A_theor(array[] real t, real A_0, real g_0, real alpha, real beta_, real A_end) {
        vector[2] y0 = [A_0, g_0]';
        array[3] real params = {alpha, beta_, A_end};

        array[num_elements(t)] vector[2] sol = ode_rk45(ode_system, y0, 47.999, t, params);
        array[num_elements(t)] real A;
    
        for (i in 1:num_elements(t)) {
            A[i] = sol[i, 1];  // Extract the first element (A) from the vector
        }
       
        return A;
    }

    // For use with individual time points
    real A_theor(real t, real A_0, real g_0, real alpha, real beta_, real A_end) {
        array[1] real t_arr = {t};
        return A_theor(t_arr, A_0, g_0, alpha, beta_, A_end)[1];
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
    real<lower=0> sigma_Dev;
    real<lower=0> sigma_rel_Dev;
    real<lower=0> sigma_Reg;
    real<lower=0> sigma_rel_Reg;

    real alpha_tilde;
    real beta_tilde;

    real A_end_tilde;

    real A_0_Dev_tilde;
    real g_0_Dev;
    real A_0_Reg_tilde;
    real g_0_Reg;
    }

transformed parameters {
   real alpha = 10^alpha_tilde;
   real beta_ = (alpha / 4) * 10^beta_tilde;

   real A_end = 10^A_end_tilde;

   real A_0_Dev = 10^A_0_Dev_tilde;
   real A_0_Reg = 10^A_0_Reg_tilde;



}

model {
    vector[N_Dev] Ahat_Dev;
    vector[N_Reg] Ahat_Reg;
    vector[N_Dev] sigma_all_Dev;
    vector[N_Reg] sigma_all_Reg;

    // Priors
    sigma_Dev ~ normal(0, 2.0);
    sigma_rel_Dev ~ normal(0, 0.1);
    sigma_Reg ~ normal(0, 2.0);
    sigma_rel_Reg ~ normal(0, 0.25);

    alpha_tilde ~ normal(-0.5, 0.5);
    beta_tilde ~ normal(0, 0.1);

    A_end_tilde ~ normal(1.0, 0.1);

    g_0_Dev ~ normal(0, 0.1);
    A_0_Dev_tilde ~ normal(0.3, 0.15);
    g_0_Reg ~ normal(0, 0.1);
    A_0_Reg_tilde ~ normal(0.3, 0.15);

    // Likelihood
    // ---- Solve ODE ONCE per condition ----
    Ahat_Dev = to_vector(
        A_theor(t_Dev, A_0_Dev, g_0_Dev, alpha, beta_, A_end)
    );

    Ahat_Reg = to_vector(
        A_theor(t_Reg, A_0_Reg, g_0_Reg, alpha, beta_, A_end)
    );

    // ---- Relative noise model ----
    sigma_all_Dev = sigma_Dev + sigma_rel_Dev .* Ahat_Dev;
    sigma_all_Reg = sigma_Reg + sigma_rel_Reg .* Ahat_Reg;

    // ---- Likelihood ----
    A_Dev ~ normal(Ahat_Dev, sigma_all_Dev);
    A_Reg ~ normal(Ahat_Reg, sigma_all_Reg);
}

generated quantities {
    vector[N_ppc] Ahat_Dev_ppc;
    vector[N_ppc] Ahat_Reg_ppc;
    vector[N_ppc] A_Dev_ppc;
    vector[N_ppc] A_Reg_ppc;
    Ahat_Dev_ppc = to_vector(
        A_theor(t_ppc, A_0_Dev, g_0_Dev, alpha, beta_, A_end)
    );

    Ahat_Reg_ppc = to_vector(
        A_theor(t_ppc, A_0_Reg, g_0_Reg, alpha, beta_, A_end)
    );

    for (i in 1:N_ppc) {
        A_Dev_ppc[i] = normal_rng(
            Ahat_Dev_ppc[i],
            sigma_Dev + sigma_rel_Dev * Ahat_Dev_ppc[i]
        );
        A_Reg_ppc[i] = normal_rng(
            Ahat_Reg_ppc[i],
            sigma_Reg + sigma_rel_Reg * Ahat_Reg_ppc[i]
        );
    }
}

