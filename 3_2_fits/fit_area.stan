functions {
    // Define the system of first-order ODEs with A_cut condition
    vector ode_system(real t, vector y, array[] real params) {
        real A = y[1];
        real g = y[2];
        real alpha = params[1];
        real beta_ = params[2];
        real A_end = params[3];
        real A_cut = params[4];
        
        vector[2] dydt;
        dydt[1] = g * A;  // dA/dt
        
        if (A < A_cut) {
            dydt[2] = -alpha * (g - beta_ * (A_end - A_cut) / A_end);  // dg/dt for A < A_cut
        } else {
            dydt[2] = -alpha * (g - beta_ * (A_end - A) / A_end);  // dg/dt for A >= A_cut
        }
        
        return dydt;
    }

    // Solve the ODE system
    array[] real A_theor(array[] real t, real A_0, real g_0, real alpha, real beta_, real A_end, real A_cut) {
        vector[2] y0 = [A_0, g_0]';
        array[4] real params = {alpha, beta_, A_end, A_cut};

        array[num_elements(t)] vector[2] sol = ode_rk45(ode_system, y0, 47.999, t, params);
        array[num_elements(t)] real A;
    
        for (i in 1:num_elements(t)) {
            A[i] = sol[i, 1];  // Extract the first element (A) from the vector
        }
       
        return A;
    }

    // For use with individual time points
    real A_theor(real t, real A_0, real g_0, real alpha, real beta_, real A_end, real A_cut) {
        array[1] real t_arr = {t};
        return A_theor(t_arr, A_0, g_0, alpha, beta_, A_end, A_cut)[1];
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

    real alpha_tilde;
    real beta_tilde;

    real A_end_Dev_tilde;
    real A_cut_Dev_tilde;
    real A_end_Reg_tilde;
    real A_cut_Reg_tilde;

    real A_0_Dev_tilde;
    real g_0_Dev;
    real A_0_Reg_tilde;
    real g_0_Reg;
    }

transformed parameters {
   real alpha = 10^alpha_tilde;
   real beta_ = (alpha / 4) * 10^beta_tilde;

   real A_end_Dev = 10^A_end_Dev_tilde;
   real A_cut_Dev = 2 + 4 * inv_logit(A_cut_Dev_tilde);

   real A_end_Reg = 10^A_end_Reg_tilde;
   real A_cut_Reg = 2 + 4 * inv_logit(A_cut_Reg_tilde);

   real A_0_Dev = 10^A_0_Dev_tilde;
   real A_0_Reg = 10^A_0_Reg_tilde;
}

model {
   // Priors
    sigma ~ normal(0, 2.0);
    alpha_tilde ~ normal(-0.5, 0.5);
    beta_tilde ~ normal(0, 0.1);

    A_end_Dev_tilde ~ normal(1.0, 0.1);
    A_cut_Dev_tilde ~ normal(0, 1);  
    A_end_Reg_tilde ~ normal(1.0, 0.1);
    A_cut_Reg_tilde ~ normal(0, 1);  
        
    g_0_Dev ~ normal(0, 0.1);
    A_0_Dev_tilde ~ normal(0.3, 0.15);
    g_0_Reg ~ normal(0, 0.1);
    A_0_Reg_tilde ~ normal(0.3, 0.15);

   // Likelihood
    A_Dev ~ normal(A_theor(t_Dev, A_0_Dev, g_0_Dev, alpha, beta_, A_end_Dev, A_cut_Dev), sigma);
    A_Reg ~ normal(A_theor(t_Reg, A_0_Reg, g_0_Reg, alpha, beta_, A_end_Reg, A_cut_Reg), sigma);
}

generated quantities {
  array[N_ppc] real A_Dev_ppc = normal_rng(A_theor(t_ppc, A_0_Dev, g_0_Dev, alpha, beta_, A_end_Dev, A_cut_Dev), sigma);
  array[N_ppc] real A_Reg_ppc = normal_rng(A_theor(t_ppc, A_0_Reg, g_0_Reg, alpha, beta_, A_end_Reg, A_cut_Reg), sigma);
}
