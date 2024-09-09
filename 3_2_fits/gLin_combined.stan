functions {
    // Define the system of first-order ODEs with L_cut condition
    vector ode_system(real t, vector y, array[] real params) {
        real L = y[1];
        real g = y[2];
        real alpha = params[1];
        real beta_ = params[2];
        real L_end = params[3];
        real L_cut = params[4];
        
        vector[2] dydt;
        dydt[1] = g * L;  // dL/dt
        
        if (L < L_cut) {
            dydt[2] = -alpha * (g - beta_ * (L_end - L_cut) / L_end);  // dg/dt for L < L_cut
        } else {
            dydt[2] = -alpha * (g - beta_ * (L_end - L) / L_end);  // dg/dt for L >= L_cut
        }
        
        return dydt;
    }

    // Solve the ODE system
    array[] real L_theor(array[] real t, real L_0, real g_0, real alpha, real beta_, real L_end, real L_cut) {
        vector[2] y0 = [L_0, g_0]';
        array[4] real params = {alpha, beta_, L_end, L_cut};

        array[num_elements(t)] vector[2] sol = ode_rk45(ode_system, y0, 47.999, t, params);
        array[num_elements(t)] real L;
    
        for (i in 1:num_elements(t)) {
            L[i] = sol[i, 1];  // Extract the first element (L) from the vector
        }
       
        return L;
    }

    // For use with individual time points
    real L_theor(real t, real L_0, real g_0, real alpha, real beta_, real L_end, real L_cut) {
        array[1] real t_arr = {t};
        return L_theor(t_arr, L_0, g_0, alpha, beta_, L_end, L_cut)[1];
    }
}

data {
   int N_Dev;
   int N_Reg;

   array[N_Dev] real t_Dev;
   vector[N_Dev] L_Dev;

   array[N_Reg] real t_Reg;
   vector[N_Reg] L_Reg;

   int<lower=1> N_ppc;
   array[N_ppc] real t_ppc;
}

parameters {
   real<lower=0> sigma;
   
   real alpha_tilde;
   real beta_tilde;
   real L_end_tilde;
   real L_cut_tilde;

   real L_0_Dev_tilde;
   real g_0_Dev;
   real L_0_Reg_tilde;
   real g_0_Reg;
}

transformed parameters {
   real alpha = 10^alpha_tilde;
   real beta_ = (alpha / 4) * 10^beta_tilde;
   real L_end = 10^L_end_tilde;
   real L_cut = 2 + 4 * inv_logit(L_cut_tilde);

   real L_0_Dev = 10^L_0_Dev_tilde;
   real L_0_Reg = 10^L_0_Reg_tilde;
}

model {
   // Priors
    sigma ~ normal(0, 2.0);
    alpha_tilde ~ normal(-0.5, 0.5);
    beta_tilde ~ normal(0, 0.1);
    L_end_tilde ~ normal(1.0, 0.1);
    L_cut_tilde ~ normal(0, 1);  
        
    g_0_Dev ~ normal(0, 0.1);
    L_0_Dev_tilde ~ normal(0.3, 0.15);
    g_0_Reg ~ normal(0, 0.1);
    L_0_Reg_tilde ~ normal(0.3, 0.15);

   // Likelihood
    L_Dev ~ normal(L_theor(t_Dev, L_0_Dev, g_0_Dev, alpha, beta_, L_end, L_cut), sigma);
    L_Reg ~ normal(L_theor(t_Reg, L_0_Reg, g_0_Reg, alpha, beta_, L_end, L_cut), sigma);
}

generated quantities {
  array[N_ppc] real L_Dev_ppc = normal_rng(L_theor(t_ppc, L_0_Dev, g_0_Dev, alpha, beta_, L_end, L_cut), sigma);
  array[N_ppc] real L_Reg_ppc = normal_rng(L_theor(t_ppc, L_0_Reg, g_0_Reg, alpha, beta_, L_end, L_cut), sigma);
}
