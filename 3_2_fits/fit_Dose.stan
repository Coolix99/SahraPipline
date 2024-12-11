functions {
    vector ode_system(real t, vector y, array[] real params) {
        real A = y[1];
        real D = y[2];
        real g = y[3];
        
        real C0 = params[1];
        real C1 = params[2];
        real D_decrease = params[3];
        real g_max = params[4];
        real tau_g = params[5];
        
        real D_end = 10 + D_decrease;

        vector[3] dydt;
        
        real g_theory;
        if (D < 10) {
            g_theory = g_max;
        } else if (D < D_end) {
            g_theory = g_max * (D_end - D) / (D_end - 10);
        } else {
            g_theory = 0.0;
        }
        
        dydt[1] = A * g;
        dydt[2] = C0 + C1 * A;
        dydt[3] = -(g - g_theory)/tau_g;

        return dydt;
    }

    array[] real A_theor(array[] real t, real A_0, real D_0, real g_0, real C0, real C1, real D_decrease, real g_max, real tau_g, real t_start) {
        vector[3] y0 = [A_0, D_0, g_0]';
        array[5] real params = {C0, C1, D_decrease, g_max, tau_g};

        array[num_elements(t)] vector[2] sol = ode_rk45(ode_system, y0, t_start, t, params);
        array[num_elements(t)] real A;

        for (i in 1:num_elements(t)) {
            A[i] = sol[i, 1];
        }
       
        return A;
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

    real C0;
    real C1;
    real D_decrease_tilde;
    real g_max_tilde;
    real tau_g_tilde;

    real A_0_Dev_tilde;
    real g_0_Dev;
    real A_0_Reg_tilde;
    real D_0_Reg;
    real g_0_Reg;
}

transformed parameters {
    real tau_g = 10^tau_g_tilde;
    real D_decrease = 10^D_decrease_tilde;
    real g_max = 10^g_max_tilde;

    real A_0_Dev = 10^A_0_Dev_tilde;
    real A_0_Reg = 10^A_0_Reg_tilde;
}

model {
    sigma ~ normal(0, 2.0);
    C0 ~ normal(0.1, 0.2);
    C1 ~ normal(0.15, 0.1);
    D_decrease_tilde ~ normal(1.0, 0.5);
    g_max_tilde ~ normal(-1.3, 0.3);
    tau_g_tilde ~ normal(0.55, 0.1);

    A_0_Dev_tilde ~ normal(0.3, 0.15);
    g_0_Dev ~ normal(0, 0.1);

    A_0_Reg_tilde ~ normal(0.3, 0.15);
    D_0_Reg ~ normal(0.0, 5.0);
    g_0_Reg ~ normal(0, 0.1);
   
    A_Dev ~ normal(A_theor(t_Dev, A_0_Dev, 0,       g_0_Dev, C0, C1, D_decrease, g_max,tau_g, 47.999), sigma);
    A_Reg ~ normal(A_theor(t_Reg, A_0_Reg, D_0_Reg, g_0_Reg, C0, C1, D_decrease, g_max,tau_g, 47.999), sigma);
}

generated quantities {
    array[N_ppc] real A_Dev_ppc = normal_rng(A_theor(t_ppc, A_0_Dev, 0,       g_0_Dev, C0, C1, D_decrease, g_max, tau_g, 47.999), sigma);
    array[N_ppc] real A_Reg_ppc = normal_rng(A_theor(t_ppc, A_0_Reg, D_0_Reg, g_0_Reg, C0, C1, D_decrease, g_max, tau_g, 47.999), sigma);
}
