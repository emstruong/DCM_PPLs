functions {
  real Sigmodal(real x1,  real x2, real delta, real alpha ) {
       real S=(1.0/(1.0+exp(alpha*(x1-(delta*x2)))))-0.5;
    return S;
  }
}


data {
  int ns;                                       
  int nt;
  real dt; 
  int ds;
  real alpha;     
  real g_1;
  real g_2;
  real g_3;
  real g_4;
  real delta;
  real tau_i;
  real tau_e;
  real h_i;
  real h_e;
  real u;
  real obs_err;
}


transformed data {
  vector[ns] x_init;
  x_init=rep_vector(0.,ns);
  real dt_full = dt/ds;
  int nt_full = (nt-1)*ds + 1;

}


generated quantities {

    matrix[ns, nt_full] x;
    vector[ns] dx;
    array[nt] int nt_indexing = linspaced_int_array(nt, 1, nt_full);
    row_vector[nt_full] x_hat_full; 
    row_vector[nt] x_hat;    
    row_vector[nt] x_hat_ppc;    
    
    x[,1] = x_init;

    for (t in 1:(nt_full-1)) {
        dx[1]=x[4,t];
        dx[2]=x[5,t];
        dx[3]=x[6,t];
        dx[4]=inv(tau_e)*(h_e*(g_1*(Sigmodal(x[9,t], x[5,t]-x[6,t], delta, alpha))+u)-(inv(tau_e)*x[1,t])-2*x[4,t]);
        dx[5]=inv(tau_e)*(h_e*(g_2*(Sigmodal(x[1,t], x[4,t], delta, alpha)))-(inv(tau_e)*x[2,t])-2*x[5,t]);
        dx[6]=inv(tau_i)*(h_i*(g_4*(Sigmodal(x[7,t], x[8,t], delta, alpha)))-(inv(tau_i)*x[3,t])-2*x[6,t]);
        dx[7]=x[8,t];
        dx[8]=inv(tau_e)*(h_e*(g_3*(Sigmodal(x[9,t], x[5,t]-x[6,t], delta, alpha)))-(inv(tau_e)*x[7,t])-2*x[8,t]);
        dx[9]=x[5,t]-x[6,t];
        x[,t+1] = x[,t] + dt_full*dx; 

		} 
    
    x_hat_full=x[9, :];
    x_hat=x_hat_full[nt_indexing];

    for (t in 1:nt) {
         x_hat_ppc[t] = normal_rng(x_hat[t], obs_err); 
          }

}
