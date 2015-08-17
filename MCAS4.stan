// MCAS simulation
data {
  int<lower=0> N_theta;
  real         theta[N_theta];
  int<lower=1> N_R2S;
  int<lower=200,upper=280> r2s[N_R2S];
  int<lower=1> N_MC;
  real<lower=0> MC_mu_a[N_MC];
  real<lower=0> MC_sigma_a[N_MC];
  real          MC_mu_b[N_MC];
  real<lower=0> MC_sigma_b[N_MC];
  real          MC_mu_c[N_MC];
  real<lower=0> MC_sigma_c[N_MC];
  int<lower=1> N_SA;
  real<lower=0> SA_mu_a[N_SA];
  real<lower=0> SA_sigma_a[N_SA];
  real          SA_mu_b[N_SA];
  real<lower=0> SA_sigma_b[N_SA];
  int<lower=1> N_OR;
  real<lower=0> OR_mu_a[N_OR];
  real<lower=0> OR_sigma_a[N_OR];
  real          OR_mu_b[N_OR];
  real<lower=0> OR_sigma_b[N_OR];
  real          OR_mu_d0[N_OR];
  real<lower=0> OR_sigma_d0[N_OR];
  real          OR_mu_d1[N_OR];
  real<lower=0> OR_sigma_d1[N_OR];
  real          OR_mu_d2[N_OR];
  real<lower=0> OR_sigma_d2[N_OR];
  real          OR_mu_d3[N_OR];
  real<lower=0> OR_sigma_d3[N_OR];
}
parameters {
#  real theta;
  ordered[4] d[N_OR];
  real<lower=0,upper=6> MC_a[N_MC];
  real<lower=-5,upper=5> MC_b[N_MC];
  real<lower=0,upper=1> MC_c[N_MC];
  real<lower=0,upper=6> SA_a[N_SA];
  real<lower=-5,upper=5> SA_b[N_SA];
  real<lower=0,upper=6> OR_a[N_OR];
  real<lower=-5,upper=5> OR_b[N_OR];
}
model {
  theta ~ normal(0,1);
  for(i in 1:N_MC){
    MC_a[i] ~ normal(MC_mu_a[i],MC_sigma_a[i]);
    MC_b[i] ~ normal(MC_mu_b[i],MC_sigma_b[i]);
    MC_c[i] ~ normal(MC_mu_c[i],MC_sigma_c[i]);
  }
  for(i in 1:N_SA){
    SA_a[i] ~ normal(SA_mu_a[i],SA_sigma_a[i]);
    SA_b[i] ~ normal(SA_mu_b[i],SA_sigma_b[i]);
  }
  for(i in 1:N_OR){
    OR_a[i]  ~ normal(OR_mu_a[i],OR_sigma_a[i]);
    OR_b[i]  ~ normal(OR_mu_b[i],OR_sigma_b[i]);
    d[i,1] ~ normal(OR_mu_d3[i],OR_sigma_d3[i]);
    d[i,2] ~ normal(OR_mu_d2[i],OR_sigma_d2[i]);
    d[i,3] ~ normal(OR_mu_d1[i],OR_sigma_d1[i]);
    d[i,4] ~ normal(OR_mu_d0[i],OR_sigma_d0[i]);
  }
}
generated quantities {
  int<lower=0,upper=1> yMC[N_theta,N_MC];
  int<lower=0,upper=1> ySA[N_theta,N_SA];
  int<lower=0,upper=4> yOR[N_theta,N_OR];
  int<lower=0,upper=N_SA+N_MC+4*N_OR> ytot[N_theta];
  int<lower=200,upper=280> yscaled[N_theta];
  real ability[N_theta];

for(j in 1:N_theta){
  ytot[j] <- 0;
  for (i in 1:N_MC){
    yMC[j,i] <- bernoulli_rng(MC_c[i]+(1-MC_c[i])*inv_logit(MC_a[i]*(theta[j]-MC_b[i])));
    ytot[j] <- ytot[j]+yMC[j,i];
  }

  for (i in 1:N_SA){
    ySA[j,i] <- bernoulli_rng(inv_logit(SA_a[i]*(theta[j]-SA_b[i])));
    ytot[j] <- ytot[j]+ySA[j,i];
  }
  for (i in 1:N_OR){
    yOR[j,i] <- ordered_logistic_rng(OR_a[i]*(theta[j]-OR_b[i]),d[i]) - 1;
    ytot[j] <- ytot[j]+yOR[j,i];
  }
  ability[j] <- theta[j];
  yscaled[j] <- r2s[ytot[j]+1];
 }
}