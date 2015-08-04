// Hierarchical IRT 1PL for Colorado "growth" model simulation with multinormal thetas
data {
  int<lower=0> N_stu;                               # Number of students
  int<lower=0> N_years;                             # Number of years of data
  int<lower=0> N_dichot;                            # Number of dichotomous questions for each student 
  int<lower=0,upper=1>y[N_stu,N_years,N_dichot];    # individual item scores
}
parameters {
  corr_matrix[N_years] Omega[N_stu];                #prior correlation
  row_vector<lower=0>[N_years] tau[N_stu];          #prior scale
  real theta[N_stu,N_years];                        #abilities parameters - one for each year for every student
  real b[N_years,N_dichot];                         #difficulty parameters b
  real mu_b;                                        #mean for b values
  real<lower=0> sigma_b;                            #sd for b values
}
model {
  real years;
  matrix[N_years,N_years] Sigma[N_stu];             #variance-covariance matrix for thetas for student i
  for(i in 1:N_stu){
    Sigma[i]<-quad_form_diag(Omega[i],tau[i]);               #compute from correlation and scale factor
  }
//hyperpriors
  mu_b ~ normal(0,2.0);                             #hyperprior for mean of b values
  sigma_b ~ normal(0,2.5);                          #hyperprior for sd of b values
//priors
  years <- N_years;
  for(i in 1:N_stu){
//    tau[i] ~ cauchy(0,2.5);                              #half-cauchy prior for scale factor
    tau[i] ~ normal(0,1);
    Omega[i] ~ lkj_corr(years);                        #prior for correlation matrix
  }
  for(i in 1:N_years){
    for(j in 1:N_dichot){
      b[i,j] ~ normal(mu_b,sigma_b);                #priors for difficulty parameters
    }
  }

//likelihood
  for (i in 1:N_stu) {                              #loop through the observed binary item scores by student
    for (j in 1:N_years){                           #loop through years
      for (k in 1:N_dichot){
         y[i,j,k] ~ bernoulli_logit((theta[i,j] - b[j,k]));       #scores
      }
    }
  }
}
generated quantities {
  int<lower=0,upper=N_dichot> y_tot[N_stu];

  for (i in 1:N_stu) {
    y_tot[i]<-0;
    for (j in 1:N_dichot){
        y_tot[i]<-y_tot[i]+bernoulli_rng(inv_logit(theta[i,N_years]-b[N_years,j]));
    }
  }
}