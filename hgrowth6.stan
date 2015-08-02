// Hierarchical IRT 1PL for Colorado "growth" model simulation
data {
  int<lower=0> N_stu;                       # Number of students
  int<lower=0> N_dichot;                    # Number of dichotomous questions for each student 
  int<lower=0,upper=1> y1[N_dichot,N_stu];  # individual item scores - year 1
  int<lower=0,upper=1> y2[N_dichot,N_stu];  # individual item scores - year 2
}
parameters {
  vector[N_dichot] b1;                      #1PL difficulty parameters year 1
  vector[N_dichot] b2;                      #1PL difficulty parameters year 2
  vector[N_stu] theta;                      #1PL ability parameter theta
  real mu_b;                                #mean of hyperprior for b1,b2
  real<lower=0>  sigma_b;                   #sd of hyperprior for b1,b2
  real<lower=0>  sigma_theta;               #sd of hyperprior for theta
}
model {
//      hyperpriors
  mu_b ~ normal(0,2.0);                     #hyperprior for mean of b values
  sigma_b ~ cauchy(0,2.5);                  #hyperprior for sd of b values
  sigma_theta ~ cauchy(0,2.5);              #hyperprior for sd of theta values

//priors
  b1 ~ normal(mu_b, sigma_b);               #prior for b1
  b2 ~ normal(mu_b, sigma_b);               #prior for b2
  theta ~ normal(0,sigma_theta);            #prior for theta (assume E(theta)=0)
//likelihood
  for (i in 1:N_dichot) {                        #loop through the binary item scores
    y1[i] ~ bernoulli_logit((theta - b1[i]));    #year 1
    y2[i] ~ bernoulli_logit((theta - b1[i]));    #year 2
    }
}
generated quantities {
  int<lower=0,upper=N_dichot> y_tot1[N_stu];     #posterior predicted total score - year 1
  int<lower=0,upper=N_dichot> y_tot2[N_stu];     #posterior predicted total score - year 2

  for (i in 1:N_stu) {                           #compute posterior predictive item scores
    y_tot1[i]<-0;                                #and add them up
    y_tot2[i]<-0;
    for (j in 1:N_dichot){
        y_tot1[i]<-y_tot1[i]+bernoulli_rng(inv_logit(theta[i]-b1[j]));
        y_tot2[i]<-y_tot2[i]+bernoulli_rng(inv_logit(theta[i]-b2[j]));
    }
  }
}