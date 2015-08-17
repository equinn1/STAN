// beta-binomial
data {
  int<lower=0> N;                 #number of binomial experiments
  int<lower=1>  n;                #number of trials in each binomial experiment
  int<lower=0,upper=n> y1[N];     #number of observed outcomes (binomial, uniform prior for p)
}
parameters {
real<lower=0,upper=1> p ;         #hyperprior for probability of success
real<lower=0,upper=1> p1[N];      #probability of success in binomial
}
model {
  p1 ~ beta(1,1);                 #separate uniform prior for each y1
  y1 ~ binomial(n,p1);            #given p1 y1 has a binomial distribution
}
generated quantities {
  int<lower=0,upper=n> y1p[N];    #posterior predictive score - year 1
  int<lower=0,upper=n> y2p[N];    #posterior predictive score - year 2
  int<lower=0,upper=n> y3p[N];    #posterior predictive score - year 3
  int<lower=0,upper=n> y4p[N];    #posterior predictive score - year 4
  int<lower=0,upper=n> y5p[N];    #posterior predictive score - year 5
  for (i in 1:N) {                #compute posterior predictive scores
    y1p[i]<-binomial_rng(n,p1[i]);  
    y2p[i]<-beta_binomial_rng(n,1+y1p[i],1+n-y1p[i]);  
    y3p[i]<-beta_binomial_rng(n,1+y1p[i]+y2p[i],1+2*n-y1p[i]-y2p[i]);
    y4p[i]<-beta_binomial_rng(n,1+y1p[i]+y2p[i]+y3p[i],1+3*n-y1p[i]-y2p[i]-y3p[i]);
    y5p[i]<-beta_binomial_rng(n,1+y1p[i]+y2p[i]+y3p[i]+y4p[i],1+4*n-y1p[i]-y2p[i]-y3p[i]-y4p[i]);
  }
}