# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 1.
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. 2. Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. 3. Neither the name of the copyright holder nor the
# names of its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Data generator for
# 1. Regime change data, and
# 2. Compare two populations

# See bottom of script for driver

# No changes beyond this point ############

rm(list=ls())

# Warm-up functions
w_none <- function(n, A=0, c=0, w=0) {
  return(rep(0, n))
}

w_warmup <- function(n, A=2, c=200, w=200) {
  # t > 0  : time steps
  # A >= 0 : amplitude
  # c>0    : time it's cold
  # w>0    : time it takes to warm up
  # signal amplitude goes to zero at time c+w onward
  # c+w <= n
  es = seq(0, 10, length.out=w)
  e = exp(es - 5)
  ew = (A / 2 ) * (1 - (e - 1) / (e + 1))
  s = rep(NA, n)
  s[1:c] = A
  s[(c+1):(c+w)] = ew
  s[(c+w+1):n] = 0
  return(s)
}


# Generators ####
# Use binormal with hp$rate=1
generator.normal <- function(hp, m, samplesm, warm=list(gen=none, A=0, c=0, w=0)) {
  n = m * samplesm
  obs = rep(0, n)
  for (i in seq(m)) {
    gidx = seq((i - 1) * samplesm + 1, i * samplesm)
    obs[gidx] = rnorm(samplesm, mean = hp$mean, sd = hp$sd)
  }
  w = warm$gen(n=n, A=warm$A, c=warm$c, w=warm$w)
  wname = sprintf("WarmA%2.3f_c%i_w_%i", warm$A, warm$c, warm$w)
  l = list(d = w+sample(obs), name=sprintf('Norm_m%2.3f_s%2.3f_%s', hp$mean, hp$sd, wname))
  return(l)
}

generator.exp <- function(hp, m, samplesm, warm=list(gen=none, A=0, c=0, w=0)) {
  n = m * samplesm
  obs = rep(0, n)
  for (i in seq(m)) {
    gidx = seq((i - 1) * samplesm + 1, i * samplesm)
    obs[gidx] = hp$bump + rexp(samplesm, rate = hp$rate)
  }
  w = warm$gen(n=n, A=warm$A, c=warm$c, w=warm$w)
  wname = sprintf("WarmA%2.3f_c%i_w_%i", warm$A, warm$c, warm$w)
  l = list(d = w+sample(obs), name=sprintf('Exp_rate%2.3f_bump=%2.3f_%s', hp$rate, hp$bump, wname))
  return(l)
}

generator.binormal <- function(hp, m, samplesm, warm=list(gen=none, A=0, c=0, w=0)) {
  n = m * samplesm
  obs = rep(0, n)
  for (i in seq(m)) {
    gidx = seq((i - 1) * samplesm + 1, i * samplesm)
    primary = ifelse(runif(1) < hp$frac, 1, 0)
    secondary = !primary
    hm = hp$m + secondary * hp$m.sec
    hs = (primary * hp$sd) + (secondary * hp$sd.sec)
    obs[gidx] = rnorm(samplesm, mean = hm, sd = hs)
  }
  w = warm$gen(n=n, A=warm$A, c=warm$c, w=warm$w)
  wname = sprintf("WarmA%2.3f_c%i_w_%i", warm$A, warm$c, warm$w)
  l = list(d = w+sample(obs), name=sprintf('BIN_Fr%2.3f_mpri%2.3f_spri%2.3f_msec%2.3f_ssec%2.3f_%s', hp$frac, hp$m, hp$sd, hp$m.sec, hp$sd.sec, wname))
  return(l)
}

generate_data_set <- function(generator = generator.normal,
                              hp = list(mean=1, sd=0.64),
                              m = 20,
                              samplesm = 100,
                              warm = list(gen=none, A=0, c=0, w=0),
                              save = FALSE,
                              saveplot = FALSE, debug = FALSE, prefix='') {
  
  # m = 50 # number of times to repeat a benchmark
  # samplesm = 300 # number of times each benchmark measures the API
  n <- m * samplesm
  
  gen <- generator(hp=hp, m=m, samplesm=samplesm, warm=warm)
  d <- gen$d
  gname <- gen$name
  name = sprintf("%sbm_n%i_Gen%s", prefix, n, gname)
  means = rep(0, m)
  # print(match.call())
  
  if (saveplot) {
    print(sprintf("saving PLOT in %s.png", name))
    png(file = sprintf("%s.png", name), width = 1200, height = 500)
  }
  
  if (debug) {
    layout(matrix(c(1, 1, 1, 2, 3, 4), 2, 3, byrow = TRUE))
    plot(d, pch = 16, col = adjustcolor("blue", alpha.f = 0.5))
    grid()
    title(
      sprintf(
        "Benchark repeats: %i. Each benchmark reports %i measurements",
        m,
        samplesm
      )
    )
    hist(d, freq = FALSE)
    abline(v = mean(d),
           col = "red",
           lwd = 5)
    
    for (i in seq(m)) {
      gidx = seq((i - 1) * samplesm + 1, i * samplesm)
      means[i] = mean(d[gidx])
    }
    hm = mean(means)
    hs = sd(means)
    
    de <- density(means)
    dn <- list(x = de$x, y = dnorm(x = de$x, mean = hm, sd = hs))
    hist(means, freq = !TRUE)
    lines(de,
          col = adjustcolor("royalblue", alpha.f = 0.95),
          lwd = 5)
    lines(dn, col = adjustcolor("black", alpha.f = 0.5), lwd = 5)
    abline(v = hm,
           col = "red",
           lwd = 5)
    
    qqnorm(y = means - hm)
    abline(
      a = 0,
      b = hs,
      col = adjustcolor("magenta", alpha.f = 0.5),
      lwd = 5
    )
  } else {
    layout(matrix(1, 1, 1))
    plot(d,
         pch = 16,
         col = adjustcolor("royalblue", alpha.f = 0.5))
    title(sprintf(
      "Benchark generator %s with %i measurements",
      gname,
      m * samplesm
    ))
    grid()
  }
  if (save) {
    # save the benchmark numbers
    name = sprintf("%s.csv", name)
    print(sprintf("saving data in %s", name))
    write.csv(d, file=name,row.names = F)
  }
  if (saveplot) {
    dev.off()
  }
}


# Driver #####

easy_warmup <- list(gen=w_warmup, A=2, c=200, w=200)
hard_warmup <- list(gen=w_warmup, A=0.76, c=200, w=1300)
none <- list(gen=w_none, A=0, c=0, w=0)

 # save also a graph of the data
#generate_data_set(generator = generator.normal, hp = list(mean=12.3, sd=0.34),m = 25,samplesm = 101,warm = none,save = T,saveplot = saveplot)
#generate_data_set(generator = generator.normal, hp = list(mean=11.8, sd=0.64),m = 20,samplesm = 100,warm = hard_warmup,save = T,saveplot = saveplot)
#generate_data_set(generator = generator.exp, hp = list(rate=0.8, bump=10),m = 20,samplesm = 100,warm = hard_warmup,save = T,saveplot = saveplot) # bm_n2000_GenExp_rate0.800_bump=10.000_WarmA0.760_c200_w_1300.csv
#generate_data_set(generator = generator.binormal, hp = list(frac=0.85, m=10, sd=1, m.sec=15, sd.sec=0.7),m = 20,samplesm = 100,warm = hard_warmup,save = T,saveplot = saveplot)
#generate_data_set(generator = generator.binormal, hp = list(frac=0.85, m=10, sd=1, m.sec=16, sd.sec=0.7),m = 20,samplesm = 100,warm = easy_warmup,save = T,saveplot = saveplot)
# generate_data_set(generator = generator.normal, hp = list(mean=12.3, sd=0.34),m = 25,samplesm = 101,warm = easy_warmup,save = T,saveplot = saveplot)
# generate_data_set(generator = generator.binormal, hp = list(frac=0.8, m=13, sd=1, m.sec=5.3, sd.sec=1.6),m = 25,samplesm = 120,warm = none, debug=T)
generate_data_set(generator = generator.binormal, hp = list(frac=0.8, m=13, sd=1, m.sec=3.3, sd.sec=0.3),m = 25,samplesm = 120,warm = none, debug=T)

# Regime change data-sets
saveplot=T
do_regime_chg = F # Save data?
if (do_regime_chg) {
  generate_data_set(generator = generator.binormal, hp = list(frac=1, m=12.3, sd=0.34, m.sec=0, sd.sec=0),m = 25,samplesm = 101,warm = easy_warmup,save = T,saveplot = saveplot) # bm_n2525_GenBIN_Fr1.000_mpri12.300_spri0.340_msec0.000_ssec0.000_WarmA2.000_c200_w_200.csv
  generate_data_set(generator = generator.binormal, hp = list(frac=0.9, m=10.3, sd=0.54, m.sec=2, sd.sec=0.33),m = 25,samplesm = 111,warm = hard_warmup,save = T,saveplot = saveplot) # bm_n2775_GenBIN_Fr0.900_mpri10.300_spri0.540_msec2.000_ssec0.330_WarmA0.760_c200_w_1300.csv
  generate_data_set(generator = generator.binormal, hp = list(frac=0.95, m=15.3, sd=0.4, m.sec=1, sd.sec=1.33),m = 30,samplesm = 200,warm = none,save = T,saveplot = saveplot)# bm_n6000_GenBIN_Fr0.950_mpri15.300_spri0.400_msec1.000_ssec1.330_WarmA0.000_c0_w_0.csv
  generate_data_set(generator = generator.binormal, hp = list(frac=0.5, m=5.3, sd=0.6, m.sec=3, sd.sec=0.1),m = 28,samplesm = 131,warm = easy_warmup,save = T,saveplot = saveplot) # bm_n3668_GenBIN_Fr0.500_mpri5.300_spri0.600_msec3.000_ssec0.100_WarmA2.000_c200_w_200.csv
  generate_data_set(generator = generator.binormal, hp = list(frac=0.5, m=3, sd=0.2, m.sec=5.3, sd.sec=0.6),m = 15,samplesm = 120,warm = hard_warmup,save = T,saveplot = saveplot) # bm_n1800_GenBIN_Fr0.500_mpri3.000_spri0.200_msec5.300_ssec0.600_WarmA0.760_c200_w_1300.csv
  generate_data_set(generator = generator.binormal, hp = list(frac=0.3, m=13, sd=1, m.sec=5.3, sd.sec=1.6),m = 25,samplesm = 120,warm = hard_warmup,save = T,saveplot = saveplot) # bm_n3000_GenBIN_Fr0.300_mpri13.000_spri1.000_msec5.300_ssec1.600_WarmA0.760_c200_w_1300.csv
  generate_data_set(generator = generator.binormal, hp = list(frac=0.3, m=13, sd=1, m.sec=5.3, sd.sec=1.6),m = 25,samplesm = 120,warm = easy_warmup,save = T,saveplot = saveplot) # bm_n3000_GenBIN_Fr0.300_mpri13.000_spri1.000_msec5.300_ssec1.600_WarmA2.000_c200_w_200.csv
  generate_data_set(generator = generator.binormal, hp = list(frac=0.3, m=10.1, sd=2, m.sec=2.3, sd.sec=1.3),m = 17,samplesm = 130,warm = none,save = T,saveplot = saveplot) # bm_n2210_GenBIN_Fr0.300_mpri10.100_spri2.000_msec2.300_ssec1.300_WarmA0.000_c0_w_0.csv
}

# Compare two API (and determine regression) data-sets 
do_cmp = F
if (do_cmp) { # Save data?
  # GEMM Same
  prefix = 'set_gemm_v1_vs_v2/gemm_v1_'
  generate_data_set(generator = generator.exp, hp = list(rate=0.8, bump=10),m = 20,samplesm = 100,warm = hard_warmup,save = T,saveplot = saveplot, prefix=prefix) 
  prefix = 'set_gemm_v1_vs_v2/gemm_v2_'
  generate_data_set(generator = generator.exp, hp = list(rate=0.6, bump=9.9),m = 19,samplesm = 110,warm = easy_warmup,save = T,saveplot = saveplot, prefix=prefix) 
  # GEMM Diff
  prefix = 'set_gemm_v1_vs_v3/gemm_v1_'
  generate_data_set(generator = generator.exp, hp = list(rate=5, bump=3),m = 15,samplesm = 120,warm = hard_warmup,save = T,saveplot = saveplot, prefix=prefix) 
  prefix = 'set_gemm_v1_vs_v3/gemm_v3_'
  generate_data_set(generator = generator.exp, hp = list(rate=6, bump=2.75),m = 22,samplesm = 90,warm = easy_warmup,save = T,saveplot = saveplot, prefix=prefix)
  
  # TRSV Same
  prefix = 'set_trsv_v1_vs_v2/trsv_v1_'
  generate_data_set(generator = generator.binormal, hp = list(frac=0.8, m=13, sd=1, m.sec=5.3, sd.sec=1.6),m = 27,samplesm = 120,warm = none, debug=F, save=T,saveplot=saveplot,prefix = prefix)
  prefix = 'set_trsv_v1_vs_v2/trsv_v2_'
  generate_data_set(generator = generator.binormal, hp = list(frac=0.8, m=13, sd=1, m.sec=3.3, sd.sec=0.3),m = 25,samplesm = 120,warm = none, debug=F, save=T,saveplot=saveplot,prefix = prefix)
  
  prefix = 'set_trsv_v1_vs_v3/trsv_v1_'
  generate_data_set(generator = generator.binormal, hp = list(frac=0.8, m=12, sd=1.1, m.sec=4.3, sd.sec=1.6),m = 28,samplesm = 130,warm = hard_warmup, debug=F, save=T,saveplot=saveplot,prefix = prefix)
  prefix = 'set_trsv_v1_vs_v3/trsv_v3_'
  generate_data_set(generator = generator.binormal, hp = list(frac=0.9, m=15, sd=0.8, m.sec=1.5, sd.sec=0.01),m = 23,samplesm = 110,warm = hard_warmup, debug=F, save=T,saveplot=saveplot,prefix = prefix)
  
  prefix = 'set_elastic_vs_kt/kt_'
  generate_data_set(generator = generator.binormal, hp = list(frac=0.6, m=12, sd=1.1, m.sec=2.3, sd.sec=0.6),m = 31,samplesm = 130,warm = none, debug=F, save=T,saveplot=saveplot,prefix = prefix)
  prefix = 'set_elastic_vs_kt/elastic_'
  generate_data_set(generator = generator.binormal, hp = list(frac=1, m=16, sd=0.53, m.sec=0, sd.sec=0),m = 29,samplesm = 90,warm = none, debug=F, save=T,saveplot=saveplot,prefix = prefix)
  
  prefix = 'set_vect372/run_a_'
  generate_data_set(generator = generator.binormal, hp = list(frac=1, m=16, sd=0.53, m.sec=0, sd.sec=0),m = 29,samplesm = 90,warm = hard_warmup, debug=F, save=T,saveplot=saveplot,prefix = prefix)
  prefix = 'set_vect372/run_b_'
  generate_data_set(generator = generator.binormal, hp = list(frac=1, m=16, sd=0.53, m.sec=0, sd.sec=0),m = 29,samplesm = 90,warm = none, debug=F, save=T,saveplot=saveplot,prefix = prefix)
  
  prefix = 'set_vect455/run_a_'
  generate_data_set(generator = generator.binormal, hp = list(frac=1, m=7, sd=1.53, m.sec=0, sd.sec=0),m = 31,samplesm = 95,warm = hard_warmup, debug=F, save=T,saveplot=saveplot,prefix = prefix)
  
  prefix = 'set_vect455/run_b_'
  generate_data_set(generator = generator.binormal, hp = list(frac=1, m=7, sd=1.53, m.sec=0, sd.sec=0),m = 31,samplesm = 95,warm = easy_warmup, debug=F, save=T,saveplot=saveplot,prefix = prefix)
  }