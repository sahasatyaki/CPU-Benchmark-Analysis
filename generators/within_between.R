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

# Run-to-run tackling within and between variation
# internal variance  << external variance
# see script terminal output:
# Internal
# [1] "var b1 = 0.359841    b2 = 0.218258"
# [1] "var b1 = 0.302897    b2 = 0.279302"
# ...
# [1] "var b1 = 0.264453    b2 = 0.264985"
# External
# [1] "VAR b1 = 10.841264    b2 = 11.294704"

# data generator
set.seed(2026)
R = 10 # number of benchmarks to run
r = 100 # number of API timings per each benchmark
prop = 0.4 # proportion used to simulate API noise

# data <- 'same' # generates bw_R10r100_s.csv and bw_R10r100_r.csv
data <- 'diff' # generates bw_R10r100_d.csv and bw_R10r100_r.csv

# For illustration
mat = T # T: show each individual benchmark (internal variability)
# mat = F # F: show agregated benchmarks view external variability

# No changes beyond this point ############

M <- rnorm(R, mean=10, sd=4)
m1 <- rep(0.1, R)
m2_same <- rep(0.3, R)

m2_diff <- rep(2, R)

if (data == 'same') {
  m2 <- m2_same
  name=sprintf("bw_R%ir%i_s.csv", R,r) 
} else {
  m2 <- m2_diff
  name=sprintf("bw_R%ir%i_d.csv", R,r) 
}


sw = runif(R) >= prop
b1 = matrix(0, nrow=r, ncol=R)
b2 = matrix(0, nrow=r, ncol=R)
# dev.off()
if (mat) {
  layout(matrix(1:10, 5, 2))
} else {
  layout(1)
}
par(oma=c(5,1,1,1), mar=c(5,1,1,1))
gmin = Inf
gmax = -Inf
for (i in seq(R)){
  b1[,i] <- rnorm(r, mean=m1[i] + M[i], sd=0.6)
  b2[,i] <- rnorm(r, mean=m2[i] + M[i], sd=0.5)
  if (sw[i]) {
    tmp <- c(b1[,i], b2[,i])
    b1[,i] <- sample(tmp, r)
    b2[,i] <- sample(tmp, r)
  }
  gmin = min(c(gmin, b1[,i], b2[,i]))
  gmax = max(c(gmax, b1[,i], b2[,i]))
  print(sprintf("var b1 = %f    b2 = %f", var(b1[,i]), var(b2[,i])))
}
  print(sprintf("VAR b1 = %f    b2 = %f", var(as.numeric(b1)), var(as.numeric(b2))))

if (!mat){  
  plot(NA, ylim=c(gmin,gmax), xlim=c(1, r))
}
for (i in seq(R)){
  if (mat) {
    ylim=c(min(c(b1[,i], b2[,i])),max(c(b1[,i], b2[,i])))
    plot(b1[,i], col='blue', ylim=ylim)
  } else {
    points(b1[,i], col='blue')
  }
  points(b2[,i], col='red')
}
  colnames(b1) <- paste0("bm_", seq(R))
  colnames(b2) <- colnames(b1)
  write.csv(x=b1, file=sprintf("bw_R%ir%i_r.csv", R,r), row.names=F)
  write.csv(x=b2, file=name, row.names=F)