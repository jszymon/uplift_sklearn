require(predtools)

d <- gusto

# reverse the hyp indicator variable such that 1 corresponds to sysbp >= 100
d$hyp <- 1-d$hyp

# remove tpa an indicator of tPA treatment (can be inferred from tx)
d$tpa <- NULL

# remove ant variable which is an indicator anterior MI (included in miloc)
d$ant <- NULL

# change pmi to 0/1 binary indicator
d$pmi <- as.integer(pmi)-1

# subtract 1 from htn to make it 0/1 binary indicator
d$htn <- d$htn-1

# subtract 1 from pan to make it 0/1 binary indicator
d$pan <- d$pan-1

# subtract 1 from fam to make it 0/1 binary indicator
d$fam <- d$fam-1

write.csv(d, "/tmp/GUSTO.csv", quote=FALSE, row.names=FALSE)
