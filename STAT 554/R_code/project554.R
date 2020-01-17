setwd('C:/Users/46541/Desktop/myhw/554/new_project')
library(INLA)
library(maps)
library(shapefiles)
library(maptools)
library(rgdal)
library(glm2)
library(SpatialEpi)
library(RColorBrewer)
library(spdep)
library(rgdal)
library(readxl)
#####################################################################
data1<-read.table("ohio.txt",header=TRUE)
head(data1)
data2<-read_xls("ohio2.xls")
head(data2)
data2<-data.frame(data2$FIPS,data2$county,data2$GDP,data2$`HIV prevalence`,
                  data2$`Drug overdose death`,data2$`Median Household income`,
                  data2$`Health behavior score`)
cnames<-c("FIPS","County","GDP","HIV prevalence","Drug overdose death",
          "Median Houshold income","Health behavior score")
colnames(data2)<-cnames
head(data2)

ohio<- readShapePoly(fn="ohio_map",
                     proj4string=CRS("+proj=longlat"),repair=T)

# match data1 and data2
o_12 <- match(data1$fips,data2$FIPS)
data2<-data2[o_12,]
row.names(data2)<-row.names(data1)
data_match<- data.frame(data1,data2)
head(data_match)
# match data_match and ohio
o <- match(ohio$CNTYIDFP00, data_match$fips)
data_match<-data_match[o,]
row.names(data_match) <- row.names(ohio)
ohio_new<-spCbind(ohio,data_match)

W_cont_el <-poly2nb(ohio_new, queen=T)
W_cont_el_mat<-nb2listw(W_cont_el, style="W", zero.policy=TRUE)
moran.test(ohio_new$Drug.overdose.death, listw=W_cont_el_mat,zero.policy=T)
# death_rate###########################
ohio_new$death_rate<-ohio_new$Drug.overdose.death/ohio_new$N
moran.test(ohio_new$death_rate, listw=W_cont_el_mat,zero.policy=T)
###############################
ohio_new$HIV_rate <-ohio_new$HIV.prevalence/ohio_new$N
moran.test(ohio_new$HIV_rate, listw=W_cont_el_mat,zero.policy=T)

moran.test(ohio_new$Median.Houshold.income, listw=W_cont_el_mat,zero.policy=T)
moran.test(ohio_new$GDP, listw=W_cont_el_mat,zero.policy=T)

moran.test(ohio_new$Health.behavior.score, listw=W_cont_el_mat,zero.policy=T)



#############################################################
hist(ohio_new$HIV.prevalence,prob=TRUE, col="grey",main="HIV prevalence")
hist(ohio_new$Drug.overdose.death,prob=TRUE, col="grey",main="Drug overdose death")

spplot(ohio_new,"Health.behavior.score",col.regions = colorRampPalette (rev ( brewer.pal (11,"RdBu"))) (50) )

spplot ( ohio_new ,"Median.Houshold.income",col.regions = colorRampPalette (rev ( brewer.pal (11 ,"RdBu"))) (50) )

spplot ( ohio_new ,"N",col.regions = colorRampPalette (rev ( brewer.pal (11 ,"RdBu"))) (50) )

spplot ( ohio_new ,"GDP",col.regions = colorRampPalette (rev ( brewer.pal (8 ,"RdBu"))) (50) )

spplot ( ohio_new ,"HIV.prevalence",col.regions = colorRampPalette (rev ( brewer.pal (11 ,"RdBu"))) (50) )

spplot ( ohio_new ,"Drug.overdose.death",col.regions = colorRampPalette (rev ( brewer.pal (11 ,"RdBu"))) (50) )

spplot ( ohio_new ,"death_rate",col.regions = colorRampPalette (rev ( brewer.pal (11 ,"RdBu"))) (50) )

spplot ( ohio_new ,"HIV_rate",col.regions = colorRampPalette (rev ( brewer.pal (11 ,"RdBu"))) (50) )

#################
plot(ohio)#  ???county????????? ??????????????????!!!!!
library(ggplot2) # For map_data. It's just a wrapper; should just use maps.
library(sp)
library(maps)
getLabelPoint <- # Returns a county-named list of label points
  function(county) {Polygon(county[c('long', 'lat')])@labpt}
df <- map_data('county', 'ohio') # NC region county data
centroids <- by(df, df$subregion, getLabelPoint) # Returns list
centroids <- do.call("rbind.data.frame", centroids) # Convert to Data Frame
names(centroids) <- c('long', 'lat') # Appropriate Header
map('county', 'ohio')
text(centroids$long, centroids$lat, rownames(centroids), offset=0, cex=0.8)
##########################

xx0 = which(ohio$CNTYIDFP00 ==39001) #adams
xx1 = which(ohio$CNTYIDFP00 == 39145) #Scioto
xx2 = which(ohio$CNTYIDFP00 == 39015)  #Brown
xx3 = which(ohio$CNTYIDFP00 == 39131)  #Pike
xx4 = which(ohio$CNTYIDFP00 == 39071)  #Highland
# plot the whole state
plot(ohio, border = "#00000075")
# highlight counties of interest
plot(ohio[xx0, ], col = "#ff000075", add = T)
plot(ohio[xx1, ], col = NA ,border = "#0000ff75",
     add = T, lwd = 2.5)
plot(ohio[xx2, ], col = NA, border = "#0000ff75",
     add = T, lwd = 2.5)
plot(ohio[xx3, ], col = NA, border = "#0000ff75",
     add = T, lwd = 2.5)
plot(ohio[xx4, ], col = NA, border = "#0000ff75",
     add = T, lwd = 2.5)
# Add some labels
text(coordinates(ohio[xx0, ]), "A", cex = 0.75,
     pos = 3, offset = 0.25)
text(coordinates(ohio[xx1, ]), "S", cex = 0.7,
     pos = 4, offset = 0.25)
text(coordinates(ohio[xx2, ]), "B", cex = 0.7,
     pos = 1, offset = 0.25)
text(coordinates(ohio[xx3, ]), "P", cex = 0.7,
     pos = 4, offset = 0.25)
text(coordinates(ohio[xx4, ]), "H", cex = 0.7,
     pos = 2, offset = 0.25)
points(coordinates(ohio[c(xx0, xx1,xx2,xx3,xx4), ]), pch = 16,
       cex = 0.5)
#################################

fit.l <- lm(death_rate~HIV_rate,data=ohio_new)
summary(fit.l)
AIC(fit.l)
res.fit.l <- fit.l$residuals
moran.test(res.fit.l,listw=W_cont_el_mat,zero.policy=T)
plot(fit.l)
#???????????????
ohio_new$res<-res.fit.l
spplot ( ohio_new ,"res",col.regions = colorRampPalette (rev ( brewer.pal (11 ,"RdBu"))) (50) )


###############spatial lag
fit.sar <-lagsarlm(death_rate~HIV_rate, data= ohio_new, listw=W_cont_el_mat,zero.policy=T, tol.solve=1e-12)
summary(fit.sar)
AIC(fit.sar)

res.fit.sar<-fit.sar$residuals
ohio_new$res2<-res.fit.sar
spplot ( ohio_new ,"res2",col.regions = colorRampPalette (rev ( brewer.pal (11 ,"RdBu"))) (50) )
moran.test(res.fit.sar,listw=W_cont_el_mat,zero.policy=T)
#### fitted plot of linear model without spatial
plot(fit.l$fitted.values,ohio_new$death_rate,
     xlab="fitted death rate",ylab="actual death rate",
     main="Fitted plot (without spatial lag term)")
fit11<-lm(ohio_new$death_rate~fit.l$fitted.values)
abline(a=fit11$coefficients[1],b=fit11$coefficients[2])

##### fitted plot of lag model
plot(fit.sar$fitted.values,fit.sar$y,
     xlab="fitted death rate",ylab="actual death rate",
     main="Fitted plot (with spatial lag term)")
fit22<-lm(fit.sar$y~fit.sar$fitted.values)
abline(a=fit22$coefficients[1],b=fit22$coefficients[2])





#############################
## an example of identify counties of interest
xx = which(wacounty$CNTY == 33)
xx2 = which(wacounty$CNTY == 63)
# plot the whole state
plot(wacounty, border = "#00000075")
# highlight counties of interest
plot(wacounty[xx, ], col = "#ff000075", add = T)
plot(wacounty[xx2, ], col = NA, border = "#0000ff75",
     add = T, lwd = 2.5)
# Add some labels
text(coordinates(wacounty[xx, ]), "King", cex = 0.75,
     pos = 3, offset = 0.25)
text(coordinates(wacounty[xx2, ]), "Spokane", cex = 0.7,
     pos = 1, offset = 0.25)
points(coordinates(wacounty[c(xx, xx2), ]), pch = 16,
       cex = 0.75)
#################################

