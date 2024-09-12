#Nuevo Método de relleno mejorado
##Relleno diario por mes
library(moments)
library(MASS)

rm(list=ls())

setwd("C:/Users/Fernando/Dropbox (CCG-UC)/Carpeta del equipo CCG-UC/Proyectos Actuales/CMPC/Desarrollo/Datos BíoBío/Relleno")

data <- read.csv("Tx_day_86-16_Chile.csv", header = TRUE, stringsAsFactors = FALSE)
data.r <- data

#Parametros del algoritmo
elim <- TRUE #Para eliminar estaciones que no tengan al menos un 70% de la información. Si es true, se dejan solo las estaciones que maximice una matriz de correlaciones no redundante
desest <- FALSE #Detrendizar la serie? (despues se vuelve a agregar la tendencia)
outliers <- FALSE #Desea eliminar los datos atípicos de la serie de datos
comp.e <- FALSE #Añade un componente estocástico a la modelación
var <- "tmax" #Variable a rellenar. Opciones: "tmax", "tmin", "pp"


##desestacionalización
if(desest==TRUE){
    for(h in 5:ncol(data)){
        if(length(which(is.na(data[,h]))) >= nrow(data)-3){#Pasar de largo estaciones que tengan cuatro datos o menos
            next
            }else{
                m.dat <- lm(data[,h]~c(1:nrow(data)))$coefficients[2]
                names(m.dat) <- colnames(data)[h]
                detrnd <- data[,h]-(m.dat*(1:nrow(data))) ##serie detrendizada
                data[,h] <- detrnd
                if(h==5){coefs <- m.dat}else{coefs <- c(coefs, m.dat)}
            }
    }
}else{
    data <- data
}
#vuel <- detrnd + (m.gcm.h*(1:nrow(data))) #reagregar la tendencia a la serie

for(g in 1:12){
    dat <- data[which(data[,3]==g),]
    dat0 <- dat

    #Ordena estaciones desde la más llena a la menos
    lnas <- apply(dat[,5:ncol(dat)],2,function(x)length(which(is.na(x))))
    posi <- match(names(sort(lnas)),colnames(dat)[5:ncol(dat)]) + 4 #4 es por la cantidad de columnas adicionales que no representan estaciones

    dat2 <- dat[,c(1:4,posi)]

    #Eliminar estaciones que no tengan al menos 70% de informacion
    if(elim==TRUE){
        pos <- which(apply(dat2[,5:ncol(dat2)],2,function(x)length(which(!is.na(x))))>=(nrow(dat2)*0.7))+4 #4 es por la cantidad de columnas adicionales que no representan estaciones
        data2 <- dat2[,pos]
    }else{
        eva <- apply(dat2,2,function(x)length(which(is.na(x))))
        #ip <- diff(eva)
        ip <- abs(eva-lm(as.numeric(eva)~c(1:length(eva)))$fitted.values)

        rn <- which(ip==max(ip))+1
        ff <- which(eva>=eva[rn])
        data2 <- dat2[,-c(1:4,ff)]
    }

    ##Remoción de outliers
    if(outliers == TRUE){
        dat <- data2
    for (w in 5:ncol(dat)){
        ploti <- dat[which(!is.na(dat[,w]) & dat[,w] !=0), w] # & dat.m[,w] !=0
        if(length(ploti)<31){next}else{ #Si no se tiene al menos un mes completo no se realiza el procedimiento
            y <- sort(ploti)
            y2 <- y + 2*(abs(min(y)))
            for(j in 1:1000){ #1000 es un valor arbitrario, que esta definido como el entero potencia de 10 inmediatemente mayor al largo del vector evaluado
                x <- sort(rnorm(length(ploti), mean(ploti), sd(ploti)))
                x2 <- x + 2*(abs(min(x)))

                reg1<-lm(y2~x2)

                bc<-boxcox(reg1, plotit = FALSE)
                lambda <- bc$x[which.max(bc$y)]

                bc_transformed<-lm(I(y2^lambda)~x2)

                trans <- bc_transformed$fitted.values

                dd <- data.frame(dat = y2, bc = trans)
                sal <- dd[as.numeric(names(boxplot(dd$bc, plot = FALSE)$out)),1]

                aa <- which(y2 %in% sal)
                if(j==1){df <- aa}else{df <- c(df,aa)}
            }
            fre <- table(df)
            umb <- 500 # 50% de los casos (son 1000 iteraciones) #mean(as.numeric(unname(fre))), #puede ser tambien de 800, o de 770 ya que este ultimo numero no es significativamente de 800 segun prop.test
            vales <- y[as.numeric(names(fre[which(as.numeric(unname(fre)) >= umb)]))]
            if(length(vales)==0){next
            }else{
                aa <- which(dat[,w]%in%vales)
                dat[aa,w] <- NA
            }
        }
    }
    }else{
        dat <- dat
    }

    data3 <- data2

    r <- cor(data2[1:ncol(data2)], use = "pairwise.complete.obs")
    r2 <- 0.7

    for(i in 1:ncol(data3)){
        res <- rep(NA,13)
        names(res) <- c("it_i", "it_j", "Rellenada", "Rellenadora", "Mes", "Falta", "Relleno", "R2", "RMSE", "MAE", "Bias", "pval-KW", "pval-Flig")
        dj <- res
        for(j in c(2:ncol(r))){
            eval<-sort(r[(i),],decreasing=T)[j]
            if(is.na(eval) | length(which(is.na(data2[,i])))==0){next}else{if(eval>=sqrt(r2) & abs(eval) != 1){
                max.r <- names(sort(r[(i),],decreasing=T)[j])
                pos <- which(colnames(data2)==max.r)

                m1<-(lm(data3[,i] ~ data2[,pos]))
                data2$est<-m1$coefficients[1]+data2[,pos]*m1$coefficients[2]

                res <- c(i ,j ,colnames(data2)[i], max.r, g, length(data2[which(is.na(data2[,i])),i]),
                sum(!is.na(data2[which(is.na(data2[,i])),ncol(data2)])), summary(m1)$r.squared,
                sqrt(sum((data2$est - data3[,i])^2, na.rm = TRUE)/(length(which(!is.na(data2$est - data3[,i]))))),
                sum(abs(data2$est - data3[,i]), na.rm = TRUE)/(length(which(!is.na(data2$est - data3[,i])))),
                sum((data2$est - data3[,i]), na.rm = TRUE)/(length(which(!is.na(data2$est - data3[,i])))),
                kruskal.test(c(data3[,i],data2$est),
                             c(rep(1,length(data3[,i])),rep(2,length(data2$est))))$p.value,
                fligner.test(data2$est, data3[,i])$p.value)
                names(res) <- c("it_i", "it_j", "Rellenada", "Rellenadora", "Mes", "Falta", "Relleno", "R2", "RMSE", "MAE", "Bias", "pval-KW", "pval-Flig")

                if(var=="Pp"){
                    data2$est<-ifelse(data2$est<0,0,data2$est)
                    data2$est<-ifelse(data2$est==m1$coefficients[1],0,data2$est)
                }else{data2$est<-data2$est}

                data2[which(is.na(data2[,i])),i] <- data2[which(is.na(data2[,i])),ncol(data2)] ##Reemplazando los datos que faltan en la estacion

                if(j==2){dj<-res}else{dj<-rbind(dj,res)}

            }else{next}
            }
        }
        if(i==1){dj2<-dj}else{dj2<-rbind(dj2,dj)}
    }

    res.f <- dj2[-which(is.na(dj2[,1])),]
    res.f <- unique(res.f)

    if(g==1){DJ <- res.f}else{DJ <- rbind(DJ, res.f)}

    pcol <- match(colnames(data2), colnames(data))
    pcol <- pcol[-length(pcol)]
    prow <- which(data[,3]==g)

    data.r[prow,pcol] <- data2[,-ncol(data2)]

}

write.csv(data.r, "Tx_rellena_daily.csv", row.names = FALSE)
write.csv(DJ, "Reporte_relleno_tx2.csv", row.names = FALSE)





##Boxplot mas histograma para evaluar outliers
layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
par(mar=c(0, 3.1, 1.1, 2.1))
boxplot(ploti , horizontal=TRUE , xaxt="n" , col=rgb(0.8,0.8,0,0.5) , frame=F)
par(mar=c(4, 3.1, 1.1, 2.1))
hist(ploti , breaks=40 , col=rgb(0.2,0.8,0.5,0.5) , border=F , main="" , xlab="value of the variable")





##Relleno no oficial utilizado hasta el momento
rm(list=ls())

ruta <- "C:/Users/Fernando/Dropbox (CCG-UC)/Carpeta del equipo CCG-UC/Proyectos Actuales/CMPC/Desarrollo/Datos BioBio/Relleno"
setwd(ruta)
dir <- paste0(getwd(),"/fin/")

dat <- read.csv("Q_day_86-16_Chile.csv", header = TRUE, stringsAsFactors = FALSE)

dat.m <- aggregate(dat[,4:ncol(dat)], list(dat[,2],dat[,3]),function(x)if(length(which(is.na(x)))>24){NA}else{sum(x)})
#dat.m1 <- dat.m

#dat.m <- dat

#Caso todos los datos
#dat.m <- dat[,-1]
dat.m1 <- dat.m

#Remover outliers
for (w in 3:ncol(dat.m)){
    ploti <- dat.m[which(!is.na(dat.m[,w])), w] # & dat.m[,w] !=0
    if(length(ploti)==0){next}else{
    aa <- which(dat.m[,w] %in% boxplot(ploti, plot = FALSE)$out)
    dat.m1[aa,w] <- NA
}
}

lnas <- apply(dat.m1[,3:ncol(dat.m1)],2,function(x)length(which(is.na(x))))
posi <- match(names(sort(lnas)),colnames(dat.m1)[3:ncol(dat.m1)]) + 2

dat.m2 <- dat.m1[,c(1,2,posi)]

#data<-read.csv("Prueba_pp_qc.csv", header=T,stringsAsFactors=T,sep=",",dec=".") #prueba jp boissier
#data<-read.csv("Prueba_tm_qc.csv", header=T,stringsAsFactors=T,sep=",",dec=".")

data <- dat.m2


#data[data==-9999]<-NA #solo caudales con -9999
#data <- (data1+data2)/2
#data <- data[301:nrow(data),] #Solo para rescatar 1985-2015 en serie que parte el 60
#data <- data2

data2 <- data[,-(1:2)]
pos <- which(apply(data[,3:ncol(data)],2,function(x)length(which(!is.na(x))))>=240)+2
data <- data[,pos]
data2 <- data
data3 <- data2

var <- "q"
r <- cor(data2[1:ncol(data2)], use = "pairwise.complete.obs")
r2 <- 0.7 #parámetro de tolerancia de corrleación

for(i in 1:ncol(data2)){
    for(j in c(2:ncol(r))){
        eval<-sort(r[(i),],decreasing=TRUE)[j]
        if(is.na(eval)){next}else{if(eval>=sqrt(r2) & eval != 1){
            max.r <- names(sort(r[(i),],decreasing=T)[j])
            pos <- which(colnames(data2)==max.r)

            m1<-summary(lm(data3[,i] ~ data2[,pos]))
            data2$est<-m1$coefficients[1]+data2[,pos]*m1$coefficients[2]

            if(var=="Pp"){
                data2$est<-ifelse(data2$est<0,0,data2$est)
                data2$est<-ifelse(data2$est==m1$coefficients[1],0,data2$est)
            }else{data2$est<-data2$est}

            #data2$est <- (data2$est*mean(data2[,i],na.rm=T))/mean(data2$est,na.rm=T)
            data2[which(is.na(data2[,i])),i] <- data2[which(is.na(data2[,i])),ncol(data2)]
            #error<-cbind(data2[,c(1,2)],(data2$est-data[,i]))
            #if(j==2){error1<-error}else{error1[which(is.na(error1[,3])),3]<- error[which(is.na(error1[,3])),3]}
            #colnames(error1)[3]<-colnames(data2)[i]
            #mean(error1[,3],na.rm=T)
        }else{next}

        }
    }

    #if(i==1){errores<-error1}else{errores<-cbind(errores,error1[,3])}
    #colnames(errores)<-colnames(data)[1:i]

    #ifelse(data2[3:(ncol(data2)-1)]>=mean(error1[,3],na.rm=T),
    #      data2[3:(ncol(data2)-1)]-mean(error1[,3],na.rm=T),
    #     data2[3:(ncol(data2)-1)])
    #mean_er<-data.frame(apply(errores[3:ncol(errores)],2,mean,na.rm=T,row.names=T))

}

exp <- data.frame(dat.m1[,1:2], data2[,-(ncol(data2))])

write.csv(exp,paste0(dir,"Q_relleno_mon_r240_86-16_Chile_v2.csv"), row.names=F)


##Segunda parte, relleno diario
rm(list=ls())

setwd("C:/Users/Fernando/Dropbox (CCG-UC)/Carpeta del equipo CCG-UC/Proyectos Actuales/CMPC/Desarrollo/Datos BioBio/")

dir1 <- "/Relleno/fin/"
dir2 <-  "/Datos Estaciones Finales BioBio/"

data <- read.csv(paste0(getwd(),dir2,"Q_BBDD_CCG_BioBio_1986.csv"), header=TRUE, stringsAsFactors = FALSE)
dat <- read.csv(paste0(getwd(),dir1,"Q_relleno_mon_r240_86-16_biobio_v2.csv"), header=TRUE, stringsAsFactors = FALSE)

mn <- unique(dat[,1])
yr <- unique(dat[,2])

#dd1 <- aggregate(data[4:ncol(data)], list(data[,1], data[,2]), mean, na.rm=TRUE)
#dd2 <- aggregate(data[4:ncol(data)], list(data[,1], data[,2]), sd, na.rm=TRUE)

dd1 <- aggregate(data[4:ncol(data)], list(data[,2], data[,3]), mean, na.rm=TRUE)
dd2 <- aggregate(data[4:ncol(data)], list(data[,2], data[,3]), sd, na.rm=TRUE)

dd3 <- aggregate(dat[3:ncol(dat)], list(dat[,2]), sd)

data2 <- data

for(i in 3:ncol(dat)){
    for(j in 1:length(yr)){
       for(k in 1:length(mn)){
            value <- dat[which(dat[,1]==mn[k] & dat[,2]==yr[j]),i]
            vec <- which(dd1[,1]==mn[k])
            posi <- which(abs(value-dd1[vec,i])==min(abs(value-dd1[vec,i]), na.rm=TRUE))[1]
            mes <- dd1[vec[posi],1]
            ano <- dd1[vec[posi],2]

            dat.e <- data[which(data[,2]==mes & data[,3]==ano),i+1]
            if(length(which(is.na(dat.e)))>0){
            dat.e[which(is.na(dat.e))] <- mean(dat.e, na.rm=TRUE)
            }else{
            dat.e <- dat.e
            }
            newvec <- dat.e/(sum(dat.e)/value) #value + (dat.e - mean(dat.e)) temp
            if(length(data2[which(data[,2]==mn[k] & data[,3]==yr[j]),i+1])==length(newvec)){
            data2[which(data[,2]==mn[k] & data[,3]==yr[j]),i+1] <- newvec
            }else{
                veq <- c(newvec, mean(newvec))
                pot <- length(data2[which(data[,2]==mn[k] & data[,3]==yr[j]),i+1])
                data2[which(data[,2]==mn[k] & data[,3]==yr[j]),i+1] <- veq[1:pot]
            }
        }
    }

}


data3 <- data

for(i in 4:ncol(data)){
data3[which(is.na(data[,i])),i] <- data2[which(is.na(data[,i])),i]
}

write.csv(data3, "BioBio_rellenas_q.csv", row.names=FALSE)


##En caudal lo mejor es ocupar la suma para hacer las estimaciones