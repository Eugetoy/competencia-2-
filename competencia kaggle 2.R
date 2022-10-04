rm(list = ls())
gc(verbose = FALSE)

install.packages('data.table')
install.packages('rlist')
install.packages('xgboost')
install.packages('DiceKriging')
install.packages('mlrMBO')
install.packages('mlr')  
install.packages('dplyr')
library(mlr)
library(mlrMBO)#estoo
library(dplyr)
library(caret) 
library(data.table)
library(rlist)
library(DiceKriging)
library(xgboost)

kBO_iter  <- 100   #cantidad de iteraciones de la Optimizacion Bayesiana

#Aqui se cargan los hiperparametros
hs <- makeParamSet( 
  makeNumericParam("eta",              lower=  0.01 , upper=    0.3),   #equivalente a learning rate
  makeNumericParam("colsample_bytree", lower=  0.2  , upper=    1.0),   #equivalente a feature_fraction
  makeIntegerParam("min_child_weight", lower=  0L   , upper=   10L),    #groseramente equivalente a  min_data_in_leaf
  makeIntegerParam("max_depth",        lower=  2L   , upper=   30L),    #profundidad del arbol, NO es equivalente a num_leaves
  makeNumericParam("prob_corte",       lower= 1/80 , upper=  1/15)     #pruebo  cortar con otras probabilidades
)

ksemilla_azar  <- 469363  #Aqui poner la propia semilla

#semillas <- c(469363,502133,621923,704017,839369)

#------------------------------------------------------------------------------
#graba a un archivo los componentes de lista
#para el primer registro, escribe antes los titulos

loguear  <- function( reg, arch=NA, folder="./exp/", ext=".txt", verbose=TRUE )
{
  archivo  <- arch
  if( is.na(arch) )  archivo  <- paste0(  folder, substitute( reg), ext )
  
  if( !file.exists( archivo ) )  #Escribo los titulos
  {
    linea  <- paste0( "fecha\t", 
                      paste( list.names(reg), collapse="\t" ), "\n" )
    
    cat( linea, file=archivo )
  }
  
  linea  <- paste0( format(Sys.time(), "%Y%m%d %H%M%S"),  "\t",     #la fecha y hora
                    gsub( ", ", "\t", toString( reg ) ),  "\n" )
  
  cat( linea, file=archivo, append=TRUE )  #grabo al archivo
  
  if( verbose )  cat( linea )   #imprimo por pantalla
}
#------------------------------------------------------------------------------
#esta funcion calcula internamente la ganancia de la prediccion probs

SCORE_PCORTE  <- log( 1/40 / ( 1 - 1/40 ) )   #esto hace falta en ESTA version del XGBoost ... misterio por ahora ...

fganancia_logistic_xgboost   <- function( scores, datos) 
{
  vlabels  <- getinfo( datos, "label")
  
  gan  <- sum( ( scores > SCORE_PCORTE  ) *
                 ifelse( vlabels== 1, 78000, -2000 ) )
  
  
  return(  list("metric" = "ganancia", "value" = gan ) )
}
#------------------------------------------------------------------------------
#esta funcion solo puede recibir los parametros que se estan optimizando
#el resto de los parametros se pasan como variables globales, la semilla del mal ...

EstimarGanancia_xgboost  <- function( x )
{
  gc()  #libero memoria
  
  #llevo el registro de la iteracion por la que voy
  GLOBAL_iteracion  <<- GLOBAL_iteracion + 1
  
  SCORE_PCORTE  <<- log( x$prob_corte / ( 1 - x$prob_corte ) ) 
  
  kfolds  <- 5   # cantidad de folds para cross validation
  
  #otros hiperparmetros, que por ahora dejo en su valor default
  param_basicos  <- list( gamma=                0.0,  #por ahora, lo dejo fijo, equivalente a  min_gain_to_split
                          alpha=                0.0,  #por ahora, lo dejo fijo, equivalente a  lambda_l1
                          lambda=               0.0,  #por ahora, lo dejo fijo, equivalente a  lambda_l2
                          subsample=            1.0,  #por ahora, lo dejo fijo
                          tree_method=       "auto",  #por ahora lo dejo fijo, pero ya lo voy a cambiar a "hist"
                          grow_policy=  "depthwise",  #ya lo voy a cambiar a "lossguide"
                          max_bin=            256,    #por ahora fijo
                          max_leaves=           0,    #ya lo voy a cambiar
                          scale_pos_weight=     1.0   #por ahora, lo dejo fijo
  )
  
  param_completo  <- c( param_basicos, x )
  
  set.seed( 469363 )
  modelocv  <- xgb.cv( objective= "binary:logistic",
                       data= dtrain,
                       feval= fganancia_logistic_xgboost,
                       disable_default_eval_metric= TRUE,
                       maximize= TRUE,
                       stratified= TRUE,     #sobre el cross validation
                       nfold= kfolds,        #folds del cross validation
                       nrounds= 9999,        #un numero muy grande, lo limita early_stopping_rounds
                       early_stopping_rounds= as.integer(50 + 5/x$eta),
                       base_score= mean( getinfo(dtrain, "label")),  
                       param= param_completo,
                       verbose= -100
  )
  
  #obtengo la ganancia
  ganancia   <- unlist( modelocv$evaluation_log[ , test_ganancia_mean] )[ modelocv$best_iter ] 
  
  ganancia_normalizada  <- ganancia* kfolds     #normailizo la ganancia
  
  #el lenguaje R permite asignarle ATRIBUTOS a cualquier variable
  attr(ganancia_normalizada ,"extras" )  <- list("nrounds"= modelocv$best_iter)  #esta es la forma de devolver un parametro extra
  
  param_completo$nrounds <- modelocv$best_iter  #asigno el mejor nrounds
  param_completo["early_stopping_rounds"]  <- NULL     #elimino de la lista el componente  "early_stopping_rounds"
  
  #logueo 
  xx  <- param_completo
  xx$ganancia  <- ganancia_normalizada   #le agrego la ganancia
  xx$iteracion <- GLOBAL_iteracion
  loguear( xx, arch= klog )
  
  return( ganancia )
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui empieza el programa

#Aqui se debe poner la carpeta de la computadora local
setwd("C:\\Users\\Euge\\OneDrive\\Escritorio\\EspecilidadDM\\economiayfinanzas")

#cargo el dataset donde voy a entrenar el modelo
dataset  <- fread("./datasets/competencia2_2022.csv")

muestra = sample(1:nrow(dataset),size= 300000,replace=FALSE) # me quedo sólo con 10000 filas aleatorias
pruebamuestra = dataset[muestra, ]
head(pruebamuestra)
rm(dataset)
dataset =pruebamuestra

#creo la carpeta donde va el experimento
# HT  representa  Hiperparameter Tuning
dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( "./exp/HT7630/", showWarnings = FALSE )
setwd("./exp/HT7630/")   #Establezco el Working Directory DEL EXPERIMENTO


#en estos archivos quedan los resultados
kbayesiana  <- "HT7631.RDATA"
klog        <- "HT7631.txt"


GLOBAL_iteracion  <- 0   #inicializo la variable global

#si ya existe el archivo log, traigo hasta donde llegue
if( file.exists(klog) )
{
  tabla_log  <- fread( klog )
  GLOBAL_iteracion  <- nrow( tabla_log )
}


#paso la clase a binaria que tome valores {0,1}  enteros
dataset[ foto_mes==202103 , clase01 := ifelse( clase_ternaria=="BAJA+2", 1L, 0L) ]


#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01") )

#dejo los datos en el formato que necesita LightGBM
dtrain  <- xgb.DMatrix( data=  data.matrix(  dataset[ foto_mes==202103, campos_buenos, with=FALSE]),
                        label= dataset[ foto_mes==202103, clase01 ] )


#Aqui comienza la configuracion de la Bayesian Optimization
funcion_optimizar  <- EstimarGanancia_xgboost   #la funcion que voy a maximizar

configureMlr( show.learner.output= FALSE)

#configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
#por favor, no desesperarse por lo complejo

obj.fun  <- makeSingleObjectiveFunction(
  fn=       funcion_optimizar, #la funcion que voy a maximizar
  minimize= FALSE,   #estoy Maximizando la ganancia
  noisy=    TRUE,
  par.set=  hs,     #definido al comienzo del programa
  has.simple.signature = FALSE   #paso los parametros en una lista
)

ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  save.file.path= kbayesiana)  #se graba cada 600 segundos
ctrl  <- setMBOControlTermination(ctrl, iters= kBO_iter )   #cantidad de iteraciones
ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI() )

#establezco la funcion que busca el maximo
surr.km  <- makeLearner("regr.km", predict.type= "se", covtype= "matern3_2", control= list(trace= TRUE))

#inicio la optimizacion bayesiana
if( !file.exists( kbayesiana ) ) {
  run  <- mbo(obj.fun, learner= surr.km, control= ctrl)
} else {
  run  <- mboContinue( kbayesiana )   #retomo en caso que ya exista
}


# Librerías necesarias
install.packages(c('data.table', 'rpart', 'ggplot2', 'lightgbm','xgboost' ))
install.packages("DiagrammeR")
library(data.table)
library(lightgbm)
library(xgboost)
library(DiagrammeR)
library(rpart)

# Poner la carpeta de la materia de SU computadora local
setwd("C:\\Users\\Euge\\OneDrive\\Escritorio\\EspecilidadDM\\economiayfinanzas")

semillas <- c(469363,502133,621923,704017,839369)

# Cargamos los datasets 
dataset <- fread("./datasets/competencia2_2022.csv")

#INICIO de la seccion donde se deben hacer cambios con variables nuevas

#creo un ctr_quarter que tenga en cuenta cuando los clientes hace 3 menos meses que estan
dataset[  , ctrx_quarter_normalizado := ctrx_quarter ]
dataset[ cliente_antiguedad==1 , ctrx_quarter_normalizado := ctrx_quarter * 5 ]
dataset[ cliente_antiguedad==2 , ctrx_quarter_normalizado := ctrx_quarter * 2 ]
dataset[ cliente_antiguedad==3 , ctrx_quarter_normalizado := ctrx_quarter * 1.2 ]

#variable extraida de una tesis de maestria de Irlanda
dataset[  , mpayroll_sobre_edad  := mpayroll / cliente_edad ]

#se crean los nuevos campos para MasterCard  y Visa, teniendo en cuenta los NA's
#varias formas de combinar Visa_status y Master_status
dataset[ , mv_status01       := pmax( Master_status,  Visa_status, na.rm = TRUE) ]
dataset[ , mv_status02       := Master_status +  Visa_status ]
dataset[ , mv_status03       := pmax( ifelse( is.na(Master_status), 10, Master_status) , ifelse( is.na(Visa_status), 10, Visa_status) ) ]
dataset[ , mv_status04       := ifelse( is.na(Master_status), 10, Master_status)  +  ifelse( is.na(Visa_status), 10, Visa_status)  ]
dataset[ , mv_status05       := ifelse( is.na(Master_status), 10, Master_status)  +  100*ifelse( is.na(Visa_status), 10, Visa_status)  ]

dataset[ , mv_status06       := ifelse( is.na(Visa_status), 
                                        ifelse( is.na(Master_status), 10, Master_status), 
                                        Visa_status)  ]

dataset[ , mv_status07       := ifelse( is.na(Master_status), 
                                        ifelse( is.na(Visa_status), 10, Visa_status), 
                                        Master_status)  ]


#combino MasterCard y Visa
dataset[ , mv_mfinanciacion_limite := rowSums( cbind( Master_mfinanciacion_limite,  Visa_mfinanciacion_limite) , na.rm=TRUE ) ]

dataset[ , mv_Fvencimiento         := pmin( Master_Fvencimiento, Visa_Fvencimiento, na.rm = TRUE) ]
dataset[ , mv_Finiciomora          := pmin( Master_Finiciomora, Visa_Finiciomora, na.rm = TRUE) ]
dataset[ , mv_msaldototal          := rowSums( cbind( Master_msaldototal,  Visa_msaldototal) , na.rm=TRUE ) ]
dataset[ , mv_msaldopesos          := rowSums( cbind( Master_msaldopesos,  Visa_msaldopesos) , na.rm=TRUE ) ]
dataset[ , mv_msaldodolares        := rowSums( cbind( Master_msaldodolares,  Visa_msaldodolares) , na.rm=TRUE ) ]
dataset[ , mv_mconsumospesos       := rowSums( cbind( Master_mconsumospesos,  Visa_mconsumospesos) , na.rm=TRUE ) ]
dataset[ , mv_mconsumosdolares     := rowSums( cbind( Master_mconsumosdolares,  Visa_mconsumosdolares) , na.rm=TRUE ) ]
dataset[ , mv_mlimitecompra        := rowSums( cbind( Master_mlimitecompra,  Visa_mlimitecompra) , na.rm=TRUE ) ]
dataset[ , mv_madelantopesos       := rowSums( cbind( Master_madelantopesos,  Visa_madelantopesos) , na.rm=TRUE ) ]
dataset[ , mv_madelantodolares     := rowSums( cbind( Master_madelantodolares,  Visa_madelantodolares) , na.rm=TRUE ) ]
dataset[ , mv_fultimo_cierre       := pmax( Master_fultimo_cierre, Visa_fultimo_cierre, na.rm = TRUE) ]
dataset[ , mv_mpagado              := rowSums( cbind( Master_mpagado,  Visa_mpagado) , na.rm=TRUE ) ]
dataset[ , mv_mpagospesos          := rowSums( cbind( Master_mpagospesos,  Visa_mpagospesos) , na.rm=TRUE ) ]
dataset[ , mv_mpagosdolares        := rowSums( cbind( Master_mpagosdolares,  Visa_mpagosdolares) , na.rm=TRUE ) ]
dataset[ , mv_fechaalta            := pmax( Master_fechaalta, Visa_fechaalta, na.rm = TRUE) ]
dataset[ , mv_mconsumototal        := rowSums( cbind( Master_mconsumototal,  Visa_mconsumototal) , na.rm=TRUE ) ]
dataset[ , mv_cconsumos            := rowSums( cbind( Master_cconsumos,  Visa_cconsumos) , na.rm=TRUE ) ]
dataset[ , mv_cadelantosefectivo   := rowSums( cbind( Master_cadelantosefectivo,  Visa_cadelantosefectivo) , na.rm=TRUE ) ]
dataset[ , mv_mpagominimo          := rowSums( cbind( Master_mpagominimo,  Visa_mpagominimo) , na.rm=TRUE ) ]

#elimino variables con datadifrting
dataset$Master_mfinanciacion_limite <- NULL
dataset$mcuenta_debitos_automaticos <- NULL
dataset$Visa_mpagosdolares <- NULL
dataset$Visa_mpagado<- NULL
dataset$mcuenta_corriente <- NULL
dataset$mcomisiones_otras <- NULL

#valvula de seguridad para evitar valores infinitos
#paso los infinitos a NULOS
infinitos      <- lapply(names(dataset),function(.name) dataset[ , sum(is.infinite(get(.name)))])
infinitos_qty  <- sum( unlist( infinitos) )
if( infinitos_qty > 0 )
{
  cat( "ATENCION, hay", infinitos_qty, "valores infinitos en tu dataset. Seran pasados a NA\n" )
  dataset[mapply(is.infinite, dataset)] <- NA
}

#valvula de seguridad para evitar valores NaN  que es 0/0
#paso los NaN a 0 , decision polemica si las hay
#se invita a asignar un valor razonable segun la semantica del campo creado
nans      <- lapply(names(dataset),function(.name) dataset[ , sum(is.nan(get(.name)))])
nans_qty  <- sum( unlist( nans) )
if( nans_qty > 0 )
{
  cat( "ATENCION, hay", nans_qty, "valores NaN 0/0 en tu dataset. Seran pasados arbitrariamente a 0\n" )
  cat( "Si no te gusta la decision, modifica a gusto el programa!\n\n")
  dataset[mapply(is.nan, dataset)] <- 0
}

##Me quedo con marzo y mayo 
marzo <- dataset[foto_mes == 202103]
mayo <- dataset[foto_mes == 202105]

## Feauter engeeniring##
marzo[ , campo3 := as.integer( ctrx_quarter <14 & mcuentas_saldo>= -1256.1 & mcaja_ahorro <2601.1 ) ]
marzo[ , campo4 := as.integer( ctrx_quarter <14 & mcuentas_saldo>= -1256.1 & mcaja_ahorro>=2601.1 ) ]

marzo[ , campo5 := as.integer( ctrx_quarter>=14 & ( Visa_status>=8 | is.na(Visa_status) ) & ( Master_status>=8 | is.na(Master_status) ) ) ]
marzo[ , campo6 := as.integer( ctrx_quarter>=14 & ( Visa_status>=8 | is.na(Visa_status) ) & ( Master_status <8 & !is.na(Master_status) ) ) ]

marzo[ , campo7 := as.integer( ctrx_quarter>=14 & Visa_status <8 & !is.na(Visa_status) & ctrx_quarter <38 ) ]
marzo[ , campo8 := as.integer( ctrx_quarter>=14 & Visa_status <8 & !is.na(Visa_status) & ctrx_quarter>=38 ) ]

marzo[, ctrx_quarter := log(ctrx_quarter + 1)]
summary(marzo$ctrx_quarter)    

marzo[, active_quarter := log(active_quarter  + 1)]
summary(marzo$active_quarter) 

mayo[ , campo3 := as.integer( ctrx_quarter <14 & mcuentas_saldo>= -1256.1 & mcaja_ahorro <2601.1 ) ]
mayo[ , campo4 := as.integer( ctrx_quarter <14 & mcuentas_saldo>= -1256.1 & mcaja_ahorro>=2601.1 ) ]

mayo[ , campo5 := as.integer( ctrx_quarter>=14 & ( Visa_status>=8 | is.na(Visa_status) ) & ( Master_status>=8 | is.na(Master_status) ) ) ]
mayo[ , campo6 := as.integer( ctrx_quarter>=14 & ( Visa_status>=8 | is.na(Visa_status) ) & ( Master_status <8 & !is.na(Master_status) ) ) ]

mayo[ , campo7 := as.integer( ctrx_quarter>=14 & Visa_status <8 & !is.na(Visa_status) & ctrx_quarter <38 ) ]
mayo[ , campo8 := as.integer( ctrx_quarter>=14 & Visa_status <8 & !is.na(Visa_status) & ctrx_quarter>=38 ) ]

mayo[, ctrx_quarter := log(ctrx_quarter + 1)]
summary(mayo$ctrx_quarter)    

mayo[, active_quarter := log(active_quarter  + 1)]
summary(mayo$active_quarter) 

library(dplyr)
marzo[, ctrx_quarter := ntile(ctrx_quarter, 10)]
marzo[, mprestamos_personales := ntile(mprestamos_personales, 10)]
marzo[, mcuentas_saldo := ntile(mcuentas_saldo, 10)]
marzo[, mactivos_margen := ntile(mactivos_margen, 10)]
marzo[, mcaja_ahorro := ntile(mcaja_ahorro, 10)]
marzo[, mcomisiones_mantenimiento := ntile(mcomisiones_mantenimiento, 10)]
marzo[, mrentabilidad := ntile(mrentabilidad, 10)]
marzo[, mpasivos_margen := ntile(mpasivos_margen, 10)]
marzo[, Visa_mlimitecompra := ntile(Visa_mlimitecompra, 10)]
marzo[, mrentabilidad_annual := ntile(mrentabilidad_annual, 10)]
marzo[, Master_mlimitecompra := ntile(Master_mlimitecompra, 10)]

mayo[, ctrx_quarter := ntile(ctrx_quarter, 10)]
mayo[, mprestamos_personales := ntile(mprestamos_personales, 10)]
mayo[, mcuentas_saldo := ntile(mcuentas_saldo, 10)]
mayo[, mactivos_margen := ntile(mactivos_margen, 10)]
mayo[, mcaja_ahorro := ntile(mcaja_ahorro, 10)]
mayo[, mcomisiones_mantenimiento := ntile(mcomisiones_mantenimiento, 10)]
mayo[, mrentabilidad := ntile(mrentabilidad, 10)]
mayo[, mpasivos_margen := ntile(mpasivos_margen, 10)]
mayo[, Visa_mlimitecompra := ntile(Visa_mlimitecompra, 10)]
mayo[, mrentabilidad_annual := ntile(mrentabilidad_annual, 10)]
mayo[, Master_mlimitecompra := ntile(Master_mlimitecompra, 10)]


#rm(dataset)# saca Dataset del global enviorment

# Clase BAJA+1 y BAJA+2 juntas
#clase_binaria <- ifelse(marzo$clase_ternaria == "CONTINUA", 0, 1)
#clase_real <- marzo$clase_ternaria
#marzo$clase_ternaria <- NULL
#mayo$clase_ternaria <- NULL

#VER ESTO

#paso la clase a binaria que tome valores {0,1}  enteros
dataset[ , clase01 := ifelse( clase_ternaria=="BAJA+2", 1L, 0L) ]

#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01") )

## ---------------------------
## Step 2: XGBoost
## ---------------------------

#dtrain <- xgb.DMatrix(
#  data = data.matrix(marzo),
#  label = clase_binaria, missing = NA)

#dejo los datos en el formato que necesita XGBoost
dtrain  <- xgb.DMatrix( data= data.matrix(  dataset[ foto_mes==202103 , campos_buenos, with=FALSE]),
                        label= dataset[ foto_mes==202103, clase01] )

#genero el modelo con los parametros por default (Gustavo)
set.seed(semillas[1])

modelo  <- xgb.train( data= dtrain,
                      param= list( objective= "binary:logistic",
                                   max_depth= 12,    #arboles de altura 1, solo dos hojas!Cambiar???
                                   min_child_weight= 6,
                                   eta = 0.2, #eta=0.1? 
                                   colsample_bynode = 0.999, #0.999
                                   #colsample_bytree=0.952 , 
                                   learning_rate = 1 ,#antes 1 
                                   num_parallel_tree = 50, # antes 10-50<--- IMPORTANTE CAMBIAR
                                   subsample = 0.8),
                      nrounds= 100 # en vez de 400
)


#99: eta=0.291; colsample_bytree=0.534; 
#min_child_weight=2; 
#max_depth=19; 
#prob_corte=0.0271 : y = 4.4e+03 : 2.5 secs : infill_ei


#4[mbo] 100: eta=0.0102; 
#colsample_bytree=0.952; 
#min_child_weight=10; 
#max_depth=6; prob_corte=0.0254 : y = 1.8e+06 : 1192.5 secs : infill_ei

#3: eta=0.1 y col sample by node 0.999

#[mbo]2 100: eta=0.0108; 
#colsample_bytree=0.999; -> feauter fraction
#min_child_weight=7; 
#max_depth=2; 
#prob_corte=0.0233 : 
#y = 8.08e+05 : 229.5 secs : infill_ei

#[mbo]1 
#99: eta=0.291; colsample_bytree=0.534; 
#min_child_weight=2; 
#max_depth=19; 
#prob_corte=0.0271 : y = 4.4e+03 : 2.5 secs : infill_ei

#---------------------------------------------------------------------

#aplico el modelo a los datos nuevos
prediccion  <- predict( modelo, 
                        data.matrix( dataset[ foto_mes==202105, campos_buenos, with=FALSE ]) )


#Genero la entrega para Kaggle
entrega  <- as.data.table( list( "numero_de_cliente"= dataset[ foto_mes==202105 , numero_de_cliente],
                                 "Predicted"= as.integer( prediccion > 1/40) )  ) #genero la salida

dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( "./exp/KA7530/", showWarnings = FALSE )
archivo_salida  <- "./exp/KA7530/KA7530_007.csv"

#genero el archivo para Kaggle
fwrite( entrega, 
        file= archivo_salida, 
        sep= "," )


