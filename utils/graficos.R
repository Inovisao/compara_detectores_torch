library("ggplot2")
library("gridExtra")
library("plyr")
library("scales")
library("dplyr")
library(tidyr)
library("Metrics")
library(data.table)

options(scipen = 999)

# -------------------------------------------------------------------
# BOXPLOT DO DESEMPENHO ENTRE TÉCNICAS
#
dados <- read.table('../results/results.csv', sep = ',', header = TRUE)

# Reordena a coluna 'ml' para garantir que o primeiro elemento seja o primeiro boxplot
dados$ml <- factor(dados$ml, levels = unique(dados$ml))

metricas <- list("mAP50", "mAP75", "mAP", "precision", "recall", "fscore", "MAE", "RMSE", "r")
graficos <- list()
i <- 1

for (metrica in metricas) {
   print(metrica)
   TITULO = sprintf("Boxplot for %s", metrica)
   g <- ggplot(dados, aes_string(x = "ml", y = metrica, fill = "ml")) + 
        geom_boxplot() +
        scale_fill_brewer(palette = "Purples") +
        labs(title = TITULO, x = "Models", y = metrica) +
        theme(legend.position = "none") +
        theme(plot.title = element_text(hjust = 0.5))
   
   graficos[[i]] <- g
   i = i + 1
}

g <- grid.arrange(grobs = graficos, ncol = 3)
ggsave(paste("../results/boxplot.png", sep = ""), g, width = 12, height = 10)
print(g)

# -------------------------------------------------------------------
# XY CONTAGEM MANUAL X AUTOMÁTICA - JUNTANDO TODAS AS DOBRAS
#
dadosContagem <- read.table('../results/counting.csv', sep = ',', header = TRUE)

dadosContagem$ml <- factor(dadosContagem$ml, levels = unique(dadosContagem$ml))

graficos <- list()
i <- 1

nets <- levels(as.factor(dadosContagem$ml))
print(nets)

# Coletando RMSE para ANOVA
rmse_values <- data.frame(Model = character(), RMSE = numeric(), stringsAsFactors = FALSE)

for (net in nets) {
   filtrado <- dadosContagem[dadosContagem$ml == net, ]

   RMSE = rmse(filtrado$groundtruth, filtrado$predicted)
   MAE = mae(filtrado$groundtruth, filtrado$predicted)
   MAPE = mape(filtrado$groundtruth, filtrado$predicted)
   R = cor(filtrado$groundtruth, filtrado$predicted, method = "pearson")
   TITULO = sprintf("%s RMSE=%.3f MAE=%.3f MAPE=%.3f r = %.3f", net, RMSE, MAE, MAPE, R)
   MAX <- max(filtrado$groundtruth, filtrado$predicted)
   
   g <- ggplot(filtrado, aes(x = groundtruth, y = predicted)) + 
        geom_point() +
        geom_smooth(method = 'lm', se = TRUE) +  # Linha de regressão linear
        labs(title = TITULO, x = "Contagem Manual (Ground Truth)", y = "Contagem Preditiva") + 
        theme(plot.title = element_text(size = 10)) +
        xlim(0, MAX) +
        ylim(0, MAX)

   print(g)
   graficos[[i]] <- g
   i = i + 1

   # Salvando RMSE para ANOVA
   rmse_values <- rbind(rmse_values, data.frame(Model = net, RMSE = RMSE))
}

g <- grid.arrange(grobs = graficos, ncol = 2)
ggsave(paste("../results/counting.png", sep = ""), g, width = 10, height = 12)
print(g)

# Arquivo para salvar os resultados
output_file <- '../results/anova_all_results.txt'

# Limpar o arquivo antes de começar
sink(output_file)
sink()
# Instalar pacotes necessários, se ainda não estiverem instalados
if (!require(multcomp)) install.packages("multcomp", dependencies = TRUE)
if (!require(multcompView)) install.packages("multcompView", dependencies = TRUE)

# Carregar os pacotes
library(multcomp)
library(multcompView)

# -------------------------------------------------------------------
# Função para realizar ANOVA para cada métrica
realizar_anova <- function(df, metric, output_file) {
  anova_result <- tryCatch({
    aov(as.formula(paste(metric, "~ ml")), data = df)
  }, error = function(e) {
    message("Erro ao realizar ANOVA para ", metric, ": ", e)
    return(NULL)
  })

  if (!is.null(anova_result)) {
    anova_summary <- summary(anova_result)

    # Abrir o arquivo para adicionar os resultados
    sink(output_file, append = TRUE)
    cat("\n------------------------------------------------------------\n")
    cat("ANOVA para", metric, "\n")
    print(anova_summary)

    # Se o resultado da ANOVA for significativo, realizar o pós-teste de Tukey
    if ("Pr(>F)" %in% colnames(anova_summary[[1]]) && anova_summary[[1]][["Pr(>F)"]][1] < 0.05) {
      tukey_result <- TukeyHSD(anova_result)
      cat("\nTukey HSD para", metric, "\n")
      print(tukey_result)

      # Obter os resultados do Tukey HSD
      tukey_df <- as.data.frame(tukey_result[[1]])
      
      # Adicionar o CLD
      cld_result <- cld(glht(anova_result, linfct = mcp(ml = "Tukey")))
      cat("\nCLD para", metric, "\n")
      print(cld_result)
    }

    # Fechar a escrita no arquivo
    sink()
  } else {
    message("ANOVA não pôde ser realizada para ", metric)
  }
}

# -------------------------------------------------------------------
# Realizar ANOVA para cada métrica
metrics <- c("mAP", "mAP50", "mAP75", "MAE", "RMSE", "r", "precision", "recall", "fscore")

for (metric in metrics) {
  realizar_anova(dados, metric, output_file)
}

# Mensagem final
cat("Os resultados foram salvos no arquivo:", output_file, "\n")